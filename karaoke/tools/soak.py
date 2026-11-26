"""Reproducible whole-song acceptance runs through the room's song queue."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from karaoke.job_schema import playback_stems, provider_error, retry_fields, video_url
from karaoke.local_asr import offline

REQUIRED = {'mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'subtitles',
            'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'}
GROUPS = {
    'Chinese': (40, ['周杰倫', '五月天', '林俊傑', '孫燕姿', '蔡依林', '鄧紫棋',
                     '張惠妹', '梁靜茹', '田馥甄', '李榮浩', '鳳凰傳奇', '周深']),
    'Japanese': (30, ['米津玄師', 'YOASOBI', 'Aimer', '宇多田ヒカル', 'Official髭男dism',
                      'King Gnu', 'あいみょん', 'ヨルシカ', 'Mrs. GREEN APPLE']),
    'English': (30, ['Adele', 'Ed Sheeran', 'Taylor Swift', 'Coldplay', 'Bruno Mars',
                     'Billie Eilish', 'Dua Lipa', 'Imagine Dragons', 'Linkin Park', 'Eminem']),
}


def stamp():
    return datetime.now(UTC).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def select_cases(pool, quotas, seed):
    """Freeze selection before any processing outcome is known."""
    rng, seen, chosen = random.Random(seed), set(), []
    for language, count in quotas.items():
        candidates = sorted((item for item in pool if item['language'] == language),
                            key=lambda item: item['url'])
        rng.shuffle(candidates)
        selected = []
        for item in candidates:
            url = video_url(item['url'])
            # Brackets often enclose the actual song title, so retain their contents.
            title = re.sub(r'official|music video|lyric video|lyrics|video|audio|ultrahd|高清|高畫質|完整版|官方|華納|mv|hd|4k', '', item['title'], flags=re.I)
            if item['language'] == 'Chinese':
                chinese = ''.join(re.findall(r'[\u3400-\u9fff]', title))
                artist = (item.get('query') or '').split(' official')[0]
                chinese = chinese.replace(artist, '')
                if chinese:
                    title = artist+chinese
            key = ''.join(c for c in title.casefold() if c.isalnum())
            if url in seen or key in seen:
                continue
            seen.update((url, key))
            selected.append(dict(item, url=url, attempts=[]))
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'{language}: only {len(selected)} distinct candidates for {count} slots')
        chosen.extend(selected)
    rng.shuffle(chosen)
    return chosen


def sample(path, data_dir, seed):
    from karaoke.video_search import matches_query, youtube_entries
    if path.exists():
        raise ValueError('The fixed manifest already exists; refusing to replace its cases')
    pool_path = path.with_name('candidate-pool.json')
    pool: dict[str, Any] = json.loads(pool_path.read_text()) if pool_path.exists() else {'queries': {}, 'items': []}
    with sqlite3.connect(f'file:{data_dir}/jobs.sqlite3?mode=ro', uri=True) as db:
        excluded = {json.loads(row[0])['input']['url'] for row in db.execute('SELECT doc FROM jobs')}
    for lyrics in (data_dir.parent/'lyrics').glob('*.json'):
        value = json.loads(lyrics.read_text()).get('source_url')
        if value:
            try:
                excluded.add(video_url(value))
            except ValueError:  # Older research sources can be local files or lyric sites.
                pass
    for language, (_, artists) in GROUPS.items():
        for artist in artists:
            query = artist + ' official music video'
            if query in pool['queries']:
                continue
            items = youtube_entries(query)['results']
            added = 0
            for item in items:
                evidence = item.get('source_evidence')
                duration = item.get('duration') or 0
                if (not evidence or not matches_query(artist, item) or not 90 <= duration <= 480
                        or item.get('live_status') in ('is_live', 'is_upcoming')):
                    continue
                url = video_url(item['url'])
                if url in excluded or any(x['url'] == url for x in pool['items']):
                    continue
                pool['items'].append(dict(url=url, title=item['title'], language=language,
                    duration=duration, channel=item.get('uploader'), evidence=evidence,
                    provider='youtube', query=query))
                added += 1
            pool['queries'][query] = dict(at=stamp(), added=added)
            save(pool_path, pool)
            print(json.dumps({'query': query, 'added': added}, ensure_ascii=False), flush=True)
            time.sleep(1)
    queries = {artist+' official music video' for _, artists in GROUPS.values() for artist in artists}
    cases = select_cases([item for item in pool['items'] if item['query'] in queries],
                         {k: v[0] for k, v in GROUPS.items()}, seed)
    selection = [dict(url=c['url'], language=c['language']) for c in cases]
    save(path, dict(schema_version=1, seed=seed, created_at=stamp(),
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
        cache_limit_bytes=10*1024**3, cases=cases, provider_backoff={}))


def eligible(case, version, now):
    attempts = case.get('attempts', [])
    if not attempts:
        return True
    last = attempts[-1]
    if last['status'] in ('submitted', 'running', 'blocked'):
        return False
    if last['status'] == 'external':
        return now >= last['next_retry_at'] and last.get('external_attempt', 1) < 3
    if last['status'] == 'cache_evicted':
        return True
    # A software failure is retried only after a fix. Prior passes must also
    # pass on the final deployed version, so fixes are exercised together.
    return last['pipeline_version'] != version


def retry_deadline(state, case):
    # Song metadata is optional; its text-model quota must not block ASR.
    required = (case['provider'], 'openai', 'openai:whisper-1',
                'openai:'+os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe'))
    if offline():
        required = (case['provider'],)
    return max((state['provider_backoff'].get(key, 0) for key in required), default=0)


def resolved_audio_failure(case, version):
    last = (case.get('audio_attempts') or [{}])[-1]
    if last.get('status') not in ('blocked', 'external', 'failed'):
        return False
    error = last.get('error') or {}
    if offline() and error.get('provider') == 'openai' and last.get('pipeline_version') != version:
        return True
    download = (case.get('download_attempts') or [{}])[-1]
    return (error.get('code') == 'download_failed' and download.get('status') == 'downloaded_audio'
            and bool(download.get('finished_at')) and bool(last.get('finished_at'))
            and datetime.fromisoformat(download['finished_at']) > datetime.fromisoformat(last['finished_at']))


def verify_stems(paths, duration, *, storage=None):
    import numpy as np
    import soundfile as sf
    with ExitStack() as stack:
        files = [stack.enter_context(sf.SoundFile(paths[name]))
                 for name in playback_stems(paths)]
        formats = {(f.samplerate, f.channels, len(f), f.subtype) for f in files}
        compact = storage is not None
        if compact and (storage.get('format') != 'FLAC' or storage.get('subtype') != 'PCM_24' or
                        not 0 < storage.get('source_gain', 0) <= 1):
            raise ValueError('Invalid compact audio storage description')
        if (len(formats) != 1 or not len(files[0]) or
                any(f.subtype != ('PCM_24' if compact else 'DOUBLE') or
                    f.format != ('FLAC' if compact else 'WAV') for f in files)):
            raise ValueError('Stems have incompatible master formats')
        rate, channels, frames, _ = formats.pop()
        if abs(frames/rate-duration) > 2/min(rate, 44100):
            raise ValueError('Stem duration differs from the complete source')
        while True:
            blocks = [f.read(65536, dtype='float64', always_2d=True) for f in files]
            if not len(blocks[0]):
                break
            if any(not np.isfinite(block).all() for block in blocks):
                raise ValueError('Non-finite stem samples')
    return dict(sample_rate=rate, channels=channels, frames=frames)


def verify_outputs(client, base, data_dir, job, *, audio_only=False, playback_policy=False):
    from karaoke.subtitles import collapsed_phrase, highlight_coverage_errors, normalized
    folder = (data_dir/job['id']).resolve()
    result = json.loads((folder/('audio-result.json' if audio_only else 'result.json')).read_text())
    required = REQUIRED-{'mv'} if audio_only else REQUIRED
    if result.get('audio_storage', {}).get('format') == 'FLAC':
        required = {name for name in required if not name.endswith('_preview')}
        required = (required-{'instrumental', 'backing_vocals', 'lead_vocals'}) | set(playback_stems(result['artifacts']))
    if not required <= result['artifacts'].keys():
        raise ValueError('Missing required output artifacts')
    paths, hashes = {}, {}
    for name, artifact in result['artifacts'].items():
        path = (folder/artifact['path']).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size != artifact['bytes']:
            raise ValueError(f'Invalid artifact: {name}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != artifact['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
        for start in (() if audio_only else (0, max(0, artifact['bytes']-1024))):
            end = min(artifact['bytes']-1, start+1023)
            response = client.get(f'{base}/v1/jobs/{job["id"]}/artifacts/{name}',
                                  headers={'Range': f'bytes={start}-{end}'})
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise ValueError(f'Artifact range playback failed: {name}, {response.status_code}')
            with path.open('rb') as stream:
                stream.seek(start)
                if response.content != stream.read(end-start+1):
                    raise ValueError(f'Artifact range content mismatch: {name}')
        paths[name], hashes[name] = path, digest
    audio_format = verify_stems(paths, result['duration_seconds'], storage=result.get('audio_storage'))
    if result.get('audio_storage') and result['shared_preview_gain'] != 1:
        raise ValueError('Compact playback audio must not apply its common gain twice')
    cues = json.loads(paths['subtitles'].read_text())
    if not cues:
        raise ValueError('No usable subtitles; audio and diagnostics retained')
    units = [u for cue in cues for u in cue['units']]
    duration = result['duration_seconds']
    review = result.get('quality', {}).get('timing_review')
    if review is not None:
        from karaoke.timing_validation import validate_timeline
        lyrics = json.loads((folder/'lyrics.json').read_text())
        validate_timeline(cues, lyrics['lines'], duration, review['onset_constraints'], result['language'])
        if review['unresolved']:
            raise ValueError('Unresolved acoustic timing contradictions')
    if any(not 0 <= c['start'] < c['end'] <= duration+.05 for c in cues):
        raise ValueError('Invalid subtitle phrase intervals')
    if (not units and not playback_policy) or any(not 0 <= u['start'] < u['end'] <= duration+.05 for u in units):
        raise ValueError('Invalid or empty subtitle intervals')
    if any(a['start'] >= b['start'] for a, b in zip(units, units[1:])):
        raise ValueError('Non-monotonic word onsets')
    if any(normalized(''.join(u['text'] for u in c['units'])) != normalized(c['text']) for c in cues):
        raise ValueError('Subtitle units omit lyric text')
    if any(collapsed_phrase(c, result['language']) for c in cues):
        raise ValueError('Collapsed lyric phrase timing')
    if not playback_policy and highlight_coverage_errors(cues, result['language']):
        raise ValueError('Subtitle words or characters lack independent highlights')
    if not playback_policy and result.get('language') == 'English':
        from karaoke.onset_consensus import context_scope_violations
        if context_scope_violations(cues, result['quality'].get('english_context_scope')):
            raise ValueError('Word timing falls outside acoustic phrase context')
    if playback_policy:
        lyrics = json.loads((folder/'lyrics.json').read_text())
        if not units or [c['text'] for c in cues] != lyrics['lines']:
            raise ValueError('Subtitles omit supplied lyric text')
    else:
        from karaoke.song_sources import lyric_coverage, missing_repetitions
        from karaoke.vocal_review import speech_intervals
        repeats_raw = (folder/'asr/lyric-repeats.json').read_bytes()
        repeats = json.loads(repeats_raw)
        if repeats['status'] != 'complete' or missing_repetitions(
                repeats['reference_lines'], cues, repeats['words'], result['language'], duration):
            raise ValueError('Subtitles omit recognized lyric repetitions')
        coverage = lyric_coverage(cues, repeats['words'], result['language'], duration,
            other_words=repeats.get('other_coverage_words', []),
            speech_intervals=speech_intervals(folder, repeats.get('non_singing_reviews', [])))
        save(folder/'lyric-coverage-validation.json', dict(coverage, input_sha256=dict(
            subtitles=hashes['subtitles'], recognition=hashlib.sha256(repeats_raw).hexdigest())))
        if coverage['status'] != 'complete':
            raise ValueError('Subtitles omit recognized lyric passages')
    for name in ('mv', 'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'):
        if name not in paths:
            continue
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(paths[name])], timeout=30))
        needed = ({'video'} if result.get('video_only') else {'audio', 'video'}) if name == 'mv' else {'audio'}
        streams = [s for s in probe['streams'] if s['codec_type'] in needed]
        if {s['codec_type'] for s in streams} != needed or any(
                abs(float(s.get('duration', probe['format']['duration']))-duration) > .15 for s in streams):
            raise ValueError(f'Truncated media or missing stream: {name}')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(paths[name])]
        command += sum((['-map', '0:'+kind[0]+':0'] for kind in sorted(needed)), [])
        subprocess.run(command+['-f', 'null', '-'], check=True, capture_output=True, timeout=300)
    return dict(validation_scope=('playback_audio' if playback_policy and audio_only else
                                  'audio_and_lyrics' if audio_only else 'full_mv'),
                artifact_sha256=hashes, audio_format=audio_format,
                subtitle_lines=len(cues), subtitle_units=len(units),
                processing_seconds=result['processing_seconds'], timings=result['timings'],
                quality=result['quality'], device=result['device'], reused_stages=result.get('reused_stages', []))


def wait_job(client, base, job_id):
    job = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
    if job['status'] not in ('queued', 'running'):
        return job
    with client.stream('GET', base+'/v1/room/events', params={'job_id': job_id}) as events:
        events.raise_for_status()
        for line in events.iter_lines():
            if not line.startswith('data: '):
                continue
            snapshot = json.loads(line[6:])
            song = snapshot.get('playback_entry') or {}
            if song.get('job_id') == job_id and song.get('status') not in ('queued', 'running'):
                current = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
                if current['status'] not in ('queued', 'running'):
                    return current
    raise ConnectionError('Room event stream ended before completion')


class ProbeSession:
    """One controller owns all its probes; infrastructure failure cancels the group."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cancelled = threading.Event()
        self.lock = None
        self.monitor_lock = threading.Lock()
        self.resources = None
        self.checked_at = 0.

    def __enter__(self):
        self.lock = (self.root/'focused-probe.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        return self

    def check(self, folder=None):
        if self.lock is None or self.lock.closed or (folder is not None and folder.parent.resolve() != self.root):
            raise ValueError('Probe session does not own this data directory')
        if self.cancelled.is_set():
            raise ValueError('Probe session stopped after an infrastructure failure')

    def check_resources(self, client, base, *, refresh=False, log=None):
        """All children share one fresh resource sample per second."""
        with self.monitor_lock:
            self.check()
            if not refresh and self.resources is not None and time.monotonic()-self.checked_at < 1:
                return self.resources
            try:
                for attempt in range(2):
                    try:
                        storage = client.get(base+'/v1/storage').raise_for_status().json()
                        if (storage['limit_bytes'] is not None and storage['used_bytes'] > storage['limit_bytes']) or not storage['can_prepare']:
                            raise ValueError('Focused probe reached the shared cache or free-space limit')
                        health = client.get(base+'/health').raise_for_status().json()
                        break
                    except httpx.TransportError as error:
                        if attempt:
                            raise
                        message = f'Probe monitor retry after {type(error).__name__}'
                        if log is None:
                            print(message, flush=True)
                        else:
                            log.write((message+'\n').encode())
                            log.flush()
                        time.sleep(.2)
                        self.check()
            except BaseException:
                self.cancelled.set()
                raise
            self.resources = health, storage
            self.checked_at = time.monotonic()
            return self.resources

    def __exit__(self, *error):
        self.cancelled.set()
        assert self.lock is not None
        self.lock.close()


def run_probe(client, base, folder, command, *, resume_preparation=None, session=None):
    """Run one registered job under the live service's shared cache budget."""
    health = client.get(base+'/health').raise_for_status().json()
    job = client.get(f'{base}/v1/jobs/{folder.name}').raise_for_status().json()
    if not health.get('preparation_paused') or health.get('running_job') or job['status'] != 'queued':
        raise ValueError('Pause preparation and register a queued job before a focused probe')
    pause = folder.parent/'preparation.pause'
    store = None
    if resume_preparation is not None:
        if pause.read_text() != resume_preparation:
            raise ValueError('Only the owner of the preparation pause may resume it')
        from karaoke.api import Store
        store = Store(folder.parent)
        store.on_change = lambda: client.post(base+'/v1/room/events').raise_for_status()
    if session is not None and resume_preparation is not None:
        raise ValueError('A shared probe session must keep ordinary preparation paused')
    with (ProbeSession(folder.parent) if session is None else nullcontext(session)) as session:
        session.check(folder)
        (folder/'error.json').unlink(missing_ok=True)
        with (folder/'probe.log').open('ab') as log, ThreadPoolExecutor(max_workers=1) as reader:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE if store is not None else log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'KARAOKE_EVENTS_STDOUT': '1',
                     'KARAOKE_STAGE_LOCK_DIR': str(folder.parent/'stage-locks')})
            try:
                output = None
                if store is not None:
                    # Claim while dispatch is paused, so the ordinary worker cannot
                    # launch this job a second time when other songs resume.
                    store.update(folder.name, status='running', active_stages=[], error=None)
                    if pause.read_text() != resume_preparation:
                        raise ValueError('Preparation pause ownership changed')
                    pause.unlink()
                    assert process.stdout is not None
                    stdout = process.stdout
                    def read_output():
                        for line in stdout:
                            log.write(line)
                            log.flush()
                            if line.startswith(b'KARAOKE_PROGRESS '):
                                progress = json.loads(line[len(b'KARAOKE_PROGRESS '):])
                                store.update(folder.name, **{key: progress[key] for key in
                                    ('active_stages', 'stage_progress', 'timings', 'song_info') if key in progress})
                    output = reader.submit(read_output)
                while True:
                    session.check(folder)
                    health, _ = session.check_resources(client, base, log=log)
                    if store is None and (not health.get('preparation_paused') or health.get('running_job')):
                        raise ValueError('Preparation resumed during the focused probe')
                    if output is not None and output.done():
                        output.result()
                    result = process.poll()
                    if result is not None:
                        break
                    time.sleep(1)
            except BaseException:
                session.cancelled.set()
                raise
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    # The parent may exit before a descendant closes stdout.
                    # Release the entire group before joining the output reader.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
            if output is not None:
                output.result()
            return result


def probe(path, data_dir, base, job_id):
    with httpx.Client(timeout=30) as client:
        job = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
        state = json.loads(path.read_text())
        case = next(c for c in state['cases'] if video_url(c['url']) == job['input']['url'])
        retry_at = max(retry_deadline(state, case), *(
            (case.get(key) or [{}])[-1].get('next_retry_at') or 0
            for key in ('attempts', 'audio_attempts')))
        if time.time() < retry_at:
            raise ValueError(f'Provider backoff lasts until {datetime.fromtimestamp(retry_at, UTC).isoformat()}')
        folder = (data_dir/job['id']).resolve()
        command = [sys.executable, '-u', '-m', 'karaoke.pipeline', '--request', str(folder/'request.json'),
                   '--output', str(folder), '--audio-only']
        version = client.get(base+'/health').raise_for_status().json()['pipeline_version']
        code = run_probe(client, base, folder, command)
        if code:
            raise RuntimeError(f'Audio probe failed ({code}); inspect {folder / "error.json"} and probe.log')
        evidence = verify_outputs(client, base, data_dir, job, audio_only=True)
        if version != client.get(base+'/health').raise_for_status().json()['pipeline_version']:
            raise ValueError('Pipeline changed during the focused probe')
        evidence['pipeline_version'] = version
        save(folder/'audio-verification.json', evidence)
        print(json.dumps(evidence, ensure_ascii=False), flush=True)


def run_audio(path, data_dir, base, *, case_url=None, defer_urls=(), keep_going=False, download_only=False):
    """Retain each audio outcome; optionally continue past failed songs."""
    from karaoke.api import Store
    from karaoke.stage_cache import digest
    ledger = 'download_attempts' if download_only else 'audio_attempts'
    prefix = 'download' if download_only else 'audio'
    passed = 'downloaded_audio' if download_only else 'passed_audio'
    controller_version = digest(Path(__file__))
    if case_url is not None:
        case_url = video_url(case_url)
    deferred = {video_url(url) for url in defer_urls}
    store = Store(data_dir)
    with path.with_suffix('.lock').open('a') as lock, httpx.Client(timeout=30) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = json.loads(path.read_text())
            selection = [dict(url=c['url'], language=c['language']) for c in state['cases']]
            if hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest() != state['selection_sha256']:
                raise ValueError('Fixed case selection changed')
            if not deferred <= {video_url(c['url']) for c in state['cases']}:
                raise ValueError('Deferred source is not in the fixed selection')
            candidates_scope = [c for c in state['cases'] if video_url(c['url']) not in deferred
                                and (case_url is None or video_url(c['url']) == case_url)]
            if not candidates_scope:
                raise ValueError('No active source in the fixed selection')
            health = client.get(base+'/health').raise_for_status().json()
            version = health['pipeline_version']
            state[f'{prefix}_pipeline_version'] = version
            state[f'{prefix}_deferred_urls'] = sorted(deferred)
            state[f'{prefix}_keep_going'] = keep_going
            state[f'{prefix}_passed_current_version'] = sum(bool(c.get(ledger)) and
                c[ledger][-1]['status'] == passed and
                c[ledger][-1]['pipeline_version'] == version for c in state['cases'])
            save(path, state)
            if state[f'{prefix}_passed_current_version'] == len(state['cases']):
                print(f'All {len(state["cases"])} {prefix} cases finished on {version}', flush=True)
                return
            if case_url is None and path.with_name(f'{prefix}.pause').exists():
                print(f'{prefix.capitalize()} regression paused', flush=True)
                return
            if any(c.get(key) and c[key][-1]['status'] in ('running', 'submitted')
                   for key in ('audio_attempts', 'download_attempts')
                   for c in state['cases']):
                print('Reconcile the unfinished audio attempt before starting another', flush=True)
                return
            if not keep_going and any(c.get(ledger) and c[ledger][-1]['status'] == 'failed' and
                   c[ledger][-1]['pipeline_version'] == version for c in candidates_scope):
                print('Repair the audio failure before resuming this pipeline version', flush=True)
                return
            candidates = []
            for case in candidates_scope:
                attempts = case.get(ledger, [])
                if attempts and attempts[-1]['status'] == passed and attempts[-1]['pipeline_version'] == version:
                    continue
                fixed_downloader = download_only and attempts and attempts[-1]['status'] == 'failed' and (
                    attempts[-1].get('controller_version') != controller_version)
                resolved = not download_only and resolved_audio_failure(case, version)
                if attempts and not (fixed_downloader or resolved) and not eligible({'attempts': attempts}, version, time.time()):
                    continue
                previous = (case.get('attempts') or [{}])[-1]
                if offline() and (previous.get('error') or {}).get('provider') == 'openai':
                    previous = {}
                deadline = retry_deadline(state, case)
                if download_only:
                    deadline = state['provider_backoff'].get(case['provider'], 0)
                    previous = (case.get('audio_attempts') or [previous])[-1]
                    if (previous.get('error') or {}).get('code') != 'download_failed':
                        previous = {}
                if time.time() < max(deadline, previous.get('next_retry_at') or 0):
                    continue
                candidates.append(case)
            if not candidates:
                print('No eligible audio cases; inspect failures and retry deadlines', flush=True)
                return
            def priority(case):
                last = (case.get(ledger) or case.get('audio_attempts') or case.get('attempts') or [{}])[-1]
                status = last.get('status')
                retry_after_survey = keep_going and status == 'external' and last.get('pipeline_version') == version
                return (retry_after_survey, status not in ('failed', 'blocked', 'external'), status in ('passed', 'passed_audio'),
                        (last.get('finished_at') or '') if status == 'external' else '', not bool(case.get('audio_attempts')))
            candidates.sort(key=priority)
            case = candidates[0]
            room = client.get(base+'/v1/room').raise_for_status().json()
            if keep_going:
                if not health.get('preparation_paused') or health.get('running_job'):
                    raise ValueError('Pause preparation before continuing audio regression')
                storage = room['storage']
                if state.get('cache_limit_bytes', storage['limit_bytes']) != storage['limit_bytes']:
                    raise ValueError('Cache limit changed during audio regression')
                if not storage['can_prepare']:
                    print('Waiting for cache space; user queue remains protected', flush=True)
                    return
            existing = {e['id'] for e in room['entries']}
            # A full playback result is not a strict audio acceptance result.
            # Keep it intact, including when its video is being restored.
            force = not download_only and any(j['input']['url'] == case['url'] and
                (j['status'] == 'completed' or j.get('restore_video') or j.get('result'))
                for j in store.list(-1))
            room = client.post(base+'/v1/room/entries', json=dict(
                url=case['url'], source_title=case['title'], force=force)).raise_for_status().json()
            entry = next(e for e in room['entries'] if e['id'] not in existing)
            job = client.get(f'{base}/v1/jobs/{entry["job_id"]}').raise_for_status().json()
            attempts = case.setdefault(ledger, [])
            attempt = dict(job_id=job['id'], entry_id=entry['id'], pipeline_version=version,
                controller_version=controller_version, status='running', submitted_at=stamp(),
                external_attempt=1+sum(a['status']=='external' for a in attempts))
            attempts.append(attempt)
            save(path, state)
            folder = data_dir/job['id']
            command = [sys.executable, '-u', '-m', 'karaoke.pipeline', '--request', str(folder/'request.json'),
                       '--output', str(folder), '--download-only' if download_only else '--audio-only']
            probe_finished = False
            try:
                if force:
                    store.cache.reuse_download(folder, case['url'])
                cached_full = job['status'] == 'completed'
                exit_code = 0 if cached_full else run_probe(client, base, folder, command)
                probe_finished = True
                if exit_code:
                    raise RuntimeError('Audio pipeline failed; retained diagnostics')
                if client.get(base+'/health').raise_for_status().json()['pipeline_version'] != version:
                    probe_finished = False
                    raise RuntimeError('Pipeline changed during audio regression')
                if download_only:
                    evidence = json.loads((folder/'source.json').read_text())
                    if (evidence.get('input_url', evidence['url']) != case['url'] or
                            evidence['sha256'] != digest(folder/'download/source.mkv')):
                        raise ValueError('Downloaded source identity or hash mismatch')
                    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i',
                        str(folder/'download/source.mkv'), '-map', '0:a:0', '-f', 'null', '-'],
                        check=True, capture_output=True, timeout=300)
                    evidence = dict(source_sha256=evidence['sha256'], duration=evidence['duration'],
                                    validation_scope='download_only')
                else:
                    evidence = verify_outputs(client, base, data_dir, job, audio_only=not cached_full)
                evidence['pipeline_version'] = version
                save(folder/f'{prefix}-verification.json', evidence)
                attempt.update(status=passed, finished_at=stamp(), evidence=evidence)
                state[f'{prefix}_passed_current_version'] += 1
                if download_only and not cached_full:
                    store.update(job['id'], status='source_ready', active_stages=[], error=None)
                if not download_only and not cached_full:
                    result = json.loads((folder/'audio-result.json').read_text())
                    store.update(job['id'], status='audio_ready', pipeline_version=version, audio_result=result,
                        song_info={k:result[k] for k in ('title','artist','language','duration_seconds')}, active_stages=[], error=None)
            except Exception as error:
                error_path = folder/'error.json'
                job['error'] = json.loads(error_path.read_text()) if error_path.is_file() else dict(code='verification_failed', message=str(error))
                record_failure(attempt, job, folder)
                store.update(job['id'], status='failed', error=job['error'], active_stages=[])
                if attempt.get('provider_blocked') and attempt.get('next_retry_at') is not None:
                    provider = attempt.get('retry_provider', case['provider'])
                    state['provider_backoff'][provider] = max(state['provider_backoff'].get(provider, 0), attempt['next_retry_at'])
            save(path, state)
            room = client.get(base+'/v1/room').raise_for_status().json()
            if (room['current_id'] != entry['id'] and entry['id'] not in room.get('playing_entry_ids', [])
                    and any(e['id']==entry['id'] for e in room['entries'])):
                client.post(base+'/v1/room/commands',json=dict(action='remove',entry_id=entry['id'],revision=room['revision'])).raise_for_status()
            print(json.dumps({k:attempt.get(k) for k in ('job_id','status','error')},ensure_ascii=False),flush=True)
            if (case_url is not None or not probe_finished or (download_only and attempt['status'] == 'blocked') or
                    (not keep_going and attempt['status'] in ('failed','blocked'))):
                return


def record_failure(attempt, job, folder):
    error = job.get('error') or dict(code='verification_failed', message='Output validation failed')
    diagnostics = {}
    for path in sorted(folder.glob('*.log')):
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-16384))
                diagnostics[path.name] = stream.read().decode(errors='replace')
        except FileNotFoundError:  # Cache eviction may race a completed job's inspection.
            continue
    names = ('worker.log', 'probe.log', 'download.log', 'transcribe.log', error['code'].removesuffix('_failed')+'.log')
    logs = [diagnostics[name][-4000:] for name in dict.fromkeys(names) if name in diagnostics]
    # Older workers reported these as generic stage failures. Classify their
    # retained logs too, so a code rollout does not cause inappropriate retries.
    parsed = provider_error('\n'.join(logs))
    error = {**error, **parsed} if parsed else error
    attempt.update(status='failed', error=error, finished_at=stamp(),
        error_fingerprint=hashlib.sha256(json.dumps(error, sort_keys=True).encode()).hexdigest())
    attempt['diagnostic_tail'] = '\n'.join(logs)
    for pattern in ('asr/*.json', 'asr-requests/*.json', 'alignment/*.json', 'phonemes/*.json', 'context/**/*.json',
                    'lyric-source-attempts/**/*.json',
                    'repeated-lyrics/*.json', 'english-onsets/*.json',
                    'lyrics.json', 'anchors.json', 'english-anchors.json', 'word-cues.json', 'ordered-cues.json', 'cues.json', 'quality.json',
                    'prepared-lyrics.json', 'playback-warning.json', 'lyrics-first-lookup.json',
                    'published-lyrics.json', 'lyric-lookup.json', 'lyric-readings.json', 'song-metadata.json',
                    'audio-result.json', 'result.json', 'lyric-coverage-validation.json',
                    'lyric-repeat-validation.json', 'highlight-validation.json', 'localization-validation.json',
                    'timing-checked/*.json', 'timing-constraints.json', 'timing-input.json', 'timing-recovery.json',
                    'timing-context-recovery.json'):
        for path in folder.glob(pattern):
            try:
                # Alignment retries contain repeated contexts which compress
                # well. Preserve them before releasing the job's cache guard.
                if path.stat().st_size <= 16*1024**2:
                    diagnostics[str(path.relative_to(folder))] = json.loads(path.read_text())
                else:
                    diagnostics.setdefault('omitted_diagnostics', []).append(dict(
                        path=str(path.relative_to(folder)), bytes=path.stat().st_size, reason='size_limit'))
            except (ValueError, OSError):
                continue
    if diagnostics:
        payload = json.dumps(diagnostics, ensure_ascii=False, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        target = folder.parent/'soak-100'/f'{job["id"]}-{attempt["error_fingerprint"][:12]}-{digest}.json.gz'
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(gzip.compress(payload, mtime=0))
        attempt.update(reproduction_file=str(target), diagnostic_sha256=digest)
    attempt.update(retry_fields(error, '\n'.join(logs), attempt.get('external_attempt', 1), failed_at=time.time()))


def queue_cases(client, base, state, version, persist):
    """Use the same room API as a person's add button; no test-only queue limit."""
    room = client.get(base+'/v1/room').raise_for_status().json()
    # Retry software regressions first without changing the fixed selection or room order.
    cases = sorted(state['cases'], key=lambda case: not (
        case['attempts'] and case['attempts'][-1]['status'] == 'failed'))
    for case in cases:
        if len(room['entries']) >= 100:
            break
        last = case['attempts'][-1] if case['attempts'] else None
        if last and last['status'] in ('submitted', 'running'):
            if last.get('entry_id'):
                continue
            job = client.get(f'{base}/v1/jobs/{last["job_id"]}').raise_for_status().json()
            if job['status'] not in ('queued', 'running', 'completed'):
                continue
            payload = dict(job_id=last['job_id'])
        else:
            if not eligible(case, version, time.time()) or time.time() < retry_deadline(state, case):
                continue
            payload = dict(url=case['url'], source_title=case['title'],
                force=bool(last and last['status']=='passed' and last['pipeline_version']!=version))
        before = {e['id'] for e in room['entries']}
        response = client.post(base+'/v1/room/entries', json=payload)
        if response.status_code == 409:
            break  # The user may have filled the last available queue slot.
        room = response.raise_for_status().json()
        entry = next(e for e in room['entries'] if e['id'] not in before)
        if not last or last['status'] not in ('submitted', 'running'):
            job = client.get(f'{base}/v1/jobs/{entry["job_id"]}').raise_for_status().json()
            last = dict(job_id=job['id'], pipeline_version=job['pipeline_version'], status='submitted',
                        submitted_at=stamp(), external_attempt=1+sum(a['status']=='external' for a in case['attempts']))
            case['attempts'].append(last)
        last['entry_id'] = entry['id']
        persist()


def release_finished_entries(client, base, state):
    room = client.get(base+'/v1/room').raise_for_status().json()
    owned = {a.get('entry_id') for c in state['cases'] for a in c['attempts']
             if a['status'] not in ('submitted', 'running')}
    for entry in list(room['entries']):
        if (entry['id'] not in owned or entry['id'] == room['current_id']
                or entry['id'] in room.get('playing_entry_ids', [])):
            continue
        response = client.post(base+'/v1/room/commands', json=dict(action='remove',
            entry_id=entry['id'], revision=room['revision']))
        if response.status_code == 409:
            return
        room = response.raise_for_status().json()


def run(path, data_dir, base):
    # Only this controller writes the manifest; it submits to the existing worker.
    with path.with_suffix('.lock').open('w') as lock, httpx.Client(timeout=40) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if path.with_suffix('.pause').exists():
                print('Paused by controller file', flush=True)
                return
            state = json.loads(path.read_text())
            selection = [dict(url=c['url'], language=c['language']) for c in state['cases']]
            if hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest() != state['selection_sha256']:
                raise ValueError('Fixed case selection changed')
            health = client.get(base+'/health').raise_for_status().json()
            version = health['pipeline_version']
            state['current_pipeline_version'] = version
            state['updated_at'] = stamp()
            state['passed_current_version'] = sum(bool(c['attempts'] and c['attempts'][-1]['status'] == 'passed'
                and c['attempts'][-1]['pipeline_version'] == version) for c in state['cases'])
            release_finished_entries(client, base, state)
            failed = any(c['attempts'] and c['attempts'][-1]['status']=='failed' and
                         c['attempts'][-1]['pipeline_version']==version for c in state['cases'])
            storage = client.get(base+'/v1/storage').raise_for_status().json()
            if storage['limit_bytes'] != state['cache_limit_bytes']:
                raise RuntimeError('Cache limit changed; refusing to silently change it')
            if not failed and storage['can_prepare']:
                queue_cases(client, base, state, version, lambda: save(path, state))
            active = min(((c, c['attempts'][-1]) for c in state['cases'] if c['attempts'] and
                          c['attempts'][-1]['status'] in ('submitted', 'running')),
                         key=lambda item: item[1]['submitted_at'], default=None)
            if active:
                case, attempt = active
                try:
                    job = wait_job(client, base, attempt['job_id'])
                except (httpx.HTTPError, ConnectionError) as error:
                    print(f'Waiting for API reconnect: {type(error).__name__}', flush=True)
                    time.sleep(15)
                    continue
                attempt['pipeline_version'] = job.get('pipeline_version', attempt['pipeline_version'])
                if job['status'] == 'completed':
                    try:
                        playback = (job.get('result') or {}).get('validation_scope') in (
                            'playback_lyrics_first', 'playback_with_lyrics_review')
                        evidence = verify_outputs(client, base, data_dir, job, playback_policy=playback)
                        attempt.update(status='passed', finished_at=stamp(), evidence=evidence)
                    except Exception as error:
                        job['error'] = dict(code='verification_failed', message=str(error)[:1000])
                        record_failure(attempt, job, data_dir/job['id'])
                elif job['status'] in ('evicted', 'video_evicted'):
                    attempt.update(status='cache_evicted', finished_at=stamp())
                else:
                    record_failure(attempt, job, data_dir/job['id'])
                if attempt.get('provider_blocked') and attempt.get('next_retry_at') is not None:
                    provider = attempt.get('retry_provider', case['provider'])
                    state['provider_backoff'][provider] = max(
                        state['provider_backoff'].get(provider, 0), attempt['next_retry_at'])
                save(path, state)
                print(json.dumps({'title': case['title'], 'job_id': job['id'], 'status': attempt['status'],
                                  'error': attempt.get('error')}, ensure_ascii=False), flush=True)
                continue
            save(path, state)
            if state['passed_current_version'] == len(state['cases']):
                print(f'All {len(state["cases"])} cases passed on {version}', flush=True)
                return
            # Stop on reproducible software failures until the maintainer deploys a fix.
            if failed:
                print('Needs repair; retained failed job diagnostics', flush=True)
                return
            if not storage['can_prepare']:
                print('Waiting for cache space; user queue remains protected', flush=True)
                time.sleep(30)
                continue
            candidates = [c for c in state['cases'] if eligible(c, version, time.time()) and
                          time.time() >= retry_deadline(state, c)]
            if not candidates:
                print('No eligible cases; external failures are on backoff or need user action', flush=True)
                return
            time.sleep(15)  # User-owned entries may currently fill the room.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sample', 'run', 'probe', 'audio-run', 'audio-fetch'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8746')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--job-id', help='A registered, queued job for an audio-only probe')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--case-url', help='Verify only this frozen audio case, preserving batch pause files')
    scope.add_argument('--defer-url', action='append', default=[],
                       help='Explicitly defer a frozen audio case without deleting or passing it')
    parser.add_argument('--keep-going', action='store_true',
                        help='Record failed audio cases and continue without retrying unchanged failures')
    args = parser.parse_args()
    if (args.case_url or args.defer_url or args.keep_going) and args.action not in ('audio-run', 'audio-fetch'):
        parser.error('--case-url, --defer-url and --keep-going require audio-run or audio-fetch')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'sample':
        sample(args.manifest, args.data_dir, args.seed)
    elif args.action == 'probe':
        if not args.job_id:
            parser.error('probe requires --job-id')
        probe(args.manifest, args.data_dir, args.base_url.rstrip('/'), args.job_id)
    elif args.action in ('audio-run', 'audio-fetch'):
        run_audio(args.manifest, args.data_dir, args.base_url.rstrip('/'),
                  case_url=args.case_url, defer_urls=args.defer_url, keep_going=args.keep_going,
                  download_only=args.action == 'audio-fetch')
    else:
        run(args.manifest, args.data_dir, args.base_url.rstrip('/'))


if __name__ == '__main__':
    main()
