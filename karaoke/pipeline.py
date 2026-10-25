"""URL-to-karaoke adapter for the existing, versioned separation/alignment stages."""
import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from io import BufferedReader
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent
# The research stages predate the package API and use sibling imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import local_asr
from .cache import can_restore_video
from .job_schema import (
    JobRequest,
    download_backoff,
    preparation_rank,
    provider_error,
    retry_fields,
    song_language,
)
from .song_sources import (
    independent_transcript_support,
    lyric_coverage,
    missing_repetitions,
    prepare_song_text,
    restore_repetitions,
    transcript_lines,
    video_song_info,
)
from .stage_cache import StageCache
from .stage_progress import parse_progress


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pipeline_fingerprint():
    digest = hashlib.sha256(b'karaoke-url-api-v1')
    for path in [p for p in sorted(ROOT.glob('*.py'))
                 if not p.name.startswith('test_') and p.name not in ('api.py', 'room.py', 'cache.py')] + [
            ROOT/'subtitle-styles.json', ROOT/'experiments/cjk-onsets-v003/run.py',
            ROOT/'experiments/cjk-onsets-v003/protocol.json']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    # Model/config changes invalidate the cache without hashing GBs on each submission.
    for path in sorted((ROOT/'models/pretrained').glob('*roformer*')):
        stat = path.stat()
        digest.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for path in sorted(p for directory in ('whisper', 'sofa', 'mms', 'qwen3-forced-aligner-0.6b')
                       for p in (ROOT/'models/alignment'/directory).rglob('*')):
        if path.is_file() and (path.suffix in ('.ckpt', '.pt', '.ptsd', '.yaml', '.json') or path.name == 'dict.txt' or
                ('qwen3-forced-aligner-0.6b' in path.parts and path.suffix in ('.safetensors', '.txt'))):
            stat = path.stat()
            digest.update(f'{path.relative_to(ROOT)}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    g2p = ROOT/'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py'
    if g2p.is_file():
        digest.update(g2p.read_bytes())
    for path in sorted((ROOT/'lyrics').glob('*.json')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class StageFailure(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def processing_error(error, output):
    details = provider_error(error)
    if details is None and isinstance(error, StageFailure) and error.stage == 'transcribe':
        saved = Path(output)/'asr/error.json'
        if saved.is_file():
            details = json.loads(saved.read_text())
        path = Path(output)/'transcribe.log'
        if details is None and path.is_file():
            with path.open('rb') as log:
                log.seek(max(0, path.stat().st_size-16384))
                details = provider_error(log.read().decode(errors='replace'))
    return details or dict(code=getattr(error, 'stage', 'processing')+'_failed',
        message=str(error)[:600] if isinstance(error, (StageFailure, ValueError)) else
        '处理失败；请查看任务目录中的日志，或提供本版本歌词后重试')


def download_failure(log):
    """Expose actionable causes without publishing raw URLs, headers or credentials."""
    diagnostic = log.lower()
    if any(text in diagnostic for text in ('unable to connect to proxy', 'proxyerror', 'tunnel connection failed')):
        return '下载被网络代理拒绝，请检查服务的代理设置后重试'
    rate_limited = bool(re.search(r'(?:http error|error code:|http/\d(?:\.\d)?)\s*429\b', diagnostic))
    login_required = any(text in diagnostic for text in ('sign in', 'login required', 'login_required'))
    if rate_limited and login_required:
        return '视频站点正在限流（429），并要求登录验证；请检查下载登录信息后再重试'
    if rate_limited:
        return '视频站点正在限流（429），请稍后重试'
    if login_required:
        return '视频站点要求登录验证，请配置下载登录信息后重试'
    if any(text in diagnostic for text in ('video unavailable', 'private video', 'video is unavailable')):
        return '视频当前无法访问，请换一个可公开播放的链接'
    if re.search(r'(?:http error|error code:|http/\d(?:\.\d)?)\s*403\b', diagnostic):
        return '视频站点拒绝下载（403）；请检查下载器与登录信息后重试'
    if any(text in diagnostic for text in ('failed to resolve', 'temporary failure in name resolution',
                                           'nodename nor servname', 'name or service not known')):
        return '无法解析视频站点地址，请检查网络或 DNS 后重试'
    if any(text in diagnostic for text in ('timed out', 'connection reset')):
        return '连接视频站点超时或中断，请稍后重试'
    return '视频下载失败，请稍后重试或换一个视频链接'


def create_poster(source, output, *, thumbnail=None, seconds=30):
    """Cache the video thumbnail, falling back to a frame; artwork is optional."""
    output = Path(output)
    temporary = output.with_name('.poster.tmp.jpg')
    candidates = [(thumbnail, [])] if isinstance(thumbnail, str) and thumbnail.startswith(('https://', 'http://')) else []
    candidates.append((source, ['-ss', str(seconds)]))
    for media, options in candidates:
        try:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', *options,
                '-i', str(media), '-frames:v', '1', '-vf',
                'scale=960:540:force_original_aspect_ratio=decrease', '-q:v', '3', str(temporary)],
                capture_output=True, timeout=15)
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size:
                temporary.replace(output)
                return True
        except Exception:
            logging.getLogger(__name__).warning('Optional poster could not be created', exc_info=True)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning('Optional poster temporary file could not be removed', exc_info=True)
    return False


def alignment_recovery_summary(raw):
    localized = [report for block in raw for report in block.get('asr_localizations', [])]
    recovered, unresolved = set(), set()
    for block in raw:
        # Context retains old evidence for other lines. Only the selected
        # prediction owns a line's current recovery status after stitching.
        owned = set(range(block['range'][0]+1, block['range'][1]+1))
        recovered.update(i for report in block.get('line_recoveries', []) if report['accepted']
                         for i in range(report['lines'][0], report['lines'][1]+1) if i in owned)
        for report in block.get('asr_localizations', []):
            unresolved.update(owned.intersection(report.get('contradicted_lines', [])))
            unresolved.difference_update(owned.intersection(report['accepted_lines']))
            recovered.update(owned.intersection(report['accepted_lines']))
        # Isolated recovery runs after localization and replaces the rejected
        # prediction. Its accepted result must retire that prediction's failure,
        # while earlier/failed attempts cannot clear a later contradiction.
        unresolved.difference_update(i for report in block.get('line_recoveries', [])
            if report['accepted'] and report.get('isolated')
            for i in range(report['lines'][0], report['lines'][1]+1) if i in owned)
        stitched = owned.intersection(block.get('stitch_recovered_lines', []))
        unresolved.difference_update(stitched)
        recovered.update(stitched)
    recovered.difference_update(unresolved)
    return dict(recovered_alignment_lines=sorted(recovered),
        unresolved_localization_lines=sorted(unresolved),
        asr_localization_windows=len({tuple(r['source_range']) for r in localized if r.get('recognition')}))


def phrase_locations(lines, blocks):
    """Keep independent phrase evidence even when word alignment failed."""
    locations = {}
    for block in blocks:
        first, stop = block['range']
        for observed in block.get('asr_localizations', []):
            if not observed.get('recognition') or 'context' not in observed:
                continue
            anchors = observed.get('anchors', [])
            offset = observed['context'][0]
            for local, anchor in enumerate(anchors):
                index = offset+local
                if not first <= index < min(stop, len(lines)):
                    continue
                located = index+1 in observed.get('accepted_lines', []) or anchor.get('match_basis') in (
                    'complete_phrase', 'complete_clause')
                if located and anchor['status'] == 'candidate' and anchor['text'] == lines[index]:
                    locations[index] = anchor
    return locations


class Pipeline:
    def __init__(self, request, output, *, separator_python=None, device='auto', text_model='gpt-5-mini',
                 max_duration=900, catalog_root=ROOT, cookies_file=None, audio_only=False,
                 playback_policy=False, prepared_lyrics=None):
        self.request = JobRequest.model_validate(request).model_dump()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.separator_python = separator_python or sys.executable
        self.device, self.text_model = device, text_model
        self.compute_concurrency = int(os.getenv('KARAOKE_COMPUTE_CONCURRENCY', '1'))
        if self.compute_concurrency < 1:
            raise ValueError('Compute concurrency must be positive')
        self.max_duration, self.catalog_root = max_duration, catalog_root
        self.cookies_file = Path(cookies_file).expanduser() if cookies_file else self.output.parent/'private/youtube-cookies.txt'
        self.audio_only = audio_only
        self.strict_audio = audio_only and not playback_policy
        self.prepared_lyrics = Path(prepared_lyrics) if prepared_lyrics else None
        self.active, self.timings, self.lock = set(), {}, threading.Lock()
        self.timeline: dict[str, dict[str, Any]] = {}
        self.clock_start = time.perf_counter()
        self.metadata_ready = None
        self.audio_first = False
        self.source_announced = False
        self.fresh_source_metadata = False
        self.recording_language = None
        self.stage_progress = {}
        self.song_info = {}
        try:
            self.song_info.update(json.loads((self.output/'progress.json').read_text()).get('song_info') or {})
        except (OSError, ValueError):
            pass
        self.aligned_cues = None
        self.alignment_outputs = set()
        self.source = self.output/'download/source.mkv'
        self.video = self.source
        self.vocals = self.output/'music-separation/song/vocals.wav'
        self.checkpoints = StageCache(self.output)

    def stage_spec(self, name, command):
        """Explicit inputs/outputs prevent a downstream fix from discarding upstream work."""
        common = ['subtitles.py', 'lyric_progress.py', 'stage_progress.py']
        whisper = ['align_lyrics_whisper.py', 'anchors_from_asr.py', 'stage_cache.py', 'local_asr.py', 'job_schema.py']
        phonemes = ['align_lyric_phonemes.py', 'phoneme_progress.py']
        english_code = [*whisper, 'align_english_onsets.py', 'onset_consensus.py', 'word_onset_metrics.py']
        english = str(ROOT/'align_english_onsets.py') in [str(a) for a in command]
        alignment_code = {key: [ROOT/p for p in common+files] for key, files in {
            'align_words': [*whisper, *phonemes, 'cjk_recovery.py'],
            'align_phonemes': english_code if english else phonemes,
            'align_repeats': [*phonemes, 'align_repeated_lyrics.py'],
            'align_context': [*phonemes, 'align_repeated_lyrics.py', 'cjk_recovery.py', 'qwen_witness.py',
                              'experiments/cjk-onsets-v003/run.py', 'experiments/cjk-onsets-v003/protocol.json'],
        }.items()}
        specs = {
            'decode_audio': (['download/source.mkv'], ['source-44100.wav'], []),
            'separate_music': (['source-44100.wav'], ['music-separation'], [ROOT/'benchmark_neural.py']),
            'resample_vocals': (['music-separation/song/vocals.wav'],
                                ['vocals-16k.wav'] + (['vocals-asr.mp3'] if self.strict_audio else []), []),
            'transcribe': (['vocals-asr.mp3', 'source-44100.wav'], ['asr'], [ROOT/p for p in
                ('transcribe_lyrics.py', 'local_asr.py', 'anchors_from_asr.py', 'subtitles.py', 'job_schema.py')]),
            'separate_lead': (['music-separation/song/vocals.wav'], ['lead-separation'], [ROOT/'benchmark_neural.py']),
            'package_three_stems': (['download/source.mkv', 'music-separation/song/vocals.wav',
                                     'lead-separation/song/vocals.wav'],
                                    ['stems'], [ROOT/'exact_stems.py', ROOT/'three_stems.py']),
            'align_words': (['vocals-16k.wav', 'source-44100.wav', 'lyrics.json', 'anchors.json', 'alignment-jobs.json'],
                            ['alignment'], alignment_code['align_words']),
            'align_phonemes': ((['vocals-16k.wav', 'lyrics.json', 'word-cues.json', 'english-anchors.json', 'english-jobs.json'] if english else
                                ['music-separation/song/vocals.wav', 'word-cues.json', 'phoneme-jobs.json']),
                               ['english-onsets' if english else 'phonemes'], alignment_code['align_phonemes']),
            'align_repeats': (['music-separation/song/vocals.wav', 'phonemes/cues.json', 'word-cues.json', 'repeat-jobs.json'],
                              ['repeated-lyrics'], alignment_code['align_repeats']),
            'align_context': (['music-separation/song/vocals.wav', 'source-44100.wav', 'ordered-cues.json', 'context-jobs.json'],
                              ['context'], alignment_code['align_context']),
            'validate_timing': (['music-separation/song/vocals.wav', 'source-44100.wav', 'timing-input.json', 'lyrics.json'],
                                ['timing-checked'], [ROOT/p for p in common+phonemes+
                                ['timing_validation.py', 'anchors_from_asr.py', 'cjk_recovery.py', 'align_english_sofa.py',
                                 'align_english_onsets.py',
                                 'experiments/cjk-onsets-v003/protocol.json']]),
        }
        if name not in specs:
            return None
        inputs, outputs, code = specs[name]
        if name in ('align_words', 'validate_timing') and json.loads((self.output/'lyrics.json').read_text())['language'] == 'Cantonese':
            code = code+[ROOT/'subtitle_style.py', ROOT/'subtitle-styles.json', ROOT/'requirements-subtitles.txt']
        config: dict[str, Any] = dict(command=[str(a) for a in command], compute_concurrency=self.compute_concurrency)
        if name == 'validate_timing':
            from .timing_validation import acoustic_model_files
            if json.loads((self.output/'lyrics.json').read_text())['language'] not in ('Chinese', 'Japanese', 'English'):
                code = code+[ROOT/p for p in ('align_lyrics_whisper.py', 'local_asr.py',
                                             'job_schema.py', 'requirements-subtitles.txt')]
            if '--lead' in command:
                inputs = [*inputs, command[command.index('--lead')+1]]
            if '--context-windows' in command:
                inputs = [*inputs, command[command.index('--context-windows')+1]]
            for evidence in ('timing-constraints.json', 'alignment/raw.json'):
                if (self.output/evidence).is_file():
                    inputs = [*inputs, evidence]
            if '--previous-alignment' in command:
                prior = Path(command[command.index('--previous-alignment')+1])
                inputs = [*inputs, *(prior/p for p in ('lyrics.json', 'cues.json', 'timing-constraints.json',
                    'timing-checked/cues.json', 'timing-checked/report.json'))]
                if (self.output/'stems/lead.wav').is_file():
                    inputs = [*inputs, 'stems/lead.wav']
            weights = acoustic_model_files(ROOT, json.loads((self.output/'lyrics.json').read_text())['language'])
            config['weights'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                 for p in weights if p.is_file()]
        if name in ('separate_music', 'separate_lead'):
            model = 'vocals_mel_band_roformer' if name == 'separate_music' else 'karaoke_bs_roformer_anvuew'
            code = code+[ROOT/f'models/pretrained/{model}.yaml']
            weights = ROOT/f'models/pretrained/{model}.ckpt'
            config['weights'] = (weights.stat().st_size, weights.stat().st_mtime_ns)
        if name.startswith('align_'):
            if name == 'align_words':
                arguments = [str(a) for a in command]
                model = arguments[arguments.index('--model')+1] if '--model' in arguments else 'large-v3-turbo'
                weights = [ROOT/f'models/alignment/whisper/{model}.pt', ROOT/'models/alignment/mms/model.pt']
                if json.loads((self.output/'lyrics.json').read_text())['language'] == 'Chinese':
                    weights.extend(sorted((ROOT/'models/alignment/sofa/mandarin').rglob('*.ckpt')))
            else:
                language = 'English' if english else json.loads((self.output/'lyrics.json').read_text())['language']
                directory = {'Chinese': 'mandarin', 'Japanese': 'japanese', 'English': 'english'}[language]
                weights = sorted((ROOT/'models/alignment/sofa'/directory).rglob('*.ckpt'))
                if name == 'align_context':
                    weights.append(ROOT/'models/alignment/mms/model.pt')
                if name == 'align_context' and language != 'English' and any(c.isascii() and c.isalpha()
                        for line in json.loads((self.output/'lyrics.json').read_text()).get('lines', []) for c in line):
                    weights.extend(p for p in (ROOT/'models/alignment/sofa/english').rglob('*')
                                   if p.suffix in ('.ckpt', '.ptsd', '.yaml') or p.name == 'dict.txt')
                    code = code+[ROOT/p for p in ['align_english_sofa.py', *english_code]]
                    g2p = ROOT/'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py'
                    if g2p.is_file():
                        code = code+[g2p]
                if name == 'align_context' and language == 'Chinese':
                    weights.extend(p for p in (ROOT/'models/alignment/qwen3-forced-aligner-0.6b').glob('*')
                                   if p.suffix in ('.safetensors', '.json', '.txt'))
                    config['qwen_python'] = os.getenv('KARAOKE_QWEN_PYTHON', sys.executable)
            # An unrelated model download must not invalidate completed stages.
            config['weights'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                for p in weights if p.is_file()]
        if name == 'transcribe':
            config['asr'] = local_asr.configuration()
            if '--lead-audio' in command:
                inputs = [*inputs, command[command.index('--lead-audio')+1]]
            config['fallback_model'] = os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe')
            weights = ROOT/'models/alignment/whisper/large-v3-turbo.pt'
            config['local_repair_weights'] = (weights.stat().st_size, weights.stat().st_mtime_ns) if weights.is_file() else None
        spec = dict(inputs=[self.output/p for p in inputs], outputs=[self.output/p for p in outputs],
                    config=config, code=code)
        if name in ('separate_music', 'separate_lead') and not self.strict_audio:
            spec['required'] = [self.output/outputs[0]/'song/vocals.wav']
        return spec

    def phase(self, name=None, active=False, *, song_info=None):
        with self.lock:
            if name is not None:
                self.active.add(name) if active else self.active.discard(name)
                self.stage_progress.pop(name, None)
            if song_info is not None:
                self.song_info.update(song_info)
            self.publish_progress()

    def publish_progress(self):
        # Caller holds the lock: parallel stages publish one coherent snapshot.
        progress = dict(active_stages=sorted(self.active), stage_progress=self.stage_progress,
            updated_at=datetime.now(UTC).isoformat(), timings=self.timings,
            song_info=self.song_info)
        write_json(self.output/'progress.json', progress)
        if os.getenv('KARAOKE_EVENTS_STDOUT') == '1':
            print('KARAOKE_PROGRESS ' + json.dumps(progress), flush=True)

    def stage_event(self, name, event, **fields):
        with self.lock:
            record: dict[str, Any] = self.timeline.setdefault(name, {})
            if event == 'ready' and 'finished' in record:
                history = record.pop('previous_attempts', [])
                record = {'previous_attempts': [*history, record]}
                self.timeline[name] = record
            instant = round(time.perf_counter()-self.clock_start, 6)
            if event == 'finished':
                record[event] = instant
            else:
                record.setdefault(event, instant)
            record.update(fields)
            write_json(self.output/'stage-timeline.json', self.timeline)

    def run(self, name, command, timeout=3600):
        self.stage_event(name, 'ready')
        spec = self.stage_spec(name, command)
        if spec is None:
            return self.run_process(name, command, timeout)
        started = time.perf_counter()
        with self.checkpoints.stage(name, **spec) as reused:
            if name.startswith('align_'):
                self.alignment_outputs.update(Path(p) for p in spec['outputs'])
            if reused:
                self.stage_event(name, 'finished', status='cached')
                with self.lock:
                    self.timings['reuse_'+name] = round(time.perf_counter()-started, 3)
                    self.publish_progress()
            else:
                self.run_process(name, command, timeout)

    def run_process(self, name, command, timeout=3600):
        self.stage_event(name, 'ready')
        with self.stage_slot(name):
            self.stage_event(name, 'resource_acquired')
            return self.run_subprocess(name, command, timeout)

    @contextmanager
    def stage_slot(self, name):
        """Bound simultaneous compute while retaining a single download/render lane."""
        directory = os.getenv('KARAOKE_STAGE_LOCK_DIR')
        lane = ('separation' if name.startswith('separate_') else
                'alignment' if name in ('transcribe', 'detect_language', 'validate_timing') or name.startswith('align_') else
                'render' if name == 'render' else
                'mux' if name in ('mux_video', 'verify_video') else
                'packaging' if name.startswith('package_') else
                'download' if name in ('download', 'download_video') else None)
        if not directory or lane is None:
            yield
            return
        import fcntl
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        request = folder/f'{lane}.{os.getpid()}.{threading.get_ident()}.{stamp}.json'
        capacity = self.compute_concurrency if lane in ('separation', 'alignment', 'packaging') else 1
        with request.open('x+') as registration, ExitStack() as handles:
            slots = [handles.enter_context((folder/(lane+('' if i == 0 else f'.{i}')+'.lock')).open('a'))
                     for i in range(capacity)]
            slot = None
            fcntl.flock(registration, fcntl.LOCK_EX)
            json.dump(dict(job=self.output.name, ready=stamp, url=self.request['url']), registration)
            registration.flush()
            try:
                waiting = False
                while True:
                    try:
                        for candidate in slots:
                            try:
                                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            slot = candidate
                            break
                        else:
                            raise BlockingIOError
                        try:
                            order = json.loads((folder/'priority.json').read_text())
                        except (OSError, ValueError):
                            order = []
                        rank = preparation_rank(order)
                        candidates = []
                        for path in folder.glob(lane+'.*.json'):
                            try:
                                with path.open('r+') as owner:
                                    try:
                                        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    except BlockingIOError:
                                        item = json.load(owner)
                                    else:
                                        # File locks die with their process; PID reuse
                                        # cannot leave an abandoned request blocking work.
                                        path.unlink(missing_ok=True)
                                        continue
                            except (OSError, ValueError):
                                continue
                            if lane != 'download' or download_backoff(item.get('url', ''), folder) <= time.time():
                                candidates.append((rank.get(item['job'], len(rank)), item['ready'], path.name))
                        limited = lane == 'download' and download_backoff(self.request['url'], folder) > time.time()
                        if not limited and (not candidates or min(candidates)[2] == request.name):
                            break
                        fcntl.flock(slot, fcntl.LOCK_UN)
                    except BlockingIOError:
                        pass
                    if not waiting:
                        with self.lock:
                            self.active.add(name)
                            self.stage_progress[name] = dict(waiting=True, detail='等待处理资源')
                            self.publish_progress()
                        waiting = True
                    time.sleep(.05)
                request.unlink(missing_ok=True)
                yield
            finally:
                request.unlink(missing_ok=True)
                if slot is not None:
                    fcntl.flock(slot, fcntl.LOCK_UN)

    def run_subprocess(self, name, command, timeout=3600):
        self.phase(name, True)
        self.stage_event(name, 'started')
        started = time.perf_counter()
        succeeded = False
        try:
            with (self.output/f'{name}.log').open('wb') as log, selectors.DefaultSelector() as selector:
                with subprocess.Popen([str(a) for a in command], cwd=ROOT.parent,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as result:
                    assert result.stdout is not None
                    stdout = cast(BufferedReader, result.stdout)
                    selector.register(stdout, selectors.EVENT_READ)
                    pending = b''
                    try:
                        while True:
                            remaining = timeout-(time.perf_counter()-started)
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(command, timeout)
                            if not selector.select(remaining):
                                raise subprocess.TimeoutExpired(command, timeout)
                            block = stdout.read1(65536)
                            log.write(block)
                            log.flush()
                            lines = (pending+block).split(b'\n')
                            pending = lines.pop() if block else b''
                            for line in lines:
                                if name == 'download' and line == b'KARAOKE_SOURCE_READY':
                                    self.source_metadata_ready()
                                value = parse_progress(line.decode(errors='replace'))
                                if value is not None:
                                    with self.lock:
                                        self.stage_progress[name] = value
                                        self.publish_progress()
                            if not block:
                                break
                        result.wait(timeout=max(.001, timeout-(time.perf_counter()-started)))
                    except BaseException:
                        result.kill()
                        raise
            if result.returncode:
                with (self.output/f'{name}.log').open('rb') as log:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell()-16384))
                    diagnostic = log.read().decode(errors='replace')
                if name in ('download', 'download_video'):
                    message = download_failure(diagnostic)
                    policy = retry_fields(dict(code=name+'_failed', message=message), diagnostic, 1, failed_at=time.time())
                    if policy['provider_blocked']:
                        # Publish while still holding the download lane, before
                        # another song can make a request to the same provider.
                        download_backoff(self.request['url'], until=policy['next_retry_at'])
                    raise StageFailure(name, message)
                if name == 'align_words' and any(message in diagnostic for message in (
                        'Empty alignment audio', 'FFmpeg failed to read "[]"')):
                    raise StageFailure(name, '歌词对齐失败：歌词时间范围内没有有效音频，请检查歌词与视频版本')
                if name == 'validate_timing' and 'contradicted acoustic onsets' in diagnostic:
                    raise StageFailure(name, '部分字幕时间仍与实际演唱冲突；音轨和局部修复诊断已保留')
                raise StageFailure(name, f'{name} 阶段失败；具体错误已保存在任务日志中')
            succeeded = True
        except subprocess.TimeoutExpired as error:
            raise StageFailure(name, f'{name} 阶段超过允许的处理时间') from error
        finally:
            self.stage_event(name, 'finished', status='completed' if succeeded else 'failed')
            with self.lock:
                self.timings[name] = round(self.timings.get(name, 0)+time.perf_counter()-started, 3)
            self.phase(name, False)

    def inputs(self, audio, name):
        import soundfile as sf
        write_json(self.output/name, [dict(id='song', input=str(audio), input_sha256=file_hash(audio),
                                          duration_seconds=sf.info(audio).duration)])

    def source_metadata_ready(self):
        """yt-dlp has written metadata, but need not have downloaded any media."""
        if self.metadata_ready is not None:
            metadata = json.loads((self.output/'download/source.info.json').read_text())
            duration = metadata.get('duration')
            if isinstance(duration, (int, float)) and math.isfinite(duration) and 0 < duration <= self.max_duration+1:
                callback, self.metadata_ready = self.metadata_ready, None
                self.phase(song_info={**video_song_info(metadata), **self.song_info})
                self.source_announced = True
                callback(metadata, duration)

    def download(self, *, artwork=True, format_id=None):
        artwork = artwork and not self.audio_only and not self.audio_first
        cached = self.output/'source.json'
        if cached.exists() and self.source.is_file() and (self.output/'download/source.info.json').is_file():
            started = time.perf_counter()
            try:
                source = json.loads(cached.read_text())
                metadata = json.loads((self.output/'download/source.info.json').read_text())
                valid = (isinstance(source, dict) and isinstance(metadata, dict)
                    and source.get('input_url', source.get('url')) == self.request['url']
                    and source['sha256'] == file_hash(self.source)
                    and type(source['duration']) in (int, float) and math.isfinite(source['duration']))
            except (OSError, ValueError, KeyError, TypeError) as error:
                valid = False
                write_json(self.output/'download-cache-error.json', dict(error=type(error).__name__, reason=str(error)))
            if valid:
                if not 0 < source['duration'] <= self.max_duration+1:
                    raise StageFailure('download', '缓存视频超过当前时长上限')
                self.source_metadata_ready()
                if source.get('audio_only') and not self.audio_only and not self.audio_first:
                    metadata = self.promote_video(metadata, source)
                if not self.source_announced:
                    self.phase(song_info={**video_song_info(metadata), **self.song_info})
                if artwork and not (self.output/'poster.jpg').exists():
                    create_poster(self.video, self.output/'poster.jpg', seconds=min(30, source['duration']*.1))
                self.timings['reuse_download'] = round(time.perf_counter()-started, 3)
                return metadata, source['duration']
        if os.getenv('KARAOKE_REQUIRE_CACHED_SOURCE') == '1':
            raise StageFailure('download', '离线处理需要同源、哈希有效的完整音频缓存')
        if (self.output/'download').exists():
            shutil.rmtree(self.output/'download')
        (self.output/'download').mkdir()
        url = self.request['url']
        command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--js-runtimes', 'node', '--no-playlist',
            '--newline', '--progress', '--progress-delta', '.5', '--progress-template',
            'download:KARAOKE_STAGE_PROGRESS {"completed":%(progress.downloaded_bytes)j,'
            '"total":%(progress.total_bytes)j,"unit":"bytes","vcodec":%(info.vcodec)j}',
            '--progress-template', 'postprocess:KARAOKE_STAGE_PROGRESS {"detail":"合并视频"}',
            '--write-info-json', '--print', 'before_dl:KARAOKE_SOURCE_READY', '--no-simulate',
            '--no-overwrites', '--retries', '2', '--fragment-retries', '2',
            '--socket-timeout', '25', '--http-chunk-size', '10M', '--max-filesize', '2G', '--match-filter',
            f'!is_live & duration <= {self.max_duration}', '-f', format_id or (
                'ba' if self.audio_only else 'ba/b[height<=720]' if self.audio_first else 'bv[height<=720]+ba/b[height<=720]'),
            '--merge-output-format', 'mkv', '--remux-video', 'mkv',
            '-o', self.output/'download/source.%(ext)s']
        if self.cookies_file.is_file():
            command.extend(['--cookies', self.cookies_file])
        command.append(url)
        self.run('download', command, timeout=900)
        info_path = self.output/'download/source.info.json'
        if not self.source.exists() or not info_path.exists():
            raise StageFailure('download', '没有下载到完整视频；请检查链接、直播状态、时长或站点登录要求')
        metadata = json.loads(info_path.read_text())
        self.fresh_source_metadata = True
        self.source_metadata_ready()
        if not self.source_announced:
            self.phase(song_info={**video_song_info(metadata), **self.song_info})
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(self.source)]))
        required = {'audio'} if self.audio_only or self.audio_first else {'audio', 'video'}
        if not required <= {s['codec_type'] for s in probe['streams']}:
            raise StageFailure('download', '输入必须同时包含视频和音频')
        duration = float(probe['format']['duration'])
        if not 0 < duration <= self.max_duration+1:
            raise StageFailure('download', f'视频时长必须介于 0 和 {self.max_duration} 秒之间')
        if artwork:
            create_poster(self.source, self.output/'poster.jpg', thumbnail=metadata.get('thumbnail'),
                          seconds=min(30, duration*.1))
        write_json(self.output/'source.json', dict(url=url, input_url=self.request['url'], sha256=file_hash(self.source),
            duration=duration, probe=probe, audio_only='video' not in {s['codec_type'] for s in probe['streams']}))
        return metadata, duration

    def promote_video(self, metadata, source):
        """Add this video's picture while preserving the already tested audio stream."""
        folder = self.output/'video-download'
        video, info = folder/'source.mkv', folder/'source.info.json'
        with self.checkpoints.stage('download_video', inputs=[self.source], outputs=[video, info],
                config=dict(url=self.request['url'], format='bv[height<=720]/b[height<=720]')) as reused:
            if not reused:
                folder.mkdir(exist_ok=True)
                command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--js-runtimes', 'node', '--no-playlist',
                    '--write-info-json', '--no-overwrites', '--retries', '2', '--fragment-retries', '2',
                    '--socket-timeout', '25', '--http-chunk-size', '10M', '--max-filesize', '2G', '--match-filter',
                    f'!is_live & duration <= {self.max_duration}', '-f', 'bv[height<=720]/b[height<=720]',
                    '--remux-video', 'mkv', '-o', folder/'source.%(ext)s']
                if self.cookies_file.is_file():
                    command.extend(['--cookies', self.cookies_file])
                if self.fresh_source_metadata and metadata.get('formats'):
                    command[2] = 'karaoke.download_from_info'
                    command.extend(['--load-info-json', self.output/'download/source.info.json'])
                else:
                    command.append(self.request['url'])
                self.run('download_video', command, timeout=900)
        added = json.loads(info.read_text())
        if not metadata.get('id') or (added.get('id'), added.get('extractor_key')) != (
                metadata['id'], metadata.get('extractor_key')):
            raise StageFailure('verify_video', '补充的视频与已测试音频来源不同')
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(video)]))
        duration = float(probe['format']['duration'])
        if (not any(s['codec_type'] == 'video' for s in probe['streams']) or
                not math.isfinite(duration) or abs(duration-source['duration']) > 1):
            raise StageFailure('verify_video', '补充视频与已测试音频时长不匹配')
        # Rendering resets video timestamps and supplies the verified stem mix.
        # The original audio never needs copying into this intermediate picture.
        self.video = video
        (folder/'with-audio.mkv').unlink(missing_ok=True)
        (self.checkpoints.directory/'prepare_video.json').unlink(missing_ok=True)
        return dict(metadata, format_id=added['format_id']+'+'+metadata['format_id'])

    def prepare_text(self, metadata, duration, lead_task=None):
        if not self.strict_audio:
            from .lyric_lookup import (
                prepare_online_lyrics,
                prepared_lyrics_identity,
                read_prepared_lyrics,
            )
            self.phase('prepare_lyrics', True)
            self.stage_event('prepare_lyrics', 'ready')
            self.stage_event('prepare_lyrics', 'started')
            started = time.perf_counter()
            try:
                if self.prepared_lyrics:
                    error = json.loads(self.prepared_lyrics.read_text()).get('error')
                    if error:
                        raise RuntimeError('Prepared lyric lookup failed: '+error['type'])
                    lyrics = read_prepared_lyrics(self.prepared_lyrics, self.request['url'], metadata)
                else:
                    cached = self.output/'prepared-lyrics.json'
                    keys = ('title', 'artist', 'language', 'lyrics')
                    overrides = {key:self.request.get(key) for key in keys}
                    selection = dict(duration=duration, metadata_model=self.text_model)
                    lyrics = None
                    if cached.is_file() and not self.request.get('force'):
                        try:
                            packet = json.loads(cached.read_text())
                            if (packet.get('request_overrides', dict.fromkeys(keys)) == overrides
                                    and packet.get('selection') == selection):
                                lyrics = read_prepared_lyrics(cached, self.request['url'], metadata)
                                if not lyrics or not lyrics.get('lines'):
                                    lyrics = None
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                            write_json(self.output/'prepared-lyrics-cache-error.json', dict(reason=str(error)))
                    if lyrics is None:
                        lyrics = (prepare_online_lyrics(self.request, metadata, duration, self.checkpoints,
                                                        self.text_model, self.catalog_root)
                                  if not local_asr.offline() or self.request.get('lyrics') else None)
                        if lyrics is not None:
                            write_json(cached, dict(identity=prepared_lyrics_identity(self.request['url'], metadata),
                                                   request_overrides=overrides, selection=selection, lyrics=lyrics))
                if lyrics is not None:
                    write_json(self.output/'lyrics.json', lyrics)
                    # An empty anchor list selects direct acoustic alignment.
                    # Neither the lyric provider nor a fabricated transcript supplies times.
                    write_json(self.output/'anchors.json', [])
                    self.alignment_outputs.add(self.output/'anchors.json')
                    self.phase(song_info={key: lyrics[key] for key in ('title', 'artist', 'language')})
                    return lyrics
                raise StageFailure('prepare_lyrics', '未找到可靠的已发表歌词；保留音轨，暂不生成猜测歌词')
            finally:
                with self.lock:
                    self.timings['prepare_lyrics'] = round(time.perf_counter()-started, 3)
                self.phase('prepare_lyrics', False)
                self.stage_event('prepare_lyrics', 'finished')
        language = self.request.get('language') or 'auto'
        # The independent ASR check can use the lead stem as an alternate input.
        if lead_task is not None:
            lead_task.result()
        lead = self.output/'lead-separation/song/vocals.wav'
        self.run('transcribe', [sys.executable, ROOT/'transcribe_lyrics.py', '--audio',
            self.output/'vocals-asr.mp3', '--language', language,
            '--original-audio', self.output/'source-44100.wav',
            *(['--lead-audio', lead] if lead.is_file() else []),
            '--allow-unresolved-text', '--output', self.output/'asr'])
        self.phase('prepare_lyrics', True)
        started = time.perf_counter()
        try:
            from .source_captions import fetch_manual_captions
            transcription = json.loads((self.output/'asr/transcription.json').read_text())
            captions = None if self.request.get('lyrics') or local_asr.offline() else fetch_manual_captions(
                self.output/'download/source.info.json', song_language(transcription.get('language')), self.checkpoints)
            catalog = list((self.catalog_root/'lyrics').glob('*.json')) if self.catalog_root else []
            with self.checkpoints.stage('prepare_lyrics',
                    inputs=[self.output/'asr/transcription.json', self.output/'vocals-16k.wav', self.vocals, self.output/'source-44100.wav',
                            *sorted((self.output/'asr').glob('initial-transcription.json')),
                            *sorted((self.output/'asr').glob('run.json')), *([captions] if captions else [])],
                    outputs=[self.output/'lyrics.json', self.output/'anchors.json', self.output/'asr/filtered.json',
                             self.output/'asr/anchor-recovery.json', self.output/'asr/lyric-repeats.json'],
                    config=dict(request=self.request,
                                metadata={k: metadata.get(k) for k in ('title', 'track', 'artist', 'artists', 'uploader', 'description')},
                                duration=duration, model=self.text_model, asr=local_asr.configuration(),
                                implementation=inspect.getsource(type(self).prepare_lyrics)),
                    code=[ROOT/p for p in ('song_sources.py', 'source_captions.py', 'lyric_lookup.py', 'anchors_from_asr.py', 'transcribe_lyrics.py', 'local_asr.py', 'anchor_recovery.py',
                                           'subtitles.py', 'job_schema.py', 'vocal_review.py')] + catalog) as reused:
                if not reused:
                    self.prepare_lyrics(metadata, duration)
            lyrics = json.loads((self.output/'lyrics.json').read_text())
            self.phase(song_info=dict(title=lyrics['title'], artist=lyrics['artist'],
                                      duration_seconds=duration, language=lyrics['language']))
        finally:
            with self.lock:
                self.timings['prepare_lyrics'] = round(time.perf_counter()-started, 3)
            self.phase('prepare_lyrics', False)
        return lyrics

    def prepare_lyrics(self, metadata, duration):
        import soundfile as sf

        from anchor_recovery import recover_anchor_words
        from anchors_from_asr import anchors_from_words
        from transcribe_lyrics import filter_silent_transcription
        transcription = json.loads((self.output/'asr/transcription.json').read_text())
        report_path = self.output/'asr/run.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        initial_path = self.output/'asr/initial-transcription.json'
        initial = json.loads(initial_path.read_text()) if initial_path.is_file() else None
        if initial and report.get('recovery', {}).get('text_selection', {}).get('selected') == 'initial_transcription':
            original_segments = initial.get('segments') or []
            original_text = ''.join(s['text'] for s in original_segments)
            if ''.join(original_text.split()) == ''.join(transcription.get('text', '').split()):
                # Recovery kept the original text but flattened its boundaries.
                # Restore them for silence filtering only, keeping independent
                # recovered word times as the alignment evidence.
                transcription = dict(transcription, segments=original_segments)
        language = song_language(transcription.get('language'))
        if language:
            self.phase(song_info=dict(language=language))
        audio, rate = sf.read(self.output/'vocals-16k.wav', dtype='float32')
        transcription, rejected = filter_silent_transcription(transcription, audio, rate)
        write_json(self.output/'asr/filtered.json', dict(transcription=transcription, rejected=rejected))
        lyrics = prepare_song_text(self.request, metadata, transcription, duration,
            catalog_root=self.catalog_root, model=self.text_model, cache=self.checkpoints)
        caption_path = self.output/'source-captions.json'
        if lyrics['provenance']['kind'] == 'asr_draft' and caption_path.is_file():
            from .source_captions import caption_candidate, manual_track
            packet = json.loads(caption_path.read_text())
            # A changed input must not reuse a formerly matching subtitle track.
            if packet.get('video_id') == metadata.get('id') and manual_track(metadata, lyrics['language']):
                lines, caption_report = caption_candidate(packet, transcription, lyrics['language'], duration)
                lyrics['provenance']['source_captions'] = caption_report
                if lines is not None:
                    lyrics.update(lines=lines, provenance=dict(lyrics['provenance'], kind='source_manual_subtitles'))
        if lyrics['provenance']['kind'] == 'asr_draft' and not local_asr.offline():
            from .lyric_lookup import fetch_lyrics, lyric_candidate
            packet = fetch_lyrics(lyrics, self.output/'download/source.info.json', self.checkpoints)
            lines, lookup_report = lyric_candidate(packet, lyrics, transcription, duration,
                audio_path=self.output/'source-44100.wav', source_title=metadata.get('title'))
            lyrics['provenance']['lyric_lookup'] = lookup_report
            if lines is not None:
                lyrics.update(lines=lines, provenance=dict(lyrics['provenance'], kind='online_lyric_text',
                                                          text_transform='whitespace_only'))
        candidates = {'transcription': transcription.get('words') or []}
        if lyrics['provenance']['kind'] != 'asr_draft' and initial:
            # Recovery of one bad passage need not discard usable whole-song
            # anchors. Only compare them against independently supplied lyrics.
            if song_language(initial.get('language')) == lyrics['language']:
                initial, _ = filter_silent_transcription(initial, audio, rate)
                words = [w for w in initial.get('words') or [] if 0 <= w['start'] < w['end'] <= duration]
                if words and all(a['start'] <= b['start'] for a, b in zip(words, words[1:])):
                    candidates['initial_transcription'] = words
        matches = {}
        for name, words in candidates.items():
            cues = anchors_from_words(lyrics['lines'], words, lyrics['language'], allow_missing=True)
            coverage = sum(c['matched_fraction']*len(c['text']) for c in cues)/max(1, sum(len(c['text']) for c in cues))
            matches[name] = (cues, coverage)
        selected = max(matches, key=lambda name: matches[name][1])
        anchors, covered = matches[selected]
        recovery_path = self.output/'asr/anchor-recovery.json'
        write_json(recovery_path, dict(attempts=[]))
        if covered < .55 and (lyrics['provenance']['kind'] != 'asr_draft' or report.get('recovery', {}).get('triggered')):
            raw_audio = self.output/'source-44100.wav'
            candidates[selected], anchors, covered = recover_anchor_words(None,
                self.output/'vocals-16k.wav', lyrics['lines'], candidates[selected], lyrics['language'],
                recovery_path, self.checkpoints, .55, duration,
                fallback_audio=raw_audio if raw_audio.is_file() else None)
        for _ in range(2):
            repeats = dict(reference_lines=lyrics['lines'], words=candidates[selected], expansions=[], status='unavailable')
            if covered >= .55:
                lines, anchors, repeats = restore_repetitions(lyrics['lines'], anchors, candidates[selected],
                    lyrics['language'], duration, allow_expansion=lyrics['provenance']['kind'] == 'video_description')
                lyrics = dict(lyrics, lines=lines)
                if repeats['expansions']:
                    lyrics['provenance'].update(text_transform='source_repetitions',
                                                repetitions=repeats['expansions'])
                covered = sum(c['matched_fraction']*len(c['text']) for c in anchors)/max(1, sum(map(len, lines)))
            coverage = lyric_coverage(anchors, candidates[selected], lyrics['language'], duration,
                                      other_words=transcription.get('words') or [])
            if covered >= .55 and coverage['uncovered_passages']:
                from .vocal_review import review_gaps, speech_intervals
                reviews = review_gaps(self.output, coverage['uncovered_passages'],
                                      [*candidates[selected], *(transcription.get('words') or [])])
                repeats['non_singing_reviews'] = reviews
                coverage = lyric_coverage(anchors, candidates[selected], lyrics['language'], duration,
                    other_words=transcription.get('words') or [], speech_intervals=speech_intervals(self.output, reviews))
            if lyrics['provenance']['kind'] != 'video_description' or (covered >= .55 and coverage['status'] == 'complete'):
                break
            # A description can quote a few perfectly matching lines while
            # omitting most of the song. The replacement needs separate word
            # evidence, just as it does for an unrelated diary or synopsis.
            support = independent_transcript_support(transcription, report, lyrics['language'])
            if support is not None and support >= .55:
                rejected_description = dict(lines=lyrics['lines'], provenance=lyrics['provenance'],
                                            anchor_coverage=covered, lyric_coverage=coverage)
                lyrics = dict(lyrics, lines=transcript_lines(transcription, lyrics['language']),
                    provenance=dict(kind='asr_draft', needs_review=True, text_transform='whitespace_only',
                        transcription_sha256=lyrics['provenance']['transcription_sha256'],
                        independent_word_support=support, rejected_description=rejected_description))
                anchors = anchors_from_words(lyrics['lines'], candidates['transcription'],
                                             lyrics['language'], allow_missing=True)
                covered = sum(c['matched_fraction']*len(c['text']) for c in anchors)/max(1, sum(map(len, lyrics['lines'])))
                selected = 'independently_corroborated_transcription'
                matches[selected] = (anchors, covered)
                candidates[selected] = candidates['transcription']
            else:
                break
        repeats['lyric_coverage'] = coverage
        repeats['other_coverage_words'] = transcription.get('words') or []
        write_json(self.output/'asr/lyric-repeats.json', repeats)
        write_json(self.output/'asr/filtered.json', dict(transcription=transcription, rejected=rejected,
            anchor_source=selected, anchor_coverage={name: match[1] for name, match in matches.items()},
            final_anchor_coverage=covered))
        write_json(self.output/'lyrics.json', lyrics)
        write_json(self.output/'anchors.json', anchors)
        if covered < .55 or not any(c['status'] == 'candidate' for c in anchors):
            raise StageFailure('prepare_lyrics', '歌词与这个视频的演唱匹配不足；请提交本版本的 lyrics 后重试')
        if repeats['status'] != 'complete':
            raise StageFailure('prepare_lyrics', '检测到歌词缺少重复唱段，但证据不足以自动恢复；详情已保留')
        if coverage['status'] != 'complete':
            raise StageFailure('prepare_lyrics', '歌词未覆盖完整演唱；遗漏唱段的识别证据已保留')

    def resolve_lyric_language(self, lyrics):
        """Use optional audio evidence for pronunciation and conflicting versions."""
        self.recording_language = None
        from .lyric_lookup import text_language
        texts = lyrics.get('provenance', {}).get('lyric_lookup', {}).get('candidate_texts', [])
        languages = {text_language(''.join(text)) for text in texts}
        if lyrics['language'] != 'Chinese' and len(languages - {None}) <= 1:
            return lyrics
        audio, path = self.output/'vocals-16k.wav', self.output/'song-language.json'
        weights = [ROOT/'models/alignment/sensevoice'/name for name in ('model.int8.onnx', 'tokens.txt')]
        try:
            from importlib.metadata import version
            identity = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in weights]
            with self.checkpoints.stage('detect_language', inputs=[audio], outputs=[path],
                    config=dict(weights=identity, sherpa_onnx=version('sherpa-onnx')),
                    code=[ROOT/'local_asr.py', ROOT/'job_schema.py']) as reused:
                if not reused:
                    self.run('detect_language', [sys.executable, ROOT/'local_asr.py', '--detect-language',
                             audio, path])
            evidence = json.loads(path.read_text())
        except (ImportError, OSError, ValueError, StageFailure) as error:
            # Classification is optional; an unavailable detector must not
            # discard known lyrics or become another preparation failure.
            write_json(self.output/'song-language-error.json', dict(error=str(error)))
            return lyrics
        self.recording_language = evidence.get('language')
        if self.recording_language != 'Cantonese' or lyrics['language'] != 'Chinese':
            return lyrics
        lyrics = dict(lyrics, language='Cantonese', provenance=dict(lyrics['provenance'],
            language_detection=evidence))
        write_json(self.output/'lyrics.json', lyrics)
        self.phase(song_info=dict(language='Cantonese'))
        return lyrics

    def align(self, lyrics, duration):
        from lyric_progress import build
        from subtitles import (
            collapsed_phrase,
            countdown_events,
            highlight_coverage_errors,
            normalized,
            presentation_cues,
        )
        language = lyrics['language']
        output = self.output
        write_json(output/'alignment-jobs.json', [dict(audio=str(output/'vocals-16k.wav'),
            original_audio=str(output/'source-44100.wav'), lyrics=str(output/'lyrics.json'),
            anchors=str(output/'anchors.json'), output=str(output/'alignment'))])
        self.run('align_words', [sys.executable, ROOT/'align_lyrics_whisper.py', '--jobs',
            output/'alignment-jobs.json', '--device', self.device])
        raw = json.loads((output/'alignment/raw.json').read_text())
        if alignment_recovery_summary(raw)['unresolved_localization_lines']:
            raise StageFailure('align_words', '部分唱句仍与识别位置冲突；定位诊断已保留')
        coarse, _ = presentation_cues(json.loads((output/'alignment/cues.json').read_text()),
                                     duration, lead=0, tail=0, bridge_gap=0)
        words, mapping = build(coarse, raw, language)
        write_json(output/'word-cues.json', words)
        self.alignment_outputs.add(output/'word-cues.json')
        if mapping['fallback_lines']:
            raise StageFailure('align_words', '部分歌词无法生成时间轴；请检查本版本歌词')
        order_fallbacks = []
        english_context_scope = None
        if language in ('Chinese', 'Japanese'):
            job = dict(song='song', language=language, audio=str(self.vocals),
                       cues=str(output/'word-cues.json'), output=str(output/'phonemes'))
            write_json(output/'phoneme-jobs.json', [job])
            self.run('align_phonemes', [sys.executable, ROOT/'align_lyric_phonemes.py',
                '--jobs', output/'phoneme-jobs.json', '--device', self.device])
            job.update(cues=str(output/'phonemes/cues.json'), coarse_cues=str(output/'word-cues.json'),
                       output=str(output/'repeated-lyrics'))
            write_json(output/'repeat-jobs.json', [job])
            self.run('align_repeats', [sys.executable, ROOT/'align_repeated_lyrics.py',
                '--jobs', output/'repeat-jobs.json', '--device', self.device])
            from align_lyric_phonemes import protect_line_order
            ordered, order_fallbacks = protect_line_order(
                json.loads((output/'repeated-lyrics/cues.json').read_text()), words)
            write_json(output/'ordered-cues.json', ordered)
            self.alignment_outputs.add(output/'ordered-cues.json')
            job.update(cues=str(output/'ordered-cues.json'), duration=duration,
                       original_audio=str(output/'source-44100.wav'),
                       output=str(output/'context/song'))
            write_json(output/'context-jobs.json', [job])
            self.run('align_context', [sys.executable, ROOT/'experiments/cjk-onsets-v003/run.py',
                '--jobs', output/'context-jobs.json', '--output', output/'context', '--device', self.device])
            cues = json.loads((output/'context/song/cues.json').read_text())
        elif language == 'English':
            from onset_consensus import blend_cues, context_scope_violations
            locations = phrase_locations(lyrics['lines'], raw)
            write_json(output/'english-anchors.json', [locations.get(i, cue) for i, cue in enumerate(words)])
            job = dict(audio=str(output/'vocals-16k.wav'), lyrics=str(output/'lyrics.json'),
                       anchors=str(output/'english-anchors.json'), output=str(output/'english-onsets'))
            write_json(output/'english-jobs.json', [job])
            self.run('align_phonemes', [sys.executable, ROOT/'align_english_onsets.py', '--method', 'singing',
                '--onset-ratio', '.02', '--full-song', '--jobs', output/'english-jobs.json', '--device', self.device])
            report = json.loads((output/'english-onsets/run.json').read_text())
            if report['failures']:
                raise StageFailure('align_phonemes', '部分英文歌词无法可靠对齐；详情保存在音素对齐日志中')
            english_context_scope = report['scope_blocks']
            primary = json.loads((output/'english-onsets/cues.json').read_text())
            cues = blend_cues(primary, words, max_disagreement=.2, primary_weight=.75)
            if context_scope_violations(cues, english_context_scope):
                raise StageFailure('align_words', '英文词起点落在已定位的唱句上下文之外，需重新对齐')
            cues = [dict(c, status='candidate') for c in cues]
            cues, _ = presentation_cues(cues, duration, lead=0, tail=0, bridge_gap=0)
        else:
            # Languages without dedicated singing weights retain the measured
            # multilingual word alignment and still undergo acoustic review.
            cues = words
        self.aligned_cues = cues
        units = [u for c in cues for u in c['units']]
        if not units or [c['text'] for c in cues] != lyrics['lines']:
            raise StageFailure('align_words', '最终字幕没有完整覆盖输入歌词')
        if any(not 0 <= u['start'] < u['end'] <= duration+.05 for u in units) or any(
                a['start'] >= b['start'] for a, b in zip(units, units[1:])):
            raise StageFailure('align_words', '歌词时间轴仍有冲突；需要核对歌词与实际演唱')
        for cue in cues:
            if normalized(''.join(u['text'] for u in cue['units'])) != normalized(cue['text']):
                raise StageFailure('align_words', '逐字时间轴遗漏了歌词')
            if collapsed_phrase(cue, language):
                raise StageFailure('align_words', '部分唱句时长异常压缩；局部对齐诊断已保留')
        highlight_errors = highlight_coverage_errors(cues, language)
        write_json(output/'highlight-validation.json', dict(errors=highlight_errors))
        if highlight_errors:
            raise StageFailure('align_words', '部分字词缺少独立的高亮起点；对齐诊断已保留')
        direct = lyrics['provenance'].get('route') == 'lyrics_first'
        if direct:
            missing = None
            restored_repetitions = len(lyrics['provenance'].get('repetitions', []))
        else:
            repeats = json.loads((output/'asr/lyric-repeats.json').read_text())
            missing = missing_repetitions(repeats['reference_lines'], cues, repeats['words'], language, duration)
            restored_repetitions = len(repeats['expansions'])
        write_json(output/'lyric-repeat-validation.json', dict(missing=missing))
        if missing:
            raise StageFailure('align_words', '最终字幕仍遗漏已识别出的重复唱段；详情已保留')
        write_json(output/'cues.json', cues)
        write_json(output/'quality.json', dict(lines=len(cues), units=len(units),
            lyrics_provenance=lyrics['provenance'], onset_accuracy=None, separation_purity=None,
            english_context_scope=english_context_scope,
            independent_asr_coverage=None if direct else True,
            restored_repetitions=restored_repetitions,
            fallback_units=sum(u.get('timing_source') == 'prior_word_fallback' for u in units),
            **alignment_recovery_summary(raw),
            line_order_fallback_lines=order_fallbacks,
            countdown_events=countdown_events(cues)))

    def align_candidates(self, lyrics, metadata, duration, lead=None):
        from .lyric_consensus import (
            alternative_lyric_candidates,
            expanded_lyric_anchors,
            observed_repetition_candidate,
            observed_source_alternative,
        )
        from .lyric_lookup import published_lyric_lines
        from .subtitles import normalized
        packet = lyrics.get('provenance', {}).get('lyric_lookup', {}) if not self.strict_audio else {}
        path = self.output/'lyric-crosscheck.json'
        if not self.strict_audio and 'candidate_texts' not in packet:
            try:
                packet = json.loads(path.read_text())
            except (OSError, ValueError):
                packet = {}
        candidates, seen = [], set()
        for candidate in [lyrics, *alternative_lyric_candidates(lyrics, packet)]:
            if candidate.get('provenance', {}).get('kind') == 'online_lyric_text':
                provenance = candidate['provenance']
                lines, excluded = published_lyric_lines('\n'.join(candidate['lines']),
                    candidate['title'], [candidate['artist']], album=provenance.get('lyric_lookup', {}).get('album'))
                if excluded:
                    candidate = dict(candidate, lines=lines, provenance=dict(provenance,
                        lyric_lookup=dict(provenance.get('lyric_lookup', {}),
                            excluded_cached_credit_lines=excluded)))
            key = normalized(''.join(candidate['lines'])).casefold()
            if key and key not in seen:
                candidates.append(candidate)
                seen.add(key)
        if not candidates:
            raise StageFailure('lyrics', '未找到可用的歌词正文')
        # A catalogue can attach another language version even to the right
        # video ID. Audio evidence orders existing candidates; it never deletes
        # text, changes the supplied words or relaxes their alignment checks.
        before = [c['language'] for c in candidates]
        if self.recording_language:
            matches = lambda language: language == self.recording_language or (
                self.recording_language == 'Cantonese' and language == 'Chinese')
            candidates.sort(key=lambda c: not matches(c['language']))
        source_order = dict(recording_language=self.recording_language, before=before,
                            after=[c['language'] for c in candidates])
        attempts, previous, recording_observations = [], None, {}
        candidate_reports = ('playback-warning.json', 'lyric-source-support.json',
            'lyric-repeat-validation.json', 'localization-validation.json',
            'timing-context-recovery.json', 'timing-recovery.json')
        for index, candidate in enumerate(candidates):
            if candidate is not lyrics and self.recording_language == 'Cantonese':
                candidate = self.resolve_lyric_language(candidate)
            for name in candidate_reports:
                (self.output/name).unlink(missing_ok=True)
            if index or candidate is not lyrics:
                self.aligned_cues = None
                self.alignment_outputs.clear()
                write_json(self.output/'lyrics.json', candidate)
                self.phase(song_info={k:candidate[k] for k in ('title', 'artist', 'language')})
            error, preferred = None, None
            observations = list(recording_observations.values())
            try:
                try:
                    if previous and candidate['provenance'].get('route') == 'lyrics_first':
                        # This prior was reviewed on this recording in this run.
                        # Preserve its phrase locations before coarse alignment,
                        # not only after an expanded text has already succeeded.
                        prior_cues = json.loads((previous/'cues.json').read_text())
                        anchors = expanded_lyric_anchors(candidate,
                            json.loads((previous/'lyrics.json').read_text()), prior_cues,
                            json.loads((previous/'timing-checked/report.json').read_text()), duration
                        ) if prior_cues == json.loads((previous/'timing-checked/cues.json').read_text()) else []
                        write_json(self.output/'anchors.json', anchors)
                    self.align(candidate, duration)
                except (StageFailure, ValueError) as caught:
                    if self.strict_audio:
                        raise
                    error = caught
                    candidate = self.playback_fallback(metadata, duration, caught)
                finally:
                    raw_path = self.output/'alignment/raw.json'
                    if raw_path.is_file():
                        for block in json.loads(raw_path.read_text()):
                            for observation in block.get('asr_localizations', []):
                                if observation.get('recognition'):
                                    key = json.dumps([observation['source_range'], observation['recognition']], sort_keys=True)
                                    recording_observations.setdefault(key, observation)
                    # Recognition belongs to the recording, not a candidate's line
                    # indexes. A narrower retry must not forget an observed verse.
                    observations = list(recording_observations.values())
                    self.recording_observations = [dict(o, context=[0, len(candidate['lines'])]) for o in observations]
                # The final timing review can expose a gap previously hidden
                # by a held caption. Keep it in the same bounded source-repair
                # loop; every replacement must pass this review again. Its
                # localization gate runs after the acoustic repair, not before.
                self.validate_alignment(candidate, duration, lead, previous=previous)
                remaining = candidates[index+1:]
                missing = observed_source_alternative(candidate, [*remaining, *candidates[:index]], observations,
                    cues=json.loads((self.output/'cues.json').read_text()), duration=duration)
                if missing:
                    # Failed sources still provide text evidence, but only an
                    # untried source may be scheduled again.
                    write_json(self.output/'lyric-source-support.json', missing)
                    if missing['alternative'] < len(remaining):
                        preferred = missing
                    raise StageFailure('subtitles', '备用歌词包含已听到但尚未覆盖的唱段；正在核验备用正文')
            except StageFailure as failure:
                if not self.strict_audio and failure.stage == 'subtitles' and observations:
                    # Text overlap orders retries after an actual failure; it
                    # cannot veto a complete, acoustically checked candidate.
                    preferred = preferred or observed_source_alternative(candidate, candidates[index+1:], observations)
                    if preferred:
                        position = index+1+preferred['alternative']
                        candidates[index+1], candidates[position] = candidates[position], candidates[index+1]
                        write_json(self.output/'lyric-source-support.json', preferred)
                    else:
                        repeat_report = self.output/'lyric-repeat-validation.json'
                        restored = json.loads(repeat_report.read_text()).get('restored', []) if repeat_report.exists() else []
                        # Misplacing an already inserted repetition is an
                        # alignment failure, not evidence for another copy.
                        repeated = None if any(r['conflict'] for r in restored) else observed_repetition_candidate(
                            candidate, json.loads((self.output/'cues.json').read_text()), observations, duration)
                        key = normalized(''.join(repeated['lines'])).casefold() if repeated else None
                        if key and key not in seen:
                            seen.add(key)
                            candidates.insert(index+1,repeated)
                if (self.strict_audio or failure.stage != 'subtitles' or index+1 == len(candidates) or
                        not (self.checkpoints.directory/'align_words.json').is_file()):
                    raise
                # Preserve the rejected whole text and its actual alignment,
                # then let the normal stage cache rebuild for the next text.
                folder = self.output/'lyric-source-attempts'/str(time.time_ns())
                folder.mkdir(parents=True)
                write_json(folder/'failure.json', dict(stage=failure.stage, message=str(failure)))
                for name in ('lyrics.json', 'alignment/raw.json', 'alignment/run.json',
                             'cues.json', 'timing-input.json', *candidate_reports,
                             'timing-checked/cues.json', 'timing-checked/report.json', 'timing-constraints.json'):
                    source = self.output/name
                    if source.is_file():
                        target = folder/name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                attempts.append(dict(source=candidate['provenance'].get('lyric_lookup'),
                    error=str(failure), diagnostics=str(folder.relative_to(self.output))))
                if all((folder/p).is_file() for p in ('lyrics.json', 'cues.json', 'timing-constraints.json',
                        'timing-checked/cues.json', 'timing-checked/report.json')):
                    previous = folder
                continue
            if attempts or candidate.get('provenance', {}).get('repetitions') or before != source_order['after']:
                quality = json.loads((self.output/'quality.json').read_text())
                if before != source_order['after']:
                    quality['lyric_source_order'] = source_order
                if attempts:
                    quality['lyric_source_fallback'] = dict(rejected=attempts,
                        selected=candidate['provenance']['lyric_lookup'])
                quality['restored_repetitions'] = len(candidate['provenance'].get('repetitions', []))
                write_json(self.output/'quality.json', quality)
            return candidate, error

    def execute(self):
        started = time.perf_counter()
        if self.strict_audio and not local_asr.offline() and not os.getenv('OPENAI_API_KEY'):
            raise StageFailure('configuration', '请在启动服务前设置 OPENAI_API_KEY，用于 ASR 和歌词整理')
        if not shutil.which(self.separator_python):
            raise StageFailure('configuration', '找不到分离环境的 Python，请设置 KARAOKE_SEPARATOR_PYTHON')
        import soundfile as sf
        text_error = None
        artifacts = dict(instrumental='stems/instrumental.wav',
            backing_vocals='stems/backing-vocals.wav', lead_vocals='stems/lead.wav')
        if self.strict_audio:
            artifacts.update(instrumental_preview='stems/instrumental.mp3',
                backing_vocals_preview='stems/backing-vocals.mp3', lead_vocals_preview='stems/lead.mp3')
        with ThreadPoolExecutor(max_workers=4) as pool:
            text, text_duration = None, None
            def start_text(metadata, duration):
                nonlocal text, text_duration
                if text is None and not self.strict_audio:
                    text_duration = duration
                    text = pool.submit(self.prepare_text, metadata, duration)
            self.metadata_ready = start_text if not self.strict_audio else None
            self.audio_first = not self.audio_only
            metadata, source_duration = self.download()
            start_text(metadata, source_duration)
            def complete_video():
                source = json.loads((self.output/'source.json').read_text())
                result = self.promote_video(metadata, source) if source.get('audio_only') else metadata
                if not (self.output/'poster.jpg').exists():
                    create_poster(self.video, self.output/'poster.jpg', thumbnail=metadata.get('thumbnail'),
                                  seconds=min(30, source_duration*.1))
                return result
            video = pool.submit(complete_video) if not self.audio_only else None
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            audio = self.output/'source-44100.wav'
            self.run('decode_audio', ['ffmpeg', '-v', 'error', '-nostdin', '-i', self.source,
                '-map', '0:a:0', '-ar', '44100', '-ac', '2', '-c:a', 'pcm_f32le', audio])
            self.inputs(audio, 'mixture-inputs.json')
            duration = sf.info(audio).duration
            common = [self.separator_python, '-u', ROOT/'benchmark_neural.py', '--device', self.device,
                      '--batch-size', os.getenv('KARAOKE_SEPARATION_BATCH_SIZE', '1'),
                      *([] if self.strict_audio else ['--target-only'])]
            self.run('separate_music', common + ['--model', 'vocals_mel_band_roformer.ckpt', '--config',
                ROOT/'models/pretrained/vocals_mel_band_roformer.yaml', '--inputs', self.output/'mixture-inputs.json',
                '--output', self.output/'music-separation'])
            self.inputs(self.vocals, 'vocal-inputs.json')
            lead = pool.submit(self.run, 'separate_lead', common + ['--model', 'karaoke_bs_roformer_anvuew.ckpt', '--config',
                ROOT/'models/pretrained/karaoke_bs_roformer_anvuew.yaml', '--target-stem', 'Vocals',
                '--inputs', self.output/'vocal-inputs.json', '--output', self.output/'lead-separation'])
            def package_stems():
                lead.result()
                self.run('package_three_stems', [sys.executable, ROOT/'three_stems.py',
                    'build', '--source', self.source, '--vocals-estimate', self.vocals, '--lead-estimate',
                    self.output/'lead-separation/song/vocals.wav', '--output', self.output/'stems',
                    '--no-karaoke-preview', *([] if self.strict_audio else ['--no-previews'])])
                records = {key: self.artifact_record(path) for key, path in artifacts.items()}
                compressed = None
                if not self.strict_audio:
                    from .exact_stems import compact_playback_audio
                    compression_started = time.perf_counter()
                    stems = json.loads((self.output/'stems/manifest.json').read_text())
                    compressed = compact_playback_audio(self.output, dict(artifacts=records,
                        shared_preview_gain=stems['preview_shared_gain']), combine=True)
                    with self.lock:
                        self.timings['compress_audio'] = round(time.perf_counter()-compression_started, 3)
                return records, compressed
            packaging = pool.submit(package_stems)
            self.run('resample_vocals', ['ffmpeg', '-v', 'error', '-nostdin', '-i', self.vocals,
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', self.output/'vocals-16k.wav',
                *(['-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k',
                   self.output/'vocals-asr.mp3'] if self.strict_audio else [])])
            try:
                lyrics = self.prepare_text(metadata, source_duration, lead) if text is None else text.result()
                # Re-evaluate duration-dependent selection if provider metadata
                # differs materially from the actual audio. Queries remain cached.
                if text is not None and abs(text_duration-source_duration) > 1:
                    lyrics = self.prepare_text(metadata, source_duration)
            except (StageFailure, ValueError) as error:
                if self.strict_audio or (isinstance(error, StageFailure) and error.stage not in ('transcribe', 'prepare_lyrics')):
                    raise
                text_error = error
                lyrics = None
            if text_error is None:
                lyrics = self.resolve_lyric_language(lyrics)
                lyrics, text_error = self.align_candidates(lyrics, metadata, duration, packaging)
            else:
                lyrics = self.playback_fallback(metadata, duration, text_error)
                self.validate_alignment(lyrics, duration, packaging)
            render_info = dict(title=lyrics['title'], artist=lyrics['artist'], language=lyrics['language'],
                               duration_seconds=duration)
            def prepare_picture():
                assert video is not None
                source_metadata = video.result()
                self.render_picture(render_info)
                return source_metadata
            picture = pool.submit(prepare_picture) if video is not None else None
            records, compressed = packaging.result()
            if picture is not None:
                metadata = picture.result()
        stems = json.loads((self.output/'stems/manifest.json').read_text())
        artifacts['subtitles'] = 'cues.json'
        if not self.audio_only:
            artifacts['mv'] = 'mv.mp4'
        if (self.output/'poster.jpg').is_file():
            artifacts['poster'] = 'poster.jpg'
        result: dict[str, Any] = dict(title=lyrics['title'], artist=lyrics['artist'], language=lyrics['language'],
            duration_seconds=duration, mv_audio=self.request['mv_audio'], shared_preview_gain=stems['preview_shared_gain'],
            source_format=metadata.get('format_id'), artifacts={key: dict(path=value) for key, value in artifacts.items()},
            quality=json.loads((self.output/'quality.json').read_text()),
            reconstruction=stems['reconstruction']['vocal_first'], device=self.device)
        if not self.audio_only:
            self.render_video(result, self.output/'captioned-picture.mp4')
        if compressed is not None:
            result.update(audio_storage=compressed['audio_storage'], shared_preview_gain=1.)
            for name in ('instrumental', 'backing_vocals', 'lead_vocals'):
                result['artifacts'].pop(name, None)
            result['artifacts'].update(compressed['artifacts'])
            records.update(compressed['artifacts'])
        for key, artifact in result['artifacts'].items():
            artifact.update(records[key] if key in records else self.artifact_record(artifact['path']))
        result.update(processing_seconds=round(time.perf_counter()-started, 3), timings=self.timings,
                      stage_timeline=self.timeline,
                      reused_stages=self.checkpoints.reused,
                      validation_scope=('playback_with_lyrics_review' if text_error is not None else
                                        'playback_lyrics_first' if lyrics.get('provenance', {}).get('route') == 'lyrics_first' else
                                        'audio_and_lyrics' if self.audio_only else 'full_mv'))
        write_json(self.output/('audio-result.json' if self.audio_only else 'result.json'), result)
        return result

    def artifact_record(self, path):
        file = self.output/path
        return dict(path=str(path), bytes=file.stat().st_size, sha256=file_hash(file))

    def validate_localization(self, lyrics, duration, cues=None):
        """Reuse observed text to check both misplaced and omitted passages."""
        from .align_lyrics_whisper import recognized_words
        from .anchors_from_asr import observed_context, timeline_support
        observations, word_views = [], []
        raw_path = self.output/'alignment/raw.json'
        if raw_path.is_file():
            if cues is None:
                cues = json.loads((self.output/'cues.json').read_text())
            timed = [dict(c, status='candidate') for c in cues]
            localizations = getattr(self, 'recording_observations', None)
            if localizations is None:
                localizations = [o for b in json.loads(raw_path.read_text()) for o in b.get('asr_localizations', [])]
            for localization in localizations:
                if not localization.get('recognition'):
                    continue
                first, stop = localization['context']
                begin, end = localization['source_range']
                words = recognized_words(begin, localization['recognition'], end)
                word_views.append(words)
                first, stop = observed_context(lyrics['lines'], words, lyrics['language'], first, stop)
                observations.append(dict(context=[first, stop], source_range=[begin, end]))
            for observation, words in zip(observations, word_views):
                first, stop = observation['context']
                begin, end = observation['source_range']
                other_views = [[w for w in view if begin <= (w['start']+w['end'])/2 <= end]
                               for view in word_views if view is not words]
                observation.update(timeline_support(cues[first:stop], words, lyrics['language'],
                    other_word_views=other_views, combine=False))
                observation['missing_repetitions'] = missing_repetitions(
                    lyrics['lines'], timed, words, lyrics['language'], duration)
            if len(word_views) > 1:
                joint = timeline_support(cues, word_views[0], lyrics['language'], other_word_views=word_views[1:])
                for observation in observations:
                    # Dialogue can repeat a lyric verbatim. A single view's
                    # isolated occurrence cannot overturn another occurrence
                    # supported by the ordered surrounding lyric sequence.
                    # A wider view can also disprove a local occurrence count
                    # without establishing which copy is the correct one.
                    for key, lines in (('competing_view_conflicts',joint['supported_lines']),
                                       ('ambiguous_occurrence_conflicts',joint['ambiguous_lines'])):
                        conflicts = observation['conflicts']
                        disputed = [c for c in conflicts if c['line']+observation['context'][0] in lines]
                        if disputed:
                            observation[key] = disputed
                            observation['conflicts'] = [c for c in conflicts if c not in disputed]
                observations[0]['joint_view_validation'] = joint
        write_json(self.output/'localization-validation.json', observations)
        repeats = []
        if cues is not None:
            for repeat in lyrics.get('provenance', {}).get('repetitions', []):
                if 'output_range' not in repeat or 'source_lines' not in repeat:
                    continue
                first, stop = repeat['output_range']
                if lyrics['lines'][first:stop] != repeat['source_lines']:
                    raise StageFailure('subtitles', '重复唱段的来源文字不一致；诊断已保留')
                for index, anchor in zip((first,stop-1),repeat['boundary_anchors']):
                    cue = cues[index]
                    conflict = cue['end']+2 < anchor['start'] or cue['start']-2 > anchor['end']
                    repeats.append(dict(line=index+1, caption_range=[cue['start'],cue['end']],
                                        observed_range=[anchor['start'],anchor['end']], conflict=conflict))
        # This report belongs to the current candidate, including an empty one.
        # Superseded source failures remain in their archived attempts.
        write_json(self.output/'lyric-repeat-validation.json', dict(
            missing=[r for o in observations for r in o['missing_repetitions']] or None,
            restored=repeats))
        if any(r['conflict'] for r in repeats):
            raise StageFailure('subtitles', '恢复的重复唱段仍未落在已观察的音频范围；诊断已保留')
        if any(o['conflicts'] or o['missing_repetitions'] for o in observations):
            raise StageFailure('subtitles', '字幕与已定位的唱段不一致或遗漏重复唱段；音频位置证据已保留')
        return observations

    def validate_alignment(self, lyrics, duration, lead=None, *, previous=None, context_windows=None):
        from .timing_validation import timing_violations, validate_timeline
        cues = json.loads((self.output/'cues.json').read_text())
        write_json(self.output/'timing-input.json', cues)
        command = [sys.executable, ROOT/'timing_validation.py',
            '--input', self.output/'timing-input.json', '--lyrics', self.output/'lyrics.json',
            '--audio', self.vocals, '--original', self.output/'source-44100.wav',
            '--output', self.output/'timing-checked', '--device', self.device,
            *([] if self.strict_audio else ['--playback-fallback']),
            *(['--context-windows', context_windows] if context_windows else []),
            *(['--previous-alignment', previous] if previous else [])]
        # Start while packaging is still running. Only unresolved local evidence
        # waits for the delivered lead master, including its native-rate conversion.
        # Review the sound the user receives, not an intermediate model output.
        deferred = ['--defer-constraints'] if not self.strict_audio else []
        self.run('validate_timing', command+(['--defer-constraints'] if lead is not None else deferred))
        report = json.loads((self.output/'timing-checked/report.json').read_text())
        lead_view = []
        if lead is not None and (report.get('pending_review_lines') or report['unresolved']):
            lead.result()
            lead_view = ['--lead', self.output/'stems/lead.wav']
            self.run('validate_timing', command+lead_view+deferred)
        checked = json.loads((self.output/'timing-checked/cues.json').read_text())
        report = json.loads((self.output/'timing-checked/report.json').read_text())
        if not self.strict_audio and report['unresolved'] and timing_violations(checked, report['constraints']):
            # Reconsider existing measured candidates under the new evidence.
            # Keep every constraint and run the same review once on the result.
            write_json(self.output/'timing-recovery.json', dict(cues=checked, report=report))
            quality = json.loads((self.output/'quality.json').read_text())
            self.aligned_cues = checked
            self.playback_fallback(lyrics, duration, StageFailure('validate_timing',
                '已有声学证据否定部分时间；正在检查保留的对齐候选'), constraints=report['constraints'])
            quality.update(json.loads((self.output/'quality.json').read_text()))
            quality['timing_candidate_recovery'] = dict(
                lines=sorted({v['proof']['line'] for v in report['unresolved']}),
                diagnostic_sha256=file_hash(self.output/'timing-recovery.json'))
            write_json(self.output/'quality.json', quality)
            write_json(self.output/'timing-input.json', json.loads((self.output/'cues.json').read_text()))
            self.run('validate_timing', command+lead_view)
            checked = json.loads((self.output/'timing-checked/cues.json').read_text())
            report = json.loads((self.output/'timing-checked/report.json').read_text())
        try:
            validate_timeline(checked, lyrics['lines'], duration, report['constraints'], lyrics['language'])
        except ValueError as error:
            raise StageFailure('validate_timing', '字幕时间校验失败；局部修复诊断已保留') from error
        # Save the reviewed candidate before the coverage gate so a missing
        # repetition is repaired from these bounds, not the superseded ones.
        # Publication still requires this gate and the full output validation.
        write_json(self.output/'cues.json', checked)
        # Text completeness is one direction of coverage. When recovery has
        # already heard a lyric elsewhere, the final/fallback captions must
        # cover that passage too. Reuse evidence; do not launch another ASR.
        try:
            observations = self.validate_localization(lyrics, duration, checked)
        except StageFailure:
            if self.strict_audio or context_windows is not None:
                raise
            observations = json.loads((self.output/'localization-validation.json').read_text())
            ranges = {}
            def observe(index, span):
                if 0 <= index < len(checked) and 0 <= span[0] < span[1] <= duration:
                    ranges.setdefault(index, []).append(span)
            for observation in observations:
                for conflict in observation['conflicts']:
                    observe(observation['context'][0]+conflict['line']-1, conflict['observed_range'])
            for repeat in lyrics.get('provenance', {}).get('repetitions', []):
                first, stop = repeat.get('output_range', (0,0))
                if not 0 <= first < stop <= len(checked) or lyrics['lines'][first:stop] != repeat.get('source_lines'):
                    continue
                for index, anchor in zip((first,stop-1), repeat['boundary_anchors']):
                    if checked[index]['end']+2 < anchor['start'] or checked[index]['start']-2 > anchor['end']:
                        observe(index, [anchor['start'],anchor['end']])
            windows = []
            for index, spans in ranges.items():
                begin, end = max(0.,min(s[0] for s in spans)-2), min(duration,max(s[1] for s in spans)+2)
                if end-begin <= 30:
                    windows.append(dict(line=index+1, text=checked[index]['text'], source_range=[begin,end]))
            if not windows:
                raise
            # Change only where the acoustic models look. Keep the old word
            # times and all constraints until new measurements justify a repair.
            context_windows = self.output/'timing-context-recovery.json'
            write_json(context_windows, dict(cues=checked, report=report,
                observations=observations, windows=windows))
            return self.validate_alignment(lyrics, duration, lead, context_windows=context_windows)
        quality = json.loads((self.output/'quality.json').read_text())
        quality.update(lines=len(checked), units=sum(len(c['units']) for c in checked),
            localization_validation=observations,
            timing_review=dict(reviewed_lines=len({r['line'] for r in report['lines']}),
                review_attempts=len(report['lines']),
                onset_constraints=report['constraints'], unresolved=report['unresolved'],
                report_sha256=file_hash(self.output/'timing-checked/report.json'),
                repaired_ranges=sum(r['accepted'] for line in report['lines'] for r in line['repairs']),
                estimated_spans=[dict(line=i+1, **u) for i,c in enumerate(checked) for u in c['units']
                                 if u.get('timing_source')=='linear_fallback' and
                                 u.get('bounds_source') in ('cross_model_onsets','cross_model_span_end',
                                                          'cross_model_onset_prior_acoustic_end')],
                pending_review_lines=report.get('pending_review_lines', []),
                unverified_lines=sorted({r['line'] for r in report['lines'] if r['status']=='unverified'})),
            onset_accuracy=None)
        quality['fallback_units'] = sum(u.get('timing_source') == 'prior_word_fallback'
                                       for c in checked for u in c['units'])
        write_json(self.output/'quality.json', quality)

    def playback_fallback(self, metadata, duration, error, *, constraints=()):
        """Keep singing available without presenting unverified word times as valid."""
        from subtitles import playback_units, presentation_cues

        from .timing_validation import timing_violations, validate_timeline
        lyrics_path = self.output/'lyrics.json'
        lyrics: dict[str, Any] = dict(
            title=self.request.get('title') or metadata.get('track') or metadata.get('title') or 'Untitled',
            artist=self.request.get('artist') or metadata.get('artist') or '',
            language=self.request.get('language') or 'English', lines=[], provenance={})
        try:
            saved = json.loads(lyrics_path.read_text())
            if isinstance(saved, dict) and isinstance(saved.get('lines'), list):
                lyrics.update(saved)
        except (OSError, ValueError):
            pass
        candidates, recovery = [], {}
        unverified_draft = lyrics.get('provenance', {}).get('kind') == 'asr_draft'
        for name in (() if unverified_draft else ('final_alignment', 'context/song/cues.json',
                'english-onsets/cues.json', 'ordered-cues.json', 'repeated-lyrics/cues.json',
                'phonemes/cues.json', 'word-cues.json', 'alignment/cues.json', 'anchors.json')):
            path = self.output/name
            if name != 'final_alignment' and (not path.is_file() or not any(
                    path == p or path.is_relative_to(p) for p in self.alignment_outputs)):
                continue
            try:
                candidate = self.aligned_cues if name == 'final_alignment' else json.loads(path.read_text())
                if isinstance(candidate, list) and len(candidate) == len(lyrics['lines']):
                    candidates.append((name, candidate))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        cues, sources, estimated, omitted = [], [], [], []
        locations = {}
        raw_path = self.output/'alignment/raw.json'
        if not unverified_draft and raw_path.parent in self.alignment_outputs and raw_path.is_file():
            from lyric_progress import partial_alignment_cues
            try:
                blocks = json.loads(raw_path.read_text())
                recovery = alignment_recovery_summary(blocks)
                locations = phrase_locations(lyrics['lines'], blocks)
                raw = ('alignment/raw.json', partial_alignment_cues(lyrics['lines'],
                    blocks, duration, lyrics['language']))
                # Coarse captions have no word timings. Do not discard usable
                # raw words merely because another line prevented refinement.
                before_coarse = next((i for i, (name, _) in enumerate(candidates)
                                      if name in ('alignment/cues.json', 'anchors.json')), len(candidates))
                candidates.insert(before_coarse, raw)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        options = []
        for line, text in enumerate(lyrics['lines']):
            options.append([])
            if line+1 in recovery.get('unresolved_localization_lines', []):
                continue
            proofs = [dict(p, line=1) for p in constraints if p['line']==line+1]
            for name, candidate in candidates:
                cue = candidate[line]
                try:
                    if (cue['text'] != text or cue['status'] != 'candidate' or
                            not all(isinstance(cue[k], (int, float)) and math.isfinite(cue[k]) for k in ('start', 'end')) or
                            not 0 <= cue['start'] < cue['end'] <= duration+.001):
                        continue
                    if constraints and name == 'final_alignment':
                        # These units have already completed acoustic review;
                        # don't quantize them through rounded display bounds.
                        validate_timeline([cue], [text], duration, language=lyrics['language'])
                        units = cue['units']
                    else:
                        units = playback_units(cue, lyrics['language'])
                    if timing_violations([dict(cue, units=units)], proofs):
                        continue
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
                options[-1].append((name, dict(cue, units=units)))
            located = next((c for name in ('word-cues.json', 'alignment/raw.json', 'alignment/cues.json', 'anchors.json')
                            for n,c in options[-1] if n==name), None)
            if (located is not None or line in locations) and not constraints:
                # A failed refinement may remain ordered while drifting into
                # the following gap. Prefer the located phrase, retaining the
                # other candidate only if it is needed for a complete path.
                location = locations.get(line)
                if location:
                    low, high = location['start'], location['end']
                else:
                    assert located is not None
                    low, high = located['units'][0]['start'], located['units'][-1]['end']
                options[-1].sort(key=lambda pair: pair[1]['units'][-1]['end'] <= low or
                                 pair[1]['units'][0]['start'] >= high)
        def follows(previous, current):
            try:
                presentation_cues([previous, current], duration, lead=0, tail=0, bridge_gap=0)
                return previous['units'][-1]['start'] < current['units'][0]['start']
            except ValueError:
                return False
        # A locally preferred refinement must not consume the next sentence.
        # Find a complete ordered path through existing candidates first;
        # priority stays unchanged and no timestamps are manufactured.
        paths = [{} for _ in options]
        for line in range(len(options)-1, -1, -1):
            for index, (_, cue) in enumerate(options[line]):
                if line == len(options)-1:
                    paths[line][index] = None
                else:
                    following = next((j for j in paths[line+1] if follows(cue, options[line+1][j][1])), None)
                    if following is not None:
                        paths[line][index] = following
        choice = next(iter(paths[0]), None) if paths else None
        for line, candidates in enumerate(options):
            if choice is not None:
                selected = candidates[choice]
                choice = paths[line][choice]
            else:
                selected = next((pair for pair in candidates if not cues or follows(cues[-1], pair[1])), None)
            if selected is None:
                omitted.append(line+1)
                continue
            name, cue = selected
            cues.append(cue)
            sources.append(dict(line=line+1, source=name))
            estimated.extend(dict(line=line+1, **u) for u in cue['units'] if u.get('timing_source') == 'linear_fallback')
        # Reuse normal display-boundary handling. Acoustic word onsets stay
        # untouched; a small overlapping tail must not delete a whole phrase.
        display, boundaries = presentation_cues(cues, duration, lead=0, tail=0, bridge_gap=0)
        if boundaries['overlap_adjustments']:
            cues = display
        units = sum(len(c['units']) for c in cues)
        warning = dict(error=processing_error(error, self.output),
                       subtitle_mode='estimated_word' if estimated else 'partial_word' if units else 'line' if cues else 'unavailable',
                       phrase_timing_sources=sources, word_alignment_passed=False,
                       estimated_spans=estimated, omitted_lines=omitted, requires_review=bool(omitted),
                       overlap_adjustments=boundaries['overlap_adjustments'],
                       unhighlighted_lines=[i for i, c in enumerate(cues, 1) if not c['units']])
        if unverified_draft:
            warning['unverified_asr_text_omitted'] = True
        write_json(self.output/'playback-warning.json', warning)
        write_json(self.output/'cues.json', cues)
        write_json(self.output/'quality.json', dict(lines=len(cues), units=units,
            lyrics_provenance=dict(lyrics.get('provenance', {}), needs_review=True),
            playback_fallback=warning, onset_accuracy=None, separation_purity=None, **recovery))
        if not cues:
            raise StageFailure('subtitles', '未生成任何可用字幕；音轨和诊断已保留，请核对歌词来源与对齐结果') from error
        if omitted:
            raise StageFailure('subtitles', f'仍有 {len(omitted)} 行歌词未能定位；已保留音轨、可用字幕和局部修复诊断') from error
        return lyrics

    def reading_inputs(self, result, subtitles):
        if result['language'] not in ('Japanese', 'Cantonese'):
            return []
        from .lyric_readings import prepare_readings
        return [prepare_readings([c['text'] for c in json.loads(subtitles.read_text())],
                                 self.request['url'], self.checkpoints, result['language'])]

    def render_picture(self, result):
        subtitles = self.output/result.get('artifacts', {}).get('subtitles', {}).get('path', 'cues.json')
        readings = self.reading_inputs(result, subtitles)
        picture = self.output/'captioned-picture.mp4'
        with self.checkpoints.stage('render_picture', inputs=[self.video, subtitles, *readings],
                outputs=[picture, picture.with_suffix('.render.json')],
                config={**{key: result[key] for key in ('title', 'artist', 'language', 'duration_seconds')},
                        'implementation': inspect.getsource(type(self).render_picture)},
                code=[ROOT/p for p in ('render_lyric_progress.py', 'render_subtitles.py', 'subtitles.py', 'lyric_progress.py',
                                      'subtitle_style.py', 'subtitle-styles.json', 'lyric_readings.py')]) as reused:
            if not reused:
                write_json(self.output/'song-info.json', dict(title=result['title'], artist=result['artist']))
                self.run('render', [sys.executable, ROOT/'render_lyric_progress.py', '--video', self.video,
                    '--video-only', '--cues', subtitles, '--song-info', self.output/'song-info.json',
                    *(['--readings', readings[0]] if readings else []),
                    '--language', {'English': 'Latin'}.get(result['language'], result['language']) or 'auto',
                    '--duration', result['duration_seconds'], '--bottom-margin', '96', '--next-line',
                    '--countdown', '--output', picture])
        return picture

    def render_video(self, result, picture=None):
        paths = {name: self.output/a['path'] for name, a in result['artifacts'].items()}
        readings = self.reading_inputs(result, paths['subtitles'])
        presentation = paths['mv'].with_suffix('.presentation.json')
        result.pop('video_only', None)
        result['video_audio'] = 'silent_aac'
        with self.checkpoints.stage('render',
                inputs=[self.video, paths['subtitles'], *readings],
                outputs=[paths['mv'], presentation],
                config={**{key: result[key] for key in ('title', 'artist', 'language', 'duration_seconds')},
                    'implementation': [inspect.getsource(type(self).render_video),
                                       inspect.getsource(type(self).render_uncached)]},
                code=[ROOT/p for p in ('playback_video.py', 'render_lyric_progress.py', 'render_subtitles.py', 'subtitles.py', 'lyric_progress.py',
                                      'subtitle_style.py', 'subtitle-styles.json', 'lyric_readings.py')]) as reused:
            if not reused:
                self.render_uncached(result, picture)
                rendered = picture if picture is not None else self.output/'captioned-picture.mp4'
                report = json.loads(rendered.with_suffix('.render.json').read_text())
                write_json(presentation, dict(countdown_events=report['countdown']['events'],
                                              opening_countdown=report['opening_countdown']))
        result.setdefault('quality', {}).update(json.loads(presentation.read_text()))

    def render_uncached(self, result, picture=None):
        picture = picture if picture is not None else self.render_picture(result)
        paths = {name: self.output/a['path'] for name, a in result['artifacts'].items()}
        self.run('mux_video', [sys.executable, ROOT/'playback_video.py', picture, paths['mv']])
        self.run('verify_video', ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', paths['mv'], '-f', 'null', '-'])
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_type', '-of', 'json', str(paths['mv'])]))
        if ({s['codec_type'] for s in probe['streams']} != {'video', 'audio'} or
                abs(float(probe['format']['duration'])-result['duration_seconds']) > .1):
            raise StageFailure('verify_video', '成品视频缺少画面、兼容音轨或时长不符')
        picture.unlink()

    def restore_video(self):
        """Rebuild only the discarded video; never run a model or change cached audio/cues."""
        from .exact_stems import decode_source, pcm_sha
        started = time.perf_counter()
        result = json.loads((self.output/'result.json').read_text())
        if not can_restore_video(self.output, result, verify_hash=True):
            raise StageFailure('restore_video', '保留的音轨或字幕不完整，无法补回视频')
        self.download(artwork=False, format_id=result.get('source_format'))
        self.phase('verify_video', True)
        verified = time.perf_counter()
        try:
            audio, _, _ = decode_source(self.source)
            if pcm_sha(audio) != result['reconstruction']['source_pcm_sha256']:
                raise StageFailure('verify_video', '下载的视频音频与缓存版本不同，已保留原音轨和字幕；需重新完整准备该版本')
            del audio
        finally:
            self.timings['verify_video'] = round(time.perf_counter()-verified, 3)
            self.phase('verify_video', False)
        self.render_video(result)
        artifact = result['artifacts']['mv']
        path = self.output/artifact['path']
        artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))
        result['video_restore'] = dict(processing_seconds=round(time.perf_counter()-started, 3), timings=self.timings)
        write_json(self.output/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore-video', action='store_true')
    parser.add_argument('--audio-only', action='store_true', help='Test all audio/subtitle stages without downloading or rendering video')
    parser.add_argument('--playback-audio', action='store_true', help='Use normal lyric lookup and playback fallbacks, exporting audio and subtitles only')
    parser.add_argument('--prepared-lyrics', type=Path, help='Source-bound lyric lookup result prepared on another host')
    parser.add_argument('--download-only', action='store_true', help='Cache the source audio without running inference')
    args = parser.parse_args()
    if args.prepared_lyrics and not args.playback_audio:
        parser.error('--prepared-lyrics requires --playback-audio')
    pipeline = Pipeline(json.loads(args.request.read_text()), args.output,
        separator_python=os.getenv('KARAOKE_SEPARATOR_PYTHON'), device=os.getenv('KARAOKE_DEVICE', 'auto'),
        text_model=os.getenv('KARAOKE_TEXT_MODEL', 'gpt-5-mini'),
        max_duration=int(os.getenv('KARAOKE_MAX_DURATION_SECONDS', '900')),
        catalog_root=None if os.getenv('KARAOKE_DISABLE_LYRIC_CATALOG') == '1' else ROOT,
        cookies_file=os.getenv('KARAOKE_COOKIES_FILE'), audio_only=args.audio_only or args.download_only or args.playback_audio,
        playback_policy=args.playback_audio, prepared_lyrics=args.prepared_lyrics)
    try:
        if args.download_only:
            pipeline.download()
        else:
            pipeline.restore_video() if args.restore_video else pipeline.execute()
    except Exception as error:
        write_json(args.output/'error.json', processing_error(error, args.output))
        raise


if __name__ == '__main__':
    main()
