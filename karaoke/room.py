"""One persistent karaoke room, with a playlist independent of preparation jobs."""
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .cache import can_restore_video
from .job_schema import JobRequest, download_backoff, playback_version, thumbnail_url

PRESETS = {
    'instrumental': dict(instrumental=1., backing_vocals=0., lead_vocals=0.),
    'karaoke': dict(instrumental=1., backing_vocals=1., lead_vocals=0.),
    'original': dict(instrumental=1., backing_vocals=1., lead_vocals=1.),
}


def artifact_url(job_id, name, artifact):
    url = f'/v1/jobs/{job_id}/artifacts/{name}'
    return url + ('?' + urlencode({'v': artifact['sha256']}) if artifact.get('sha256') else '')


class StemLevels(BaseModel):
    model_config = ConfigDict(extra='forbid')
    instrumental: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    backing_vocals: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    lead_vocals: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class RoomCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['play', 'pause', 'seek', 'next', 'select', 'remove', 'finish', 'reorder', 'mix', 'levels', 'pitch', 'ready', 'buffering', 'retry', 'reprepare']
    entry_id: str | None = None
    position: float = Field(default=0, ge=0, allow_inf_nan=False)
    order: list[str] | None = None
    revision: int | None = None
    playback_started_at: float | None = Field(default=None, allow_inf_nan=False)
    mix: Literal['karaoke', 'instrumental', 'original'] | None = None
    levels: StemLevels | None = None
    pitch_semitones: int | None = Field(default=None, ge=-6, le=6, strict=True)


class AddSong(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str | None = Field(default=None, max_length=2048)
    job_id: str | None = None
    title: str | None = Field(default=None, max_length=200)
    source_title: str | None = Field(default=None, max_length=500)
    source_thumbnail: str | None = Field(default=None, max_length=4096)
    force: bool = False
    if_absent: bool = False


class Updates:
    """Subscribers need the latest state, not an unbounded log of old snapshots."""
    def __init__(self):
        self.subscribers = set()
        self.generation = 0

    def publish(self):
        self.generation += 1
        for queue in tuple(self.subscribers):
            if queue.empty():
                queue.put_nowait(True)

    async def events(self, snapshot):
        queue = asyncio.Queue(maxsize=1)
        self.subscribers.add(queue)
        try:
            while True:
                for chunk in snapshot():
                    yield chunk
                while True:
                    try:
                        # Keep the timeout on this stream task. wait_for would
                        # create and wake another task per viewer per update.
                        async with asyncio.timeout(15):
                            await queue.get()
                        break
                    except TimeoutError:
                        yield 'event: ping\ndata: {}\n\n'
        finally:
            self.subscribers.discard(queue)


class RoomFeed:
    """Build and encode shared state once per change, not once per viewer."""
    def __init__(self, room):
        self.room = room
        self.stream = str(uuid4())
        self.generation = None
        self.data = {}

    @staticmethod
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    def frame(self, previous, job_id=None):
        generation = self.room.updates.generation
        if generation != self.generation:
            data = self.room.snapshot()
            data.pop('server_time')
            changes, collections = {}, {}
            for key, value in data.items():
                old = self.data.get(key)
                if old == value:
                    continue
                if key in ('entries', 'history', 'library') and old is not None:
                    before, after = ({item['id']: item for item in rows} for rows in (old, value))
                    # Legacy history can contain duplicate IDs. Replace that list.
                    if len(before) == len(old) and len(after) == len(value):
                        delta = dict(upsert=[item for ident, item in after.items() if before.get(ident) != item],
                                     remove=[ident for ident in before if ident not in after])
                        if list(before) != list(after):
                            delta['order'] = list(after)
                        collections[key] = delta
                        continue
                changes[key] = value
            self.base = self.generation
            self.data = data
            self.generation = generation
            self.personal = {}
            common = dict(stream=self.stream, sequence=generation)
            self.full = ('event: room\ndata: ' + self.encode(dict(data, **common))[:-1]).encode()
            self.delta = ('event: room_patch\ndata: ' + self.encode(dict(common, base=self.base, changes=changes, collections=collections))[:-1]).encode()
        if job_id not in self.personal:
            self.personal[job_id] = self.encode(self.room.describe(dict(id=job_id, job_id=job_id))) if job_id else 'null'
        # A slow/new viewer gets a full snapshot; never apply a delta across a gap.
        delta = previous is not None and previous == self.base
        body = self.delta if delta else self.full
        # Yield the shared body separately: concatenating it would retain one
        # full-library string for every slow connection until its next write.
        return (body, f',"server_time":{time.time()},"playback_entry":{self.personal[job_id]}}}\n\n'.encode())


class Room:
    def __init__(self, store, worker, version, updates):
        self.store, self.worker, self.version, self.updates = store, worker, version, updates
        self.playing_connections = Counter()
        self.feed = RoomFeed(self)
        with store.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS room (id INTEGER PRIMARY KEY, state TEXT NOT NULL)')
            initial = dict(entries=[], history=[], current_id=None, playing=False, position=0,
                           started_at=None, mix='karaoke', revision=0)
            db.execute('INSERT OR IGNORE INTO room VALUES (1, ?)', (json.dumps(initial),))

    def read(self):
        with self.store.connect() as db:
            state = json.loads(db.execute('SELECT state FROM room WHERE id=1').fetchone()[0])
        state.setdefault('levels', PRESETS.get(state['mix'], PRESETS['karaoke']).copy())
        state.setdefault('pitch_semitones', 0)
        state.setdefault('history', [])
        return state

    def save(self, state):
        state['revision'] += 1
        with self.store.connect() as db:
            db.execute('UPDATE room SET state=? WHERE id=1', (json.dumps(state),))
        self.worker.prioritize(self.preparation_order(state))
        self.updates.publish()

    @staticmethod
    def preparation_order(state):
        return [entry['job_id'] for entry in state['entries']]

    @staticmethod
    def position(state):
        return state['position'] + (max(0, time.time()-state['started_at'])
                                    if state['playing'] and state['started_at'] else 0)

    def pause_on_shutdown(self):
        state = self.read()
        if state['playing']:
            self.mark_played(state)
            state.update(position=self.position(state), playing=False, started_at=None)
            self.save(state)

    @staticmethod
    def mark_played(state):
        # Loading and the opening count-in are not an actual performance.
        if state['playing'] and state['started_at'] is not None and time.time() > state['started_at']:
            current = next((e for e in state['entries'] if e['id'] == state['current_id']), None)
            if current:
                current.setdefault('played_at', state['started_at'])

    def remember_song(self, state, entry):
        if not entry or 'played_at' not in entry:
            return False
        song = self.describe(entry)
        state['history'].insert(0, dict(
            {key: song.get(key) for key in ('id', 'job_id', 'url', 'title', 'artist', 'language',
                                            'duration', 'poster', 'played_at')},
            finished_at=time.time()))
        return True

    def describe(self, entry, job=None, *, catalog=None):
        if job is None:
            job = self.store.read(entry['job_id'])
        if job is None:
            return dict(**entry, title='歌曲已不可用', artist='', status='failed', artifacts={})
        result, inputs = job.get('result') or {}, job['input']
        artifacts = {key: artifact_url(job['id'], key, value)
                     for key, value in result.get('artifacts', {}).items()
                     if key != 'poster' or (self.store.root/job['id']/value['path']).is_file()}
        song_info = job.get('song_info') or {}
        featured = (next((s for s in self.featured() if s['url'] == inputs['url']), {})
                    if catalog is None else catalog.get(inputs['url'], {}))
        return dict(**entry,
            title=inputs.get('title') or result.get('title') or song_info.get('title') or featured.get('title') or '新点的歌',
            artist=inputs.get('artist') or result.get('artist') or song_info.get('artist') or featured.get('artist', ''),
            url=inputs['url'], status='unprepared' if job['status'] in ('audio_ready', 'source_ready', 'cancelled') else job['status'],
            active_stages=job.get('active_stages', []),
            stage_progress=job.get('stage_progress', {}) if job['status'] == 'running' else {},
            preparation_started_at=job.get('started_at') if job['status'] == 'running' else None,
            next_retry_at=(job.get('download_retry') or {}).get('next_retry_at') if job['status'] == 'queued' else None,
            language=result.get('language') or song_info.get('language') or inputs.get('language') or featured.get('language', ''),
            error=job.get('error'), duration=result.get('duration_seconds') or song_info.get('duration_seconds', 0),
            poster=(artifacts['poster']
                    if (job['status'] == 'completed' or job.get('restore_video')) and 'poster' in artifacts
                    else thumbnail_url(song_info.get('thumbnail')) or featured.get('poster')), lyrics_need_review=result.get('quality', {}).get(
                'lyrics_provenance', {}).get('needs_review', False),
            lyrics_unavailable=result.get('quality', {}).get('playback_fallback', {}).get('subtitle_mode') == 'unavailable',
            opening_countdown=result.get('quality', {}).get('opening_countdown'),
            artifacts=artifacts, playback_version=playback_version(result))

    def snapshot(self):
        state = self.read()
        jobs = self.store.library_jobs(self.preparation_order(state))
        by_id = {job['id']: job for job in jobs}
        featured = self.featured()
        catalog = {song['url']: song for song in reversed(featured)}
        return {**state, 'entries': [self.describe(e, by_id.get(e['job_id']), catalog=catalog)
                                   for e in state['entries']],
                'playing_entry_ids': [e['id'] for e in state['entries'] if self.playing_connections[e['id']]],
                'server_time': time.time(), 'preparation_job': self.worker.current,
                'preparation_jobs': list(self.worker.processes),
                'storage': self.store.cache.snapshot(), 'library': self.library(jobs=jobs, featured=featured)}

    async def events(self, job_id=None, playing_entry_id=None):
        # A second device may finish/remove the shared queue entry while this
        # browser is still singing. Keep its media until the connection closes.
        leased = job_id if job_id and self.store.read(job_id) else None
        if leased:
            self.store.cache.acquire(leased, record_access=False)
        playing = next((e['id'] for e in self.read()['entries'] if e['id'] == playing_entry_id
                        and e['job_id'] == leased and self.describe(e)['status'] == 'completed'), None) if playing_entry_id and leased else None
        if playing:
            self.playing_connections[playing] += 1
            if self.playing_connections[playing] == 1:
                self.updates.publish()

        previous = None

        def snapshot():
            nonlocal previous
            frame = self.feed.frame(previous, leased)
            previous = self.feed.generation
            return frame

        stream = self.updates.events(snapshot)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()
            if leased:
                self.store.cache.release(leased)
            if playing:
                self.playing_connections[playing] -= 1
                if not self.playing_connections[playing]:
                    del self.playing_connections[playing]
                    self.updates.publish()

    @staticmethod
    def featured():
        path = Path(__file__).with_name('web')/'catalog.json'
        return json.loads(path.read_text()) if path.exists() else []

    def library(self, *, jobs=None, featured=None):
        if jobs is None:
            jobs = self.store.library_jobs(self.preparation_order(self.read()))
        featured = self.featured() if featured is None else featured
        catalog = {song['url']: song for song in reversed(featured)}
        songs: list[dict[str, Any]] = [dict(song, status='unprepared') for song in featured]
        by_url = {s['url']: s for s in songs}
        seen = {}
        for job in jobs:
            url = job['input']['url']
            if seen.get(url) == 2:
                continue
            result = job.get('result') or {}
            artifacts = result.get('artifacts', {})
            ready = job['status'] == 'completed' and bool(artifacts) and all(
                (self.store.root/job['id']/a['path']).is_file() for name, a in artifacts.items() if name != 'poster')
            priority = 2 if ready else int(bool(job.get('restore_video')) and
                can_restore_video(self.store.root/job['id'], result))
            if url in seen and priority <= seen[url]:
                continue
            seen[url] = priority
            song = by_url.get(url)
            info = self.describe(dict(id=job['id'], job_id=job['id']), job, catalog=catalog)
            if song is None:
                song = dict(id=job['id'])
                songs.append(song)
                by_url[url] = song
            song.update({key: info[key] for key in ('title', 'artist', 'url', 'language', 'duration',
                'poster', 'status', 'active_stages', 'stage_progress', 'preparation_started_at', 'next_retry_at', 'error',
                'artifacts', 'playback_version')})
            song['job_id'] = job['id'] if ready or job.get('restore_video') or job['status'] in ('queued', 'running') else None
            if job['status'] == 'completed' and not ready:
                song.update(status='evicted', poster=None)
        return songs

    def add(self, request):
        if bool(request.url) == bool(request.job_id):
            raise HTTPException(422, '请提供视频链接或已准备好的歌曲')
        replay = bool(request.job_id) and not request.force
        state = self.read()
        existing = None
        if request.if_absent:
            if request.job_id:
                requested = self.store.read(request.job_id)
                url = requested['input']['url'] if requested else None
            else:
                try:
                    url = JobRequest(url=request.url).url
                except ValueError as error:
                    raise HTTPException(422, '请输入有效的 YouTube 单个视频链接') from error
            for entry in state['entries']:
                job = self.store.read(entry['job_id'])
                if job and job['input']['url'] == url:
                    if replay or job['status'] != 'completed' or job.get('pipeline_version') == self.version or entry['id'] == state['current_id']:
                        return self.snapshot()
                    existing = entry
                    request = request.model_copy(update={'url': None, 'job_id': job['id']})
                    break
        if request.job_id:
            job = self.store.read(request.job_id)
            if job and (request.force or job['status'] in ('evicted', 'video_evicted', 'cancelled', 'audio_ready', 'source_ready') or job.get('restore_video') or
                        (not replay and job['status'] == 'completed' and job.get('pipeline_version') != self.version)):
                version = job['pipeline_version'] if replay and job.get('restore_video') else self.version
                try:
                    payload = JobRequest(**{**job['input'], 'force': request.force})
                except ValueError as error:
                    raise HTTPException(422, '这首歌的来源已不支持重新准备，请使用 YouTube 视频链接') from error
                job, cached = self.store.create(payload, version)
                if not cached:
                    self.worker.queue.put_nowait(job['id'])
            elif not job or job['status'] not in ('completed', 'queued', 'running'):
                raise HTTPException(404, '这首歌的任务已不可用，请重新提交链接')
            if job['status'] == 'completed' and not all((self.store.root/job['id']/a['path']).is_file()
                       for name, a in job['result']['artifacts'].items() if name != 'poster'):
                raise HTTPException(409, '这首歌的文件已不可用，请重新提交链接')
        else:
            try:
                payload = JobRequest(url=request.url, title=request.title, force=request.force)
            except ValueError as error:
                raise HTTPException(422, '请输入有效的 YouTube 单个视频链接') from error
            job, cached = self.store.create(payload, self.version)
            if not cached:
                if request.source_title and not (job.get('song_info') or {}).get('title'):
                    job = self.store.update(job['id'], song_info={'title': request.source_title})
                self.worker.queue.put_nowait(job['id'])
        if thumbnail_url(request.source_thumbnail) and not (job.get('song_info') or {}).get('thumbnail'):
            job = self.store.update(job['id'], song_info={'thumbnail': request.source_thumbnail})
        if existing is None:
            state['entries'].append(dict(id=str(uuid4()), job_id=job['id']))
        else:
            existing['job_id'] = job['id']
        self.store.touch(job['id'])
        self.save(state)
        return self.snapshot()

    def command(self, command):
        state = self.read()
        if command.revision is not None and command.revision != state['revision']:
            raise HTTPException(409, '歌单刚刚更新了，请再试一次')
        current = next((e for e in state['entries'] if e['id'] == state['current_id']), None)
        entry = next((e for e in state['entries'] if e['id'] == command.entry_id), None)
        action = command.action
        if entry is None and (action == 'finish' or action == 'remove' and command.entry_id):
            return self.snapshot()  # A retried command or another browser already removed it.
        if action == 'finish' and self.describe(entry)['status'] != 'completed':
            return self.snapshot()  # Skipping an unprepared song must not delete it.
        if action in ('pause', 'seek', 'ready', 'buffering', 'next') and command.entry_id != state['current_id']:
            raise HTTPException(409, '已经切换到另一首歌')
        if action in ('select', 'remove', 'retry', 'reprepare') and entry is None:
            raise HTTPException(404, '歌曲已从歌单移除')
        self.mark_played(state)
        if action == 'play':
            if current is None and state['entries']:
                state['current_id'] = state['entries'][0]['id']
            if state['current_id'] is None:
                raise HTTPException(409, '先点一首歌吧')
            if not state['playing']:
                state.update(playing=True, started_at=None)
        elif action == 'pause':
            state.update(position=self.position(state), playing=False, started_at=None)
        elif action == 'seek':
            if not current:
                raise HTTPException(409, '还没有正在播放的歌')
            duration = self.describe(current)['duration']
            state.update(position=min(command.position, max(0, duration-.05)),
                         started_at=time.time() if state['started_at'] and command.position > 0 else None)
        elif action == 'buffering':
            # A delayed request from before a seek/pause must not rewind this
            # song or restart its clock. Mix/volume revisions are independent.
            if (not state['playing'] or state['started_at'] is None
                    or command.playback_started_at != state['started_at']):
                raise HTTPException(409, '播放位置已经改变')
            duration = self.describe(current)['duration']
            state.update(position=min(command.position, duration), started_at=None)
        elif action == 'ready':
            if current and self.describe(current)['status'] == 'completed' and state['playing']:
                if state['started_at'] is None:
                    opening = self.describe(current)['opening_countdown'] or {}
                    delay = max(0., -opening.get('start', 0.)) if state['position'] == 0 else 0.
                    state['started_at'] = time.time()+delay
        elif action in ('next', 'select'):
            if action == 'next':
                if current and self.describe(current)['status'] == 'completed':
                    self.remember_song(state, current)
                    state['entries'].remove(current)
                entry = next((e for e in state['entries'] if e != current), None)
            else:
                if current != entry and self.remember_song(state, current):
                    state['entries'].remove(current)
                state['entries'].remove(entry)
                state['entries'].insert(0, entry)
            state.update(current_id=entry['id'] if entry else None, position=0,
                         started_at=None, playing=bool(entry))
        elif action in ('remove', 'finish'):
            assert entry is not None  # Missing entries were handled above.
            if command.playback_started_at is not None:
                entry['played_at'] = min(time.time(), command.playback_started_at)
            self.remember_song(state, entry)
            state['entries'].remove(entry)
            if entry == current:
                following = state['entries'][0] if state['entries'] else None
                state.update(current_id=following['id'] if following else None, position=0,
                             started_at=None, playing=state['playing'] and bool(following))
        elif action == 'reorder':
            ids = [e['id'] for e in state['entries']]
            if not command.order or len(command.order) != len(ids) or set(command.order) != set(ids):
                raise HTTPException(422, '排序必须包含歌单中的每一首歌，且不能重复')
            by_id = {e['id']: e for e in state['entries']}
            # Remote clients never control a device's transport. Protect live
            # performances even when a phone submits an older queue snapshot.
            playing = [i for i in ids if self.playing_connections[i]]
            state['entries'] = [by_id[i] for i in playing + [i for i in command.order if i not in playing]]
        elif action == 'mix':
            if command.mix is None:
                raise HTTPException(422, '请选择人声模式')
            state['mix'] = command.mix
            state['levels'] = PRESETS[command.mix].copy()
        elif action == 'levels':
            values = command.levels.model_dump(exclude_none=True) if command.levels else {}
            if not values:
                raise HTTPException(422, '请提供至少一个音轨音量')
            state['levels'].update(values)
            state['mix'] = next((name for name, levels in PRESETS.items() if levels == state['levels']), 'custom')
        elif action == 'pitch':
            if command.pitch_semitones is None:
                raise HTTPException(422, '请提供升降调半音数')
            state['pitch_semitones'] = command.pitch_semitones
        elif action in ('retry', 'reprepare'):
            assert entry is not None  # Missing entries were handled above.
            old = self.store.read(entry['job_id'])
            waiting = old['status'] == 'queued' and (old.get('download_retry') or {}).get('next_retry_at')
            if (action == 'reprepare' and old['status'] != 'completed') or (
                    action == 'retry' and old['status'] != 'failed' and not waiting):
                raise HTTPException(409, '歌曲状态已改变，请查看最新进度')
            try:
                payload = JobRequest(**{**old['input'], 'force': action == 'reprepare'})
            except ValueError as error:
                raise HTTPException(422, '这首歌的来源已不支持重新准备，请使用 YouTube 视频链接') from error
            until = download_backoff(old['input']['url'], self.store.root/'stage-locks')
            if until > time.time() and (waiting or (old.get('error') or {}).get('stage') == 'download'
                    or (old.get('error') or {}).get('code') == 'download_failed'):
                raise HTTPException(429, f'视频站点仍在限流，约 {max(1, round(until-time.time()))} 秒后可重试')
            if waiting:
                job = self.store.update(old['id'], error=None,
                    download_retry={**old['download_retry'], 'next_retry_at': None, 'count': 0})
                cached = False
            elif action == 'retry':
                job, cached = self.store.retry(old['id'], self.version), False
            else:
                job, cached = self.store.create(payload, self.version)
            if action == 'reprepare':
                self.store.cache.reuse_download(self.store.root/job['id'], old['input']['url'])
                # A new queue identity lets browsers already singing keep their
                # old media; their later finish event cannot remove this attempt.
                replacement = dict(id=str(uuid4()), job_id=job['id'])
                state['entries'][state['entries'].index(entry)] = replacement
                if state['current_id'] == entry['id']:
                    state.update(current_id=None, playing=False, position=0, started_at=None)
            entry['job_id'] = job['id']
            if not cached:
                self.worker.queue.put_nowait(job['id'])
        if action in ('play', 'select', 'ready', 'next'):
            selected = next((e for e in state['entries'] if e['id'] == state['current_id']), None)
            if selected:
                self.store.touch(selected['job_id'])
        self.save(state)
        if action in ('next', 'remove', 'finish', 'select', 'retry', 'reprepare'):
            self.store.cache.enforce()
        return self.snapshot()
