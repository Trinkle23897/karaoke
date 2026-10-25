"""Local persistent jobs API. Run one server worker for one GPU queue."""
import asyncio
import fcntl
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .cache import GIB, CacheFull, SongCache, can_restore_video, valid_artifact
from .job_schema import (
    JobRequest,
    download_backoff,
    playback_stems,
    playback_version,
    preparation_rank,
    retry_after_seconds,
)
from .job_schema import retry_fields as source_retry_fields
from .lyric_availability import LyricAvailability
from .room import AddSong, Room, RoomCommand, Updates, artifact_url
from .video_search import SearchCursorExpired, VideoSearch

ROOT = Path(__file__).resolve().parent


def retry_fields(error, diagnostic, number, *, failed_at):
    """Recover interactive downloads promptly, retaining shared provider limits."""
    policy = source_retry_fields(error, diagnostic, number, failed_at=failed_at)
    if (error['code'] in ('download_failed', 'download_video_failed') and
            policy['status'] == 'external' and not policy['provider_blocked']):
        delay = min(1800, 5*4**min(10, max(0, number-1)))
        policy['next_retry_at'] = failed_at+max(delay, error.get('retry_after_seconds') or 0,
                                               retry_after_seconds(diagnostic) or 0)
    return policy


CHUNK_SECONDS = 8


def playback_sources(folder, result, playback_rate):
    import soundfile as sf
    paths = []
    formats = []
    names = playback_stems(result['artifacts'])
    for name in names:
        path = (folder / result['artifacts'][name]['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError('Missing stem')
        info = sf.info(path)
        paths.append(path)
        formats.append((info.samplerate, info.channels, info.frames))
    rate, channels, frames = formats[0]
    gain = result['shared_preview_gain']
    if len(set(formats)) != 1 or channels not in (1, 2) or frames <= 0 or not 0 < rate <= 192000:
        raise ValueError('Incompatible stems')
    if not math.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError('Invalid preview gain')
    # Both ends of the padding must land on integer samples in the source AND
    # browser clocks. Otherwise per-window resampling introduces a phase jump.
    common_rate = math.gcd(rate, playback_rate)
    padding = math.ceil(.02 * common_rate) * rate // common_rate
    return paths, dict(stems=names, sample_rate=rate, channels=channels, frames=frames,
                       chunk_frames=rate * CHUNK_SECONDS, padding_frames=padding)


def playback_window_bytes(path, info, chunk, gain):
    import numpy as np
    import soundfile as sf
    first = chunk * info['chunk_frames']
    if first >= info['frames']:
        raise ValueError('Window is past the end')
    start = max(0, first - info['padding_frames'])
    end = min(info['frames'], first + info['chunk_frames'] + info['padding_frames'])
    with sf.SoundFile(path) as source:
        source.seek(start)
        samples = source.read(end - start, dtype='float64', always_2d=True)
    samples *= gain
    if len(samples) != end - start or not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 1:
        raise ValueError('Invalid playback samples')
    # FLAC has no encoder priming/padding. 24-bit previews preserve the common
    # headroom without sending float PCM's much larger payload over Wi-Fi.
    output = io.BytesIO()
    sf.write(output, samples, info['sample_rate'], format='FLAC', subtype='PCM_24')
    return output.getvalue()


def now():
    return datetime.now(UTC).isoformat()


def job_cache_key(payload, version):
    inputs = {key: value for key, value in payload.items() if key != 'force'}
    return hashlib.sha256(json.dumps([version, inputs], sort_keys=True).encode()).hexdigest()


def validated_result(folder):
    result = json.loads((folder/'result.json').read_text())
    artifacts = result['artifacts']
    if not {'mv', 'subtitles', *playback_stems(artifacts)} <= artifacts.keys():
        raise ValueError('Pipeline omitted a required output')
    for name, artifact in list(artifacts.items()):
        if not valid_artifact(folder, artifact, verify_hash=True):
            if name == 'poster':
                logging.getLogger(__name__).warning('Job %s: optional poster unavailable', folder.name)
                del artifacts[name]
            else:
                raise ValueError(f'Invalid result artifact: {name}')
    cues = json.loads((folder/artifacts['subtitles']['path']).read_text())
    if not isinstance(cues, list) or not cues or not all(
            cue['text'].strip() and 0 <= cue['start'] < cue['end'] <= result['duration_seconds']+.001 for cue in cues):
        raise ValueError('No usable subtitle timeline')
    return result


class AudioWindowCache:
    """Share encoded bytes across listeners, with bounded memory and one producer."""
    def __init__(self, max_bytes=512*1024**2, max_entries=256):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.values, self.pending = OrderedDict(), {}
        self.size_bytes = 0
        self.lock = threading.Lock()

    def get(self, key, produce):
        with self.lock:
            if key in self.values:
                self.values.move_to_end(key)
                return self.values[key]
            waiting = self.pending.get(key)
            if waiting is None:
                waiting = self.pending[key] = Future()
                owner = True
            else:
                owner = False
        if not owner:
            return waiting.result()
        try:
            value = produce()
            with self.lock:
                if len(value) <= self.max_bytes:
                    self.values[key] = value
                    self.size_bytes += len(value)
                    while self.size_bytes > self.max_bytes or len(self.values) > self.max_entries:
                        self.size_bytes -= len(self.values.popitem(last=False)[1])
        except BaseException as error:
            waiting.set_exception(error)
            raise
        else:
            waiting.set_result(value)
            return value
        finally:
            with self.lock:
                self.pending.pop(key)


def known_song_info(job):
    return {**(job.get('song_info') or {}), **{
        key: value for key, value in (job.get('result') or {}).items()
        if key in ('title', 'artist', 'language', 'duration_seconds') and value}}


class Store:
    def __init__(self, root, cache_limit_bytes=None):
        self.on_change = lambda: None
        self.access_times = OrderedDict()
        self.access_lock = threading.Lock()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'jobs.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, cache_key TEXT, status TEXT, created_at TEXT, doc TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_cache ON jobs(cache_key, status)')
            db.execute("""CREATE INDEX IF NOT EXISTS jobs_source_latest
                ON jobs(json_extract(doc, '$.input.url'), created_at DESC)""")
            db.execute("""CREATE INDEX IF NOT EXISTS jobs_library_available ON jobs(id)
                WHERE status IN ('queued','running','completed') OR json_extract(doc, '$.restore_video')=1
                OR (status='audio_ready' AND json_extract(doc, '$.audio_result') IS NOT NULL)""")
            db.execute('DROP TABLE IF EXISTS library_sources')
            db.execute('DROP INDEX IF EXISTS jobs_library_media')
            db.execute('DROP INDEX IF EXISTS jobs_library_candidates')
        self.cache = SongCache(self, cache_limit_bytes)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def read(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT doc FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id, **fields):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT doc FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            doc = json.loads(row[0])
            if 'song_info' in fields:
                # Stages learn metadata incrementally; missing values do not erase it.
                fields['song_info'] = {**(doc.get('song_info') or {}), **{
                    key: value for key, value in (fields['song_info'] or {}).items()
                    if value is not None and value != ''}}
            if fields.get('status') in ('completed', 'failed', 'cancelled') or (
                    'song_info' in fields and fields['song_info'] != doc.get('song_info')):
                # Tiny, content-checked metadata survives media LRU eviction.
                metadata = {}
                for name in ('song-metadata.json', 'checkpoints/extract_song_metadata.json'):
                    path = self.root/job_id/name
                    if path.is_file() and not path.is_symlink() and path.stat().st_size <= 262144:
                        try:
                            metadata[name] = json.loads(path.read_text())
                        except (OSError, ValueError):
                            pass
                if len(metadata) == 2:
                    fields['metadata_cache'] = metadata
            doc.update(fields, updated_at=now())
            if 'pipeline_version' in fields:
                cache_key = job_cache_key(doc['input'], doc['pipeline_version'])
                db.execute('UPDATE jobs SET cache_key=? WHERE id=?', (cache_key, job_id))
            db.execute('UPDATE jobs SET status=?, doc=? WHERE id=?',
                       (doc['status'], json.dumps(doc, ensure_ascii=False), job_id))
        self.on_change()
        return doc

    def touch(self, job_id):
        # Listing or receiving SSE snapshots is not use of a cached song.
        # A popular song can serve hundreds of chunks a second. Persist real
        # access at one-second resolution instead of contending on each chunk.
        with self.access_lock:
            stamp = time.monotonic()
            if stamp - self.access_times.get(job_id, -math.inf) < 1:
                return
            with self.connect() as db:
                db.execute("UPDATE jobs SET doc=json_set(doc, '$.last_accessed_at', ?) WHERE id=?", (now(), job_id))
            self.access_times[job_id] = stamp
            self.access_times.move_to_end(job_id)
            if len(self.access_times) > 1024:
                self.access_times.popitem(last=False)

    def requeue(self, db, doc, key, version):
        folder = self.cache.folder(doc['id'])
        folder.mkdir(exist_ok=True)
        (folder/'request.json').write_text(json.dumps(doc['input'], ensure_ascii=False)+'\n')
        if not doc.get('restore_video'):
            doc['pipeline_version'] = version
        if not doc.get('restore_video') or doc['status'] == 'failed':
            doc['attempt_count'] = doc.get('attempt_count', 1)+1
        if doc.get('download_retry'):
            doc['download_retry'].update(count=0, next_retry_at=None)
        stamp = now()
        doc.update(status='queued', active_stages=[], stage_progress={}, timings={}, error=None,
                   updated_at=stamp, last_accessed_at=stamp)
        db.execute('UPDATE jobs SET cache_key=?, status=?, doc=? WHERE id=?',
                   (key, 'queued', json.dumps(doc, ensure_ascii=False), doc['id']))
        return doc

    def retry(self, job_id, version):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT doc FROM jobs WHERE id=? AND status='failed'", (job_id,)).fetchone()
            if not row:
                raise HTTPException(409, '原任务状态已改变')
            doc = json.loads(row[0])
            if doc.get('restore_video') and not can_restore_video(self.cache.folder(job_id), doc.get('result')):
                raise HTTPException(409, '保留的音轨或字幕不完整，无法补回视频')
            return self.requeue(db, doc, job_cache_key(doc['input'], version), version)

    def create(self, payload, version):
        key_payload = payload.model_dump(exclude={'force'})
        key = job_cache_key(key_payload, version)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not payload.force:
                rows = db.execute("""SELECT doc FROM jobs WHERE json_extract(doc, '$.input.url')=? AND (
                    (cache_key=? AND status='running') OR status IN ('queued','failed','cancelled','audio_ready','source_ready','completed')
                    OR json_extract(doc, '$.restore_video')=1)
                    ORDER BY (cache_key=?) DESC, (status='completed') DESC,
                             COALESCE(json_extract(doc, '$.restore_video'), 0) DESC, created_at DESC""", (payload.url, key, key))
                for row in rows:
                    doc = json.loads(row[0])
                    if {k: v for k, v in doc['input'].items() if k != 'force'} != key_payload:
                        continue
                    if doc['status'] == 'audio_ready' and can_restore_video(self.root/doc['id'], doc.get('audio_result')):
                        result = {**doc['audio_result'], 'artifacts':{**doc['audio_result']['artifacts'],
                            'mv':dict(path='mv.mp4', bytes=0, sha256='')}}
                        doc.update(result=result, restore_video=True)
                        (self.root/doc['id']/'result.json').write_text(json.dumps(result, ensure_ascii=False)+'\n')
                    if doc['status'] == 'queued' and not doc.get('restore_video'):
                        # Pending work has not executed the old pipeline yet.
                        doc.update(pipeline_version=version, updated_at=now())
                        db.execute('UPDATE jobs SET cache_key=? WHERE id=?', (key, doc['id']))
                    if doc['status'] in ('failed', 'cancelled', 'audio_ready', 'source_ready') and not doc.get('restore_video'):
                        folder = self.root/doc['id']
                        if not folder.is_dir():
                            continue
                        source_ready = doc['status'] == 'source_ready' and all((folder/name).is_file()
                            for name in ('source.json', 'download/source.mkv', 'download/source.info.json'))
                        if (not source_ready and doc['status'] != 'cancelled'
                                and not any((folder/'checkpoints').glob('*.json'))):
                            continue
                        return self.requeue(db, doc, key, version), False
                    if doc.get('restore_video'):
                        if doc['status'] not in ('queued', 'running'):
                            folder = self.root/doc['id']
                            if not can_restore_video(folder, doc.get('result')):
                                continue
                            return self.requeue(db, doc, key, version), False
                    if doc['status'] != 'completed' or all(
                            (self.root/doc['id']/item['path']).is_file()
                            for name, item in doc['result']['artifacts'].items() if name != 'poster'):
                        doc['last_accessed_at'] = now()
                        db.execute('UPDATE jobs SET doc=? WHERE id=?', (json.dumps(doc, ensure_ascii=False), doc['id']))
                        return doc, True
            job_id, stamp = str(uuid4()), now()
            doc: dict[str, Any] = dict(id=job_id, status='queued', active_stages=[], created_at=stamp, updated_at=stamp,
                       last_accessed_at=stamp, input=payload.model_dump(), result=None, error=None, pipeline_version=version)
            previous = db.execute("SELECT doc FROM jobs WHERE json_extract(doc, '$.input.url')=? ORDER BY created_at DESC",
                                  (payload.url,))
            for row in previous:
                old = json.loads(row[0])
                info = known_song_info(old)
                doc['song_info'] = {**{k: v for k, v in info.items() if v is not None and v != ''},
                                    **doc.get('song_info', {})}
                if old.get('metadata_cache') and not doc.get('metadata_cache'):
                    doc['metadata_cache'] = old['metadata_cache']
            folder = self.root/job_id
            folder.mkdir()
            (folder/'request.json').write_text(json.dumps(payload.model_dump(), ensure_ascii=False)+'\n')
            if doc.get('song_info'):
                (folder/'progress.json').write_text(json.dumps({'song_info': doc['song_info']}, ensure_ascii=False)+'\n')
            if not payload.force:
                self.cache.reuse_download(folder, payload.url)
            for name, value in doc.get('metadata_cache', {}).items():
                if name in ('song-metadata.json', 'checkpoints/extract_song_metadata.json'):
                    path = folder/name
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps(value, ensure_ascii=False)+'\n')
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)',
                       (job_id, key, 'queued', stamp, json.dumps(doc, ensure_ascii=False)))
        return doc, False

    def library_jobs(self, queue_job_ids=()):
        # Keep live work/media and queue failures. Only those sources need their
        # latest attempt, which preserves card identity and metadata on retries.
        with self.connect() as db:
            rows = db.execute("""WITH candidates AS (
                SELECT id, json_extract(doc, '$.input.url') AS url FROM jobs
                WHERE status IN ('queued','running','completed') OR json_extract(doc, '$.restore_video')=1
                OR (status='audio_ready' AND json_extract(doc, '$.audio_result') IS NOT NULL)
                UNION SELECT id, json_extract(doc, '$.input.url') FROM jobs
                WHERE id IN (SELECT value FROM json_each(?)))
                SELECT doc FROM jobs WHERE id IN (
                    SELECT id FROM candidates
                    UNION SELECT (SELECT id FROM jobs
                        WHERE json_extract(doc, '$.input.url')=candidates.url
                        ORDER BY created_at DESC, rowid DESC LIMIT 1) FROM candidates)
                ORDER BY created_at DESC""", (json.dumps(queue_job_ids),))
            return [json.loads(row[0]) for row in rows]

    def list(self, limit=20, status=None):
        with self.connect() as db:
            rows = (db.execute('SELECT doc FROM jobs WHERE status=? ORDER BY created_at LIMIT ?', (status, limit))
                    if status else db.execute('SELECT doc FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,)))
            return [json.loads(row[0]) for row in rows]


class Worker:
    def __init__(self, store, command=None, version=None, concurrency=1):
        self.store, self.command = store, command
        self.version = version
        if concurrency is not None and concurrency < 1:
            raise ValueError('Preparation concurrency must be positive or None')
        self.concurrency = concurrency
        self.queue, self.processes = asyncio.Queue(), {}
        self.tasks, self.cancelled = {}, set()
        self.dispatch_errors = {}
        self.budget_checked_at, self.budget_ok = -float('inf'), True

    def check_budget(self):
        # Every job watches the same disk budget. Share its periodic scan rather
        # than walking the whole cache once per running job on the API thread.
        stamp = time.monotonic()
        if not self.budget_ok or stamp-self.budget_checked_at >= 1:
            self.budget_ok = self.store.cache.enforce(self.reserve)
            self.budget_checked_at = stamp
        return self.budget_ok

    @property
    def current(self):
        return next(iter(self.processes), None)

    @property
    def process(self):
        return next(iter(self.processes.values()), None)

    @property
    def reserve(self):
        return self.store.cache.headroom * max(1, len(self.processes))

    def prioritize(self, job_ids):
        folder = self.store.root/'stage-locks'
        folder.mkdir(exist_ok=True)
        temporary = folder/(str(uuid4())+'.tmp')
        temporary.write_text(json.dumps(job_ids))
        temporary.replace(folder/'priority.json')
        pending = []
        while not self.queue.empty():
            pending.append(self.queue.get_nowait())
            self.queue.task_done()
        rank = preparation_rank(job_ids)
        for job_id in sorted(dict.fromkeys(pending), key=lambda key: rank.get(key, len(rank))):
            job = self.store.read(job_id) or {}
            if job.get('status') == 'queued':
                self.queue.put_nowait(job_id)
            else:
                self.dispatch_errors.pop(job_id, None)

    def in_room(self, job_id):
        with self.store.connect() as db:
            row = db.execute('SELECT state FROM room WHERE id=1').fetchone()
        return any(e['job_id'] == job_id for e in json.loads(row[0])['entries'])

    def mark_cancelled(self, job_id):
        job = self.store.read(job_id)
        retry = {'download_retry': {**job['download_retry'], 'next_retry_at': None}} if job.get('download_retry') else {}
        self.store.update(job_id, status='cancelled', active_stages=[], stage_progress={},
                          error=None, cancelled_at=now(), **retry)

    async def cancel_unneeded(self, job_id):
        """Stop removed work, retaining verified checkpoints and completed media."""
        job = self.store.read(job_id)
        if self.in_room(job_id) or not job or job['status'] not in ('queued', 'running'):
            return
        if job['status'] == 'running' and job_id not in self.tasks:
            return  # An independent probe owns this process, not this worker.
        self.cancelled.add(job_id)
        try:
            task = self.tasks.get(job_id)
            if job_id not in self.processes:
                # The task may have been dispatched but not entered consume yet.
                self.mark_cancelled(job_id)
                self.store.cache.compact(self.store.read(job_id))
            if task:
                if not task.cancelling():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # A second device can re-add the song while its old process exits.
            # Resume only after that process and cache cleanup have finished.
            if self.in_room(job_id) and self.store.read(job_id)['status'] == 'cancelled':
                retry = {'download_retry': {**job['download_retry'], 'next_retry_at': None, 'count': 0}} if job.get('download_retry') else {}
                self.store.update(job_id, status='queued', active_stages=[], stage_progress={}, timings={},
                                  error=None, attempt_count=job.get('attempt_count', 1)+1, **retry)
                self.queue.put_nowait(job_id)
            with self.store.connect() as db:
                state = json.loads(db.execute('SELECT state FROM room WHERE id=1').fetchone()[0])
            self.prioritize(Room.preparation_order(state))
            self.store.cache.enforce()
        finally:
            self.cancelled.discard(job_id)

    def download_failed(self, job, error, room_managed):
        """Persist a bounded retry without occupying a worker during backoff."""
        if self.store.read(job['id'])['status'] == 'cancelled':
            return
        folder = self.store.root/job['id']
        log = folder/(error['code'].removesuffix('_failed')+'.log')
        diagnostic = ''
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-16384))
                diagnostic = stream.read().decode(errors='replace')
        previous = job.get('download_retry') or {}
        count = previous.get('count', 0)+1
        failed_at = time.time()
        policy = retry_fields(error, diagnostic, count, failed_at=failed_at)
        provider_until = max(previous.get('provider_until', 0),
            policy['next_retry_at'] if policy['provider_blocked'] else 0)
        download_backoff(job['input']['url'], self.store.root/'stage-locks', until=provider_until)
        retry_at = policy['next_retry_at'] if policy['status'] == 'external' and count <= 3 else None
        if room_managed and not self.in_room(job['id']):
            retry_at = None
        history = [*previous.get('history', []), dict(at=failed_at, error=error, **policy)]
        retry = dict(count=count, next_retry_at=retry_at, provider_until=provider_until,
                     room_managed=room_managed, history=history)
        with (folder/'download-retries.log').open('a') as stream:
            stream.write(json.dumps(dict(at=failed_at, error=error, diagnostic=diagnostic), ensure_ascii=False)+'\n')
        self.store.update(job['id'], status='queued' if retry_at else 'failed',
                          active_stages=[], error=error, download_retry=retry)
        if retry_at:
            self.queue.put_nowait(job['id'])

    async def start(self):
        for job in self.store.list(100000, 'running'):
            self.store.update(job['id'], status='failed', active_stages=[],
                error=dict(code='interrupted', message='上次服务在处理时退出；可重新提交链接'))
        for job in self.store.list(100000):
            if job['status'] in ('completed', 'video_evicted', 'failed', 'cancelled', 'audio_ready'):
                self.store.cache.compact(job)
        self.store.cache.enforce()
        for job in self.store.list(100000, 'queued'):
            self.queue.put_nowait(job['id'])
        self.dispatcher = asyncio.create_task(self.dispatch())

    async def stop(self):
        self.dispatcher.cancel()
        await asyncio.gather(self.dispatcher, return_exceptions=True)
        for task in self.tasks.values():
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def dispatch(self):
        while True:
            while ((self.store.root/'preparation.pause').exists() or self.queue.empty() or
                   (self.concurrency is not None and len(self.tasks) >= self.concurrency)):
                await asyncio.sleep(.2)
            for _ in range(self.queue.qsize()):
                job_id = self.queue.get_nowait()
                try:
                    job = self.store.read(job_id)
                    deadline = (max((job.get('download_retry') or {}).get('next_retry_at') or 0,
                        download_backoff(job['input']['url'], self.store.root/'stage-locks'))
                        if (job or {}).get('status') == 'queued' else 0)
                except Exception as error:
                    reason = type(error).__name__+': '+str(error)
                    if self.dispatch_errors.get(job_id) != reason:
                        logging.getLogger(__name__).exception('Cannot dispatch job %s; retaining it in the queue', job_id)
                    self.dispatch_errors[job_id] = reason
                    self.queue.put_nowait(job_id)
                    self.queue.task_done()
                    continue
                self.dispatch_errors.pop(job_id, None)
                if (job or {}).get('status') != 'queued' or job_id in self.tasks:
                    self.queue.task_done()
                    continue
                if deadline > time.time():
                    self.queue.put_nowait(job_id)
                    self.queue.task_done()
                    continue
                task = asyncio.create_task(self.consume(job_id))
                self.tasks[job_id] = task
                def finished(task, key=job_id):
                    if self.tasks.get(key) is task:
                        self.tasks.pop(key)
                    self.queue.task_done()
                task.add_done_callback(finished)
                break
            else:
                await asyncio.sleep(.2)

    async def terminate(self, process):
        if process:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                pass
            finally:
                # The parent can exit while a descendant ignores SIGTERM or
                # closes stdout. Never let that orphan keep writing the cache.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()

    async def consume(self, job_id):
        job = self.store.read(job_id)
        if (job or {}).get('status') != 'queued':
            return
        process = self.processes[job_id] = None
        folder = self.store.root/job_id
        room_managed = self.in_room(job_id)
        # Restoring a cached recording only rebuilds its presentation. Keep the
        # actual audio/subtitle version; a new preparation updates it separately.
        version_key = 'presentation_pipeline_version' if job.get('restore_video') else 'pipeline_version'
        version = {version_key: self.version} if self.version else {}
        retry = {'download_retry': {**job['download_retry'], 'next_retry_at': None}} if job.get('download_retry') else {}
        job = self.store.update(job_id, status='running', error=None, started_at=now(), **version, **retry)
        try:
            (folder/'error.json').unlink(missing_ok=True)
            if not self.store.cache.enforce(self.reserve):
                raise CacheFull('歌曲缓存或硬盘剩余空间不足；请移除待唱歌曲、提高上限或释放磁盘后重试')
            command = self.command(folder) if self.command else [sys.executable, '-u', '-m',
                'karaoke.pipeline', '--request', str(folder/'request.json'), '--output', str(folder)]
            if job.get('restore_video'):
                (folder/'result.json').write_text(json.dumps(job['result'], ensure_ascii=False)+'\n')
                if not self.command:
                    command.append('--restore-video')
            with (folder/'worker.log').open('ab') as log:
                spawn = asyncio.create_task(asyncio.create_subprocess_exec(*command, cwd=ROOT.parent,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, 'KARAOKE_EVENTS_STDOUT': '1',
                         'KARAOKE_STAGE_LOCK_DIR': str(self.store.root/'stage-locks')}, start_new_session=True))
                try:
                    process = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    process = await spawn
                    raise
                self.processes[job_id] = process
                await self.supervise(log, job_id, process)
            result_path = folder/'result.json'
            if process.returncode == 0 and result_path.exists():
                result = await asyncio.to_thread(validated_result, folder)
                self.store.update(job_id, status='completed', active_stages=[], result=result, restore_video=False, error=None,
                                  completed_at=now(), last_accessed_at=now())
            else:
                error_path = folder/'error.json'
                error = json.loads(error_path.read_text()) if error_path.exists() else dict(
                    code='worker_failed', message='处理进程退出；请检查该任务的 worker.log')
                if error['code'] in ('download_failed', 'download_video_failed'):
                    self.download_failed(job, error, room_managed)
                else:
                    self.store.update(job_id, status='failed', active_stages=[], error=error)
        except CacheFull as error:
            await self.terminate(process)
            self.store.update(job_id, status='failed', active_stages=[],
                              error=dict(code='cache_full', message=str(error)))
        except asyncio.CancelledError:
            await self.terminate(process)
            if job_id in self.cancelled:
                self.mark_cancelled(job_id)
            else:
                self.store.update(job_id, status='failed', active_stages=[],
                    error=dict(code='interrupted', message='服务关闭，任务已中止；可重新提交链接'))
            raise
        except Exception:
            logging.getLogger(__name__).exception('Job %s failed output/worker validation', job_id)
            await self.terminate(process)
            self.store.update(job_id, status='failed', active_stages=[],
                error=dict(code='worker_error', message='任务执行或输出验证失败；请检查服务器日志'))
        finally:
            self.processes.pop(job_id)
            self.store.cache.compact(self.store.read(job_id))
            self.store.cache.enforce()

    async def supervise(self, log, job_id, process):
        async def output():
            async for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(b'KARAOKE_PROGRESS '):
                    try:
                        progress = json.loads(line[len(b'KARAOKE_PROGRESS '):])
                        shapes = dict(active_stages=list, stage_progress=dict, timings=dict, song_info=dict)
                        if not isinstance(progress, dict) or any(
                                key in progress and not isinstance(progress[key], shape) for key, shape in shapes.items()):
                            raise ValueError('Invalid progress fields')
                    except (ValueError, TypeError, UnicodeError):
                        logging.getLogger(__name__).warning('Job %s: ignored malformed progress event', job_id)
                        continue
                    self.store.update(job_id, **{key: progress[key]
                        for key in ('active_stages', 'stage_progress', 'timings', 'song_info') if key in progress})
            await process.wait()

        async def budget():
            while True:
                await asyncio.sleep(1)
                if not self.check_budget():
                    raise CacheFull('预处理达到缓存或磁盘空间限制，已停止并清理临时文件；请释放空间后重试')

        tasks = [asyncio.create_task(output()), asyncio.create_task(budget())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
            if not self.store.cache.enforce():
                raise CacheFull('生成文件超过缓存容量；请提高上限或移除待唱歌曲后重试')
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class StorageLimit(BaseModel):
    limit_gib: float | None = Field(ge=1, le=4096, allow_inf_nan=False)


class LeasedFileResponse(FileResponse):
    # Media responses are large; avoid a thread-pool handoff for every 64 KiB.
    # Keep each read bounded so slow clients still apply backpressure.
    chunk_size = 1024 * 1024

    def __init__(self, *args, cache, job_id, record_access=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache, self.job_id = cache, job_id
        cache.acquire(job_id, record_access=record_access)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cache.release(self.job_id)


def create_app(data_dir=None, *, worker_command=None, version=None, cache_limit_bytes=None,
               worker_concurrency='env'):
    store = Store(data_dir or os.getenv('KARAOKE_DATA_DIR', str(ROOT/'service-data')), cache_limit_bytes)
    if version is None:
        from .local_asr import configuration
        from .pipeline import pipeline_fingerprint
        settings = {key: os.getenv(key) for key in ('KARAOKE_TEXT_MODEL', 'KARAOKE_ASR_FALLBACK_MODEL', 'KARAOKE_DEVICE',
            'KARAOKE_DISABLE_LYRIC_CATALOG', 'KARAOKE_FONT_FILE', 'KARAOKE_SEPARATOR_PYTHON',
            'KARAOKE_SEPARATION_BATCH_SIZE', 'KARAOKE_COMPUTE_CONCURRENCY')}
        settings['asr'] = configuration()
        version = hashlib.sha256((pipeline_fingerprint()+json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    if worker_concurrency == 'env':
        configured = os.getenv('KARAOKE_PREPARATION_CONCURRENCY', '1').strip().lower()
        worker_concurrency = None if configured == 'none' else int(configured)
    worker = Worker(store, worker_command, version, worker_concurrency)
    updates = Updates()
    store.on_change = updates.publish
    room = Room(store, worker, version, updates)
    video_search = VideoSearch()
    lyric_availability = LyricAvailability()
    playback_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='playback')
    playback_windows = AudioWindowCache()
    playback_tasks = set()

    @asynccontextmanager
    async def lifespan(app):
        with (store.root/'worker.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('此任务目录已有服务在运行。每个 GPU 队列只运行一个 API worker。') from error
            await worker.start()
            worker.prioritize(room.preparation_order(room.read()))
            try:
                yield
            finally:
                room.pause_on_shutdown()
                await video_search.close()
                await lyric_availability.close()
                await worker.stop()
                playback_pool.shutdown(wait=False, cancel_futures=True)
                await asyncio.gather(*playback_tasks, return_exceptions=True)

    app = FastAPI(title='Karaoke preparation API', version='1.0', lifespan=lifespan,
        description='提交 YouTube 视频链接，后台生成字幕视频、底轨（背景音乐＋伴唱）和主唱音轨。视频含静音兼容轨，播放时使用独立音轨。')
    origins = os.getenv('KARAOKE_CORS_ORIGINS', 'http://127.0.0.1:8745,http://localhost:8745')
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins.split(',') if o.strip()],
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    app.state.store, app.state.worker, app.state.room = store, worker, room

    def public_job(doc, request):
        result = json.loads(json.dumps(doc))
        result['status_url'] = str(request.url_for('get_job', job_id=doc['id']))
        if result['result']:
            for name, artifact in result['result']['artifacts'].items():
                artifact['url'] = str(request.base_url).rstrip('/') + artifact_url(doc['id'], name, artifact)
                artifact.pop('path', None)
        return result

    @app.get('/health')
    async def health():
        failed = bool(worker.dispatch_errors) or worker.dispatcher.done()
        return dict(status='degraded' if failed else 'ok', queued=worker.queue.qsize(), running_job=worker.current,
                    dispatch_blocked_jobs=list(worker.dispatch_errors),
                    running_jobs=list(worker.processes), preparation_concurrency=worker.concurrency,
                    preparation_paused=(store.root/'preparation.pause').exists(),
                    pipeline_version=version, device=os.getenv('KARAOKE_DEVICE', 'auto'))

    @app.get('/v1/library')
    async def library():
        return dict(songs=room.library())

    async def search_metadata(provider, q, cursor):
        query = q.strip()
        if not query:
            raise HTTPException(422, '请输入歌名或歌手')
        try:
            return await video_search.search(provider, query, cursor)
        except SearchCursorExpired as error:
            raise HTTPException(410, '搜索已过期，请重新搜索') from error
        except OverflowError as error:
            raise HTTPException(429, '搜索较多，请稍后重试') from error
        except Exception as error:
            logging.getLogger(__name__).warning('%s search failed: %s', provider, type(error).__name__)
            raise HTTPException(502, 'YouTube暂时搜不到，请重试') from error

    @app.get('/v1/search')
    async def search_videos(provider: Literal['youtube'], q: str = Query(min_length=1, max_length=160),
                            cursor: str | None = Query(default=None, min_length=1, max_length=128)):
        return dict(query=q.strip(), provider=provider, **await search_metadata(provider, q, cursor))

    @app.get('/v1/music/search')
    async def search_music(kind: Literal['songs', 'artists'] = 'songs', q: str = Query(min_length=1, max_length=160),
                           cursor: str | None = Query(default=None, min_length=1, max_length=128)):
        return dict(query=q.strip(), provider='youtube_music', **await search_metadata(kind, q, cursor))

    @app.get('/v1/music/artists/{browse_id}')
    async def artist_songs(browse_id: str, cursor: str | None = Query(default=None, min_length=1, max_length=128)):
        if not re.fullmatch(r'UC[\w-]{22}', browse_id):
            raise HTTPException(422, '无效的歌手链接')
        return await search_metadata('artist', browse_id, cursor)

    @app.get('/v1/music/albums/{browse_id}')
    async def album_songs(browse_id: str):
        if not re.fullmatch(r'MPRE[\w-]+', browse_id):
            raise HTTPException(422, '无效的专辑链接')
        return await search_metadata('album', browse_id, None)

    @app.get('/v1/lyrics/availability')
    async def check_lyrics(ids: str = Query(min_length=11, max_length=95)):
        videos = list(dict.fromkeys(ids.split(',')))
        if len(videos) > 8 or any(not re.fullmatch(r'[A-Za-z0-9_-]{11}', video) for video in videos):
            raise HTTPException(422, '一次最多查询 8 个有效视频编号')
        results = await asyncio.gather(*(lyric_availability.check(video) for video in videos))
        return dict(results={video: dict(result, cache_seconds=max(0, result['expires_at']-time.time()))
                             for video, result in zip(videos, results)})

    @app.get('/v1/storage')
    async def storage():
        store.cache.enforce()
        return store.cache.snapshot()

    @app.post('/v1/storage')
    async def storage_limit(payload: StorageLimit):
        try:
            store.cache.set_limit(None if payload.limit_gib is None else round(payload.limit_gib*GIB))
        except CacheFull as error:
            raise HTTPException(409, str(error)) from error
        return room.snapshot()

    @app.get('/v1/room')
    async def room_state():
        return room.snapshot()

    @app.post('/v1/room/entries', status_code=201)
    async def add_song(payload: AddSong):
        before = {e['id'] for e in room.read()['entries']}
        state = room.add(payload)
        added = [e for e in room.read()['entries'] if e['id'] not in before]
        if added:
            logging.getLogger('uvicorn.error').info('Queue add revision=%s entries=%s', state['revision'], json.dumps(added))
        return state

    @app.post('/v1/room/commands')
    async def room_command(payload: RoomCommand):
        previous = room.read()
        before = {e['job_id'] for e in previous['entries']}
        state = room.command(payload)
        remaining = {e['id'] for e in state['entries']}
        removed = [e for e in previous['entries'] if e['id'] not in remaining]
        if payload.action in ('remove', 'finish', 'next', 'select', 'retry', 'reprepare', 'reorder'):
            logging.getLogger('uvicorn.error').info('Queue command action=%s entry=%s revision=%s removed=%s',
                payload.action, payload.entry_id, state['revision'], json.dumps(removed))
        for job_id in before - {e['job_id'] for e in state['entries']}:
            # Finish cleanup even if the deleting browser disconnects.
            await asyncio.shield(worker.cancel_unneeded(job_id))
        return room.snapshot()

    @app.get('/v1/room/events')
    async def room_events(job_id: str | None = None, playing_entry_id: str | None = None):
        return StreamingResponse(room.events(job_id, playing_entry_id), media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post('/v1/room/events', status_code=204)
    async def notify_room_events():
        # External repair workers write the same local store, then request a
        # fresh snapshot. This endpoint cannot supply or approve job results.
        updates.publish()
        return Response(status_code=204)

    @app.post('/v1/jobs', status_code=202)
    async def submit(payload: JobRequest, request: Request, response: Response):
        doc, cached = store.create(payload, version)
        if not cached:
            worker.queue.put_nowait(doc['id'])
            updates.publish()
        elif doc['status'] == 'completed':
            response.status_code = 200
        return dict(**public_job(doc, request), cache_hit=cached)

    @app.get('/v1/jobs')
    async def list_jobs(request: Request, limit: int = Query(20, ge=1, le=100)):
        return dict(jobs=[public_job(doc, request) for doc in store.list(limit)])

    @app.get('/v1/jobs/{job_id}', name='get_job')
    async def get_job(job_id: UUID, request: Request):
        doc = store.read(str(job_id))
        if doc is None:
            raise HTTPException(404, 'Unknown job')
        return public_job(doc, request)

    @app.get('/v1/jobs/{job_id}/artifacts/{name}', name='get_artifact')
    async def get_artifact(job_id: UUID, name: str, v: str | None = None):
        doc = store.read(str(job_id))
        if doc is None:
            raise HTTPException(404, 'Unknown job')
        if doc['status'] == 'evicted':
            raise HTTPException(410, '歌曲缓存已回收，请重新点歌')
        if doc.get('restore_video') and name == 'mv':
            raise HTTPException(410, '视频缓存已回收，重新点歌即可补回')
        if doc['status'] != 'completed' and not doc.get('restore_video'):
            raise HTTPException(409, 'Artifacts are available when the job completes')
        artifact = doc['result']['artifacts'].get(name)
        if artifact is None:
            raise HTTPException(404, 'Unknown artifact')
        if v is not None and v != artifact.get('sha256'):
            raise HTTPException(409, '歌曲文件已更新，请重新加载')
        folder = store.root/str(job_id)
        path = (folder/artifact['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise HTTPException(404, 'Artifact is unavailable')
        response = LeasedFileResponse(path, cache=store.cache, job_id=str(job_id),
                            record_access=False,
                            headers={'Cache-Control': 'private, max-age=31536000, immutable' if v else 'no-cache'},
                            media_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                            filename=path.name, content_disposition_type='inline')
        try:
            if name != 'poster':
                await run_in_threadpool(store.touch, str(job_id))
        except BaseException:
            store.cache.release(str(job_id))
            raise
        return response

    def read_playback_window(job_id, stem, chunk, sample_rate, version):
        key = str(job_id)
        doc = store.read(key)
        if doc is None:
            raise HTTPException(404, 'Unknown job')
        if doc['status'] == 'evicted':
            raise HTTPException(410, '歌曲缓存已回收，请重新点歌')
        if doc['status'] != 'completed':
            raise HTTPException(409, '歌曲尚未准备完成')
        current_version = playback_version(doc['result'])
        if version is not None and version != current_version:
            raise HTTPException(409, '歌曲音轨已更新，请重新加载')
        names = playback_stems(doc['result']['artifacts'])
        if chunk < 0 or stem is not None and stem not in names:
            raise HTTPException(404, 'Unknown audio window')
        if stem is not None:
            store.touch(key)
        def check_publication():
            if current_version is not None:
                latest = store.read(key)
                if not latest or latest['status'] != 'completed' or playback_version(latest['result']) != current_version:
                    raise HTTPException(409, '歌曲音轨已更新，请重新加载')
        try:
            paths, info = playback_sources(store.cache.folder(key), doc['result'], sample_rate)
            if stem is None:
                check_publication()
                return {**info, 'version': current_version}
            path = paths[names.index(stem)]
            stamp = path.stat()
            gain = doc['result']['shared_preview_gain']
            cache_key = (current_version, path, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns, gain, chunk,
                   *(info[name] for name in ('sample_rate', 'channels', 'frames', 'chunk_frames', 'padding_frames')))
            def encode():
                data = playback_window_bytes(path, info, chunk, gain)
                check_publication()  # A replacement during decoding must not enter the old cache.
                return data
            data = playback_windows.get(cache_key, encode)
            return Response(data, media_type='audio/flac', headers={'Cache-Control': 'no-store'})
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            raise HTTPException(409, '播放音轨不完整，请重新准备') from error

    @app.get('/v1/jobs/{job_id}/playback')
    @app.get('/v1/jobs/{job_id}/playback/{stem}/{chunk}')
    async def playback_window(job_id: UUID, stem: str | None = None, chunk: int = 0,
                              sample_rate: int = Query(48000, ge=8000, le=192000), v: str | None = None):
        # Workers take the next window without waiting for the busy event loop
        # to release a semaphore. Keep encoding separate from file transfers.
        key = str(job_id)
        store.cache.acquire(key, record_access=False)
        try:
            work = playback_pool.submit(read_playback_window, job_id, stem, chunk, sample_rate, v)
        except BaseException:
            store.cache.release(key)
            raise
        task = asyncio.wrap_future(work)
        playback_tasks.add(task)

        def finished(done):
            # Lease changes stay on the event loop, alongside cache eviction.
            store.cache.release(key)
            playback_tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Observe errors even after HTTP cancellation.

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            work.cancel()  # Cancels queued work; running threads retain leases.
            raise

    assets = StaticFiles(directory=ROOT/'web')
    app.mount('/assets', assets, name='assets')

    @app.get('/ui/{version}/{path:path}', include_in_schema=False)
    async def frontend_asset(version: str, path: str, request: Request):
        return await assets.get_response(path, request.scope)

    frontend = None
    frontend_checked = -math.inf

    @app.get('/', include_in_schema=False)
    async def home(request: Request):
        nonlocal frontend, frontend_checked
        # Hot deployments remain visible within a second. Concurrent arrivals
        # reuse the same HTML instead of rereading/hashing every asset per user.
        if frontend is None or time.monotonic() - frontend_checked >= 1:
            web = ROOT/'web'
            digest = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(web.rglob('*'))
                if p.suffix in ('.js', '.mjs', '.css'))).hexdigest()[:16]
            html = (web/'index.html').read_text().replace('/assets/', f'/ui/{digest}/')
            frontend = HTMLResponse(html, headers={'Cache-Control': 'no-cache',
                'ETag': '"' + hashlib.sha256(html.encode()).hexdigest() + '"'})
            frontend_checked = time.monotonic()
        if request.headers.get('if-none-match') == frontend.headers['etag']:
            return Response(status_code=304, headers={key: frontend.headers[key] for key in ('etag', 'cache-control')})
        return frontend

    return app


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host=os.getenv('KARAOKE_HOST', '127.0.0.1'),
                port=int(os.getenv('KARAOKE_PORT', '8746')), timeout_graceful_shutdown=5,
                timeout_keep_alive=2*CHUNK_SECONDS)
