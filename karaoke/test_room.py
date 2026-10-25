import asyncio
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from karaoke.api import Store, Worker, create_app
from karaoke.job_schema import JobRequest
from karaoke.room import AddSong, Room, RoomCommand, Updates
from karaoke.test_api import URL, cache_fixture, fake_worker, finished


async def read_room_event(stream):
    text = (await anext(stream) + await anext(stream)).decode()
    return json.loads(text.split('data: ', 1)[1])


def ready_job(store, url=URL):
    job, _ = store.create(JobRequest(url=url), 'test')
    folder = store.root/job['id']
    artifacts = {}
    for name in ['mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview']:
        (folder/name).write_bytes(b'media')
        artifacts[name] = dict(path=name, bytes=5, sha256='fixture')
    return store.update(job['id'], status='completed', result=dict(title='Test song', artist='Singer',
                        duration_seconds=10, artifacts=artifacts))


@pytest.mark.parametrize('path, key', [('/v1/library', 'songs'), ('/v1/room', 'library')])
def test_library_keeps_completed_song_after_1000_newer_jobs(tmp_path, path, key):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    with TestClient(app) as client:
        oldest = ready_job(store)
        for number in range(1000):
            store.create(JobRequest(url=f'https://www.youtube.com/watch?v=capacity{number:03d}', force=True), 'test')
        songs = client.get(path).json()[key]
        assert len(songs) == 1001
        song = next(song for song in songs if song['job_id'] == oldest['id'])
        assert song['status'] == 'completed'
        assert client.get(song['artifacts']['mv']).content == b'media'


@pytest.mark.parametrize('distinct_sources', [False, True])
def test_progress_snapshots_do_not_read_retired_attempts_after_database_upgrade(tmp_path, monkeypatch, distinct_sources):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    prepared = ready_job(store)
    active, _ = store.create(JobRequest(url=URL, force=True), 'test')
    rows = []
    for number in range(10000):
        job: dict[str, Any] = dict(active, id=str(uuid4()), status='failed' if number % 2 else 'evicted',
                   created_at='2000-01-01T00:00:00Z')
        if distinct_sources:
            job['input'] = {**job['input'], 'url': f'https://www.youtube.com/watch?v=retired{number:04d}'}
        rows.append((job['id'], 'retired', job['status'], job['created_at'], json.dumps(job)))
    with store.connect() as db:
        db.executemany('INSERT INTO jobs VALUES (?,?,?,?,?)', rows)
        # An existing installation has the history but no source index yet.
        db.execute('DROP TABLE IF EXISTS library_sources')
        db.execute('DROP INDEX IF EXISTS jobs_library_media')
        db.execute('DROP INDEX IF EXISTS jobs_library_candidates')
        db.execute('DROP INDEX IF EXISTS jobs_library_available')
        db.execute('DROP INDEX IF EXISTS jobs_source_latest')
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    state = room.read()
    state['entries'] = [dict(id='active', job_id=active['id'])]
    room.save(state)
    decoded, steps = [], []
    loads, connect = json.loads, store.connect

    def counted_loads(value, *args, **kwargs):
        result = loads(value, *args, **kwargs)
        if isinstance(result, dict) and 'input' in result:
            decoded.append(result['id'])
        return result

    @contextmanager
    def counted_connect():
        with connect() as db:
            db.set_progress_handler(lambda: steps.append(True) or 0, 100)
            yield db

    monkeypatch.setattr(json, 'loads', counted_loads)
    monkeypatch.setattr(store, 'connect', counted_connect)
    previous = None
    for completed in range(3):
        progress = {'download': dict(completed=completed, total=3, unit='bytes')}
        store.update(active['id'], status='running', stage_progress=progress)
        frame = room.feed.frame(previous)
        data = json.loads(b''.join(frame).decode().split('data: ', 1)[1])
        if previous is None:
            assert len(data['library']) == 1 and data['library'][0]['job_id'] == prepared['id']
            assert data['entries'][0]['stage_progress'] == progress
        else:
            assert data['collections']['entries']['upsert'][0]['stage_progress'] == progress
        previous = room.feed.generation
    assert set(decoded) == {prepared['id'], active['id']}
    # Allow ample room for indexed queries and writes, but not a history scan.
    assert len(steps) < 200
    decoded.clear()
    steps.clear()
    new, hit = store.create(JobRequest(url='https://youtu.be/NEW00000001'), 'test')
    assert not hit and new['status'] == 'queued'
    assert not decoded and len(steps) < 10  # Neither job matching nor download reuse scans history.


@pytest.mark.parametrize('missing_newest_media', [False, True])
def test_library_keeps_older_media_when_a_newer_completed_attempt_is_incomplete(tmp_path, missing_newest_media):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    older = ready_job(store)
    newer, _ = store.create(JobRequest(url=URL, force=True), 'test')
    for artifact in older['result']['artifacts'].values():
        (tmp_path/newer['id']/artifact['path']).write_bytes(b'media')
    store.update(newer['id'], status='completed', result=older['result'])
    if missing_newest_media:
        (tmp_path/newer['id']/'mv').unlink()
    card = next(song for song in room.library() if song['url'] == URL)
    assert card['id'] == newer['id']
    assert card['job_id'] == (older if missing_newest_media else newer)['id']
    assert card['status'] == 'completed'


def test_library_keeps_queue_failures_without_unrelated_retired_sources(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    queued, _ = store.create(JobRequest(url=URL, force=True), 'test')
    newer, _ = store.create(JobRequest(url=URL, force=True), 'test')
    retired, _ = store.create(JobRequest(url='https://www.youtube.com/watch?v=retired0000'), 'test')
    for job in (queued, newer, retired):
        store.update(job['id'], status='failed')
    state = room.read()
    state['entries'] = [dict(id='queued', job_id=queued['id'])]
    room.save(state)
    snapshot = room.snapshot()
    assert snapshot['entries'][0]['job_id'] == queued['id']
    assert snapshot['entries'][0]['status'] == 'failed'
    assert len(snapshot['library']) == 1
    assert snapshot['library'][0]['id'] == newer['id']
    assert snapshot['library'][0]['status'] == 'failed'
    state['entries'] = []
    room.save(state)
    assert room.library() == []
    assert all(store.read(job['id'])['status'] == 'failed' for job in (queued, newer, retired))


def test_snapshot_shares_batch_reads_but_keeps_older_entries_and_fresh_metadata(tmp_path, monkeypatch):
    room = create_app(tmp_path, version='test').state.room
    jobs = [room.store.create(JobRequest(url=f'https://www.youtube.com/watch?v=abcdefghij{i}'), 'test')[0]
            for i in range(3)]
    state = room.read()
    state['entries'] = [dict(id=f'entry-{i}', job_id=job['id']) for i, job in enumerate(jobs)]
    state['entries'].append(dict(id='missing', job_id='missing'))
    room.save(state)
    featured = [dict(id=f'catalog-{i}', url=job['input']['url'], title=f'Catalog {i}')
                for i, job in enumerate(jobs)]
    batches, catalogs, single_reads = [], [], []
    read = room.store.read

    def read_one(key):
        single_reads.append(key)
        return read(key)

    monkeypatch.setattr(room.store, 'read', read_one)
    # A queued entry missing from the batch still needs its metadata.
    monkeypatch.setattr(room.store, 'library_jobs', lambda queue_job_ids: batches.append(True) or jobs[:2])
    monkeypatch.setattr(room, 'featured', lambda: catalogs.append(True) or featured)
    snapshot = room.snapshot()
    assert [song['title'] for song in snapshot['entries']] == ['Catalog 0', 'Catalog 1', 'Catalog 2', '歌曲已不可用']
    assert [song['id'] for song in snapshot['entries']] == [e['id'] for e in state['entries']]
    assert len(batches) == len(catalogs) == 1
    assert single_reads == [jobs[2]['id'], 'missing']
    assert [song['title'] for song in snapshot['library']] == ['Catalog 0', 'Catalog 1', 'Catalog 2']
    jobs[0] = room.store.update(jobs[0]['id'], song_info={'title': 'Resolved title'})
    featured[1]['title'] = 'Updated catalog title'
    refreshed = room.snapshot()
    assert [song['title'] for song in refreshed['entries'][:2]] == ['Resolved title', 'Updated catalog title']
    assert [song['title'] for song in refreshed['library'][:2]] == ['Resolved title', 'Updated catalog title']


def test_room_distinguishes_missing_lyrics_from_alignment_review(tmp_path):
    app = create_app(tmp_path, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert not entry['lyrics_unavailable']
        for mode in ('unavailable', 'partial_word', 'line'):
            result = dict(job['result'], quality=dict(lyrics_provenance=dict(needs_review=True),
                                                     playback_fallback=dict(subtitle_mode=mode)))
            app.state.store.update(job['id'], result=result)
            song = client.get('/v1/room').json()['entries'][0]
            assert song['lyrics_need_review']
            assert song['lyrics_unavailable'] == (mode == 'unavailable')


def test_poster_survives_cleanup_and_browsing_does_not_refresh_lru(tmp_path):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job = ready_job(store)
    folder = store.root/job['id']
    (folder/'poster.jpg').write_bytes(b'cover')
    job['result']['artifacts']['poster'] = dict(path='poster.jpg', bytes=5, sha256='fixture')
    store.update(job['id'], result=job['result'], last_accessed_at='2000-01-01T00:00:00Z')
    store.cache.compact(store.read(job['id']))
    with TestClient(app) as client:
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['job_id'] == job['id'])
        poster = song['poster']
        assert poster == f"/v1/jobs/{job['id']}/artifacts/poster?v=fixture"
        response = client.get(poster)
        assert response.status_code == 200 and response.content == b'cover'
        assert response.headers['content-type'] == 'image/jpeg'
        assert store.read(job['id'])['last_accessed_at'] == '2000-01-01T00:00:00Z'
        assert not store.cache.leases
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert entry['poster'] == poster
        (folder/'poster.jpg').unlink()
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][-1]
        assert entry['status'] == 'completed' and entry['poster'] is None
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['job_id'] == job['id'])
        assert song['status'] == 'completed'
        reused, cached = store.create(JobRequest(url=URL), 'test')
        assert cached and reused['id'] == job['id']
        store.update(job['id'], status='evicted')
        assert client.get(poster).status_code == 410
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['id'] == job['id'])
        assert song['poster'] is None and song['job_id'] is None


def test_repaired_media_changes_urls_without_changing_room_or_other_artifacts(tmp_path):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job = ready_job(store)
    with TestClient(app) as client:
        before = client.post('/v1/room/entries', json={'job_id':job['id']}).json()
        entry = before['entries'][0]
        old_url = entry['artifacts']['mv']
        assert client.get(old_url).content == b'media'
        updated = b'fixed'
        (tmp_path/job['id']/'mv').write_bytes(updated)
        job['result']['artifacts']['mv']['sha256'] = hashlib.sha256(updated).hexdigest()
        store.update(job['id'], result=job['result'])
        after = client.get('/v1/room').json()
        repaired = after['entries'][0]
        new_url = repaired['artifacts']['mv']
        assert new_url != old_url
        assert repaired['id'] == entry['id'] and repaired['job_id'] == job['id']
        assert {k:after[k] for k in ('revision','playing','position','levels')} == {
            k:before[k] for k in ('revision','playing','position','levels')}
        assert {k:v for k,v in repaired['artifacts'].items() if k != 'mv'} == {
            k:v for k,v in entry['artifacts'].items() if k != 'mv'}
        song = next(s for s in after['library'] if s['job_id'] == job['id'])
        assert song['artifacts']['mv'] == new_url
        public = client.get(f"/v1/jobs/{job['id']}").json()['result']['artifacts']['mv']
        assert public['url'].endswith(new_url)
        response = client.get(new_url, headers={'Range':'bytes=1-3'})
        assert response.status_code == 206 and response.content == b'ixe'
        assert 'immutable' in response.headers['cache-control']
        assert client.get(old_url, headers={'Range':'bytes=1-3'}).status_code == 409
        legacy = client.get(new_url.split('?')[0])
        assert legacy.content == updated and legacy.headers['cache-control'] == 'no-cache'


def test_opening_countdown_waits_for_ready_and_keeps_original_song_clock(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job = ready_job(store)
    opening = dict(start=-3.369, onset=.631, dots=3, step_seconds=1., settle_seconds=1.)
    job['result']['quality'] = dict(opening_countdown=opening)
    store.update(job['id'], result=job['result'])
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert entry['opening_countdown'] == opening
        def command(action, **args):
            return client.post('/v1/room/commands', json=dict(action=action, entry_id=entry['id'], **args)).json()
        waiting = command('play')
        assert waiting['started_at'] is None and waiting['position'] == 0
        playing = command('ready')
        assert playing['started_at'] == pytest.approx(103.369)
        assert command('ready')['started_at'] == playing['started_at']
        now[0] = 102.
        assert app.state.room.position(app.state.room.read()) == 0
        paused = command('pause')
        assert paused['position'] == 0 and paused['started_at'] is None
        command('play')
        assert command('ready')['started_at'] == pytest.approx(105.369)
        now[0] = 106.
        assert command('pause')['position'] == pytest.approx(.631)
        command('play')
        assert command('ready')['started_at'] == 106.  # Resuming never inserts time into the song.
        command('seek', position=6.)
        assert command('pause')['position'] == 6.
        command('play')
        assert command('ready')['started_at'] == 106.
        assert command('seek', position=0.)['started_at'] is None
        assert command('ready')['started_at'] == pytest.approx(109.369)


def test_playlist_duplicate_song_order_and_stale_end_event(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        ids = []
        for _ in range(3):
            data = client.post('/v1/room/entries', json={'job_id':job['id']}).json()
            ids.append(data['entries'][-1]['id'])
        assert len(set(ids)) == 3
        assert len(app.state.store.list()) == 1
        first = client.post('/v1/room/commands', json={'action':'play'}).json()
        assert first['current_id'] == ids[0] and first['started_at'] is None
        playing = client.post('/v1/room/commands', json={'action':'ready','entry_id':ids[0]}).json()
        assert playing['started_at'] is not None
        response = client.post('/v1/room/commands', json={'action':'reorder','order':[ids[0],ids[2],ids[1]],'revision':playing['revision']})
        assert response.status_code == 200
        moved = client.post('/v1/room/commands', json={'action':'reorder','order':[ids[2],ids[0],ids[1]]})
        assert moved.status_code == 200
        assert moved.json()['current_id'] == ids[0]
        following = client.post('/v1/room/commands', json={'action':'next','entry_id':ids[0]}).json()
        assert following['current_id'] == ids[2]
        stale = client.post('/v1/room/commands', json={'action':'next','entry_id':ids[0]})
        assert stale.status_code == 409
        assert client.get('/v1/room').json()['current_id'] == ids[2]
        assert client.post('/v1/room/commands', json={'action':'reorder','order':[ids[2],ids[2]]}).status_code == 422


def test_two_clients_share_queue_without_lost_adds_or_accidental_duplicates(tmp_path):
    import httpx

    app = create_app(tmp_path, version='test')
    jobs = [ready_job(app.state.store, url) for url in
            (URL, 'https://www.youtube.com/watch?v=abcdefghijk', 'https://www.youtube.com/watch?v=12345678901')]

    async def check():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as alice, \
                httpx.AsyncClient(transport=transport, base_url='http://test') as bob:
            first, same = await asyncio.gather(
                alice.post('/v1/room/entries', json={'job_id':jobs[0]['id'], 'if_absent':True}),
                bob.post('/v1/room/entries', json={'url':'https://youtu.be/n5sjpCQhTDk?t=10', 'if_absent':True}))
            assert first.status_code == same.status_code == 201
            initial = first.json()
            assert len(initial['entries']) == 1
            assert same.json()['entries'] == initial['entries']
            added = await asyncio.gather(*[
                client.post('/v1/room/entries', json={'job_id':job['id'], 'if_absent':True})
                for client, job in zip((alice, bob), jobs[1:])])
            assert all(response.status_code == 201 for response in added)
            for client in (alice, bob):
                snapshot = (await client.get('/v1/room')).json()
                assert len(snapshot['entries']) == 3
                assert snapshot['entries'][0]['job_id'] == jobs[0]['id']
                assert {e['job_id'] for e in snapshot['entries']} == {job['id'] for job in jobs}
            stale = await alice.post('/v1/room/commands', json={
                'action':'reorder', 'order':[e['id'] for e in initial['entries']], 'revision':initial['revision']})
            assert stale.status_code == 409
            await asyncio.gather(
                alice.post('/v1/room/commands', json={'action':'levels','levels':{'lead_vocals':.3}}),
                bob.post('/v1/room/commands', json={'action':'levels','levels':{'backing_vocals':.6}}))
            for client in (alice, bob):
                snapshot = (await client.get('/v1/room')).json()
                assert len(snapshot['entries']) == 3
                assert snapshot['levels'] == dict(instrumental=1, backing_vocals=.6, lead_vocals=.3)
            repeat = await bob.post('/v1/room/entries', json={'job_id':jobs[0]['id'], 'if_absent':False})
            assert repeat.status_code == 201
            assert len(repeat.json()['entries']) == 4
            assert repeat.json()['entries'][-1]['id'] != initial['entries'][0]['id']
    asyncio.run(check())


def test_device_finish_is_idempotent_and_does_not_move_another_legacy_transport(tmp_path):
    app = create_app(tmp_path, version='test')
    room = app.state.room
    job = ready_job(app.state.store)
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    second = room.add(AddSong(job_id=job['id']))['entries'][1]
    room.command(RoomCommand(action='play'))
    playing = room.command(RoomCommand(action='ready', entry_id=first['id']))
    with TestClient(app) as client:
        body = dict(action='finish', entry_id=second['id'], playback_started_at=time.time()-5)
        done = client.post('/v1/room/commands', json=body).json()
        assert [e['id'] for e in done['entries']] == [first['id']]
        assert [e['id'] for e in done['history']] == [second['id']]
        assert {k:done[k] for k in ('current_id','playing','position','started_at')} == {
            k:playing[k] for k in ('current_id','playing','position','started_at')}
        repeated = client.post('/v1/room/commands', json=body).json()
        assert repeated['revision'] == done['revision'] and repeated['history'] == done['history']


@pytest.mark.parametrize('status', ['queued', 'running', 'failed', 'audio_ready'])
def test_unprepared_queue_entry_survives_finish_and_skip_until_explicit_removal(tmp_path, status):
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    store.update(job['id'], status=status, result=None)
    before = room.read()
    ignored = room.command(RoomCommand(action='finish', entry_id=entry['id'], playback_started_at=time.time()-5))
    assert [e['id'] for e in ignored['entries']] == [entry['id']]
    assert room.read() == before
    room.command(RoomCommand(action='play'))
    skipped = room.command(RoomCommand(action='next', entry_id=entry['id']))
    assert [e['id'] for e in skipped['entries']] == [entry['id']]
    assert skipped['history'] == []
    removed = room.command(RoomCommand(action='remove', entry_id=entry['id']))
    assert removed['entries'] == [] and removed['history'] == []


def test_request_only_device_cannot_reorder_a_live_performance_behind_other_songs(tmp_path):
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    ids = [room.add(AddSong(job_id=job['id']))['entries'][-1]['id'] for _ in range(3)]
    transport = {key:room.read()[key] for key in ('current_id', 'playing', 'position', 'started_at')}

    async def check():
        first, duplicate = room.events(job['id'], ids[0]), room.events(job['id'], ids[0])
        stale = room.events(job['id'], 'removed-entry')
        try:
            event = await read_room_event(first)
            assert event['playing_entry_ids'] == [ids[0]]
            await read_room_event(duplicate)
            await read_room_event(stale)
            # The phone can have an older snapshot from before playback began.
            # The server uses live presence, not the phone's local play button.
            moved = room.command(RoomCommand(action='reorder', order=[ids[2], ids[0], ids[1]]))
            assert [e['id'] for e in moved['entries']] == [ids[0], ids[2], ids[1]]
            assert {key:room.read()[key] for key in transport} == transport
            await first.aclose()
            assert room.snapshot()['playing_entry_ids'] == [ids[0]]
            await duplicate.aclose()
            assert room.snapshot()['playing_entry_ids'] == []
            idle = room.command(RoomCommand(action='reorder', order=[ids[2], ids[0], ids[1]]))
            assert [e['id'] for e in idle['entries']] == [ids[2], ids[0], ids[1]]
        finally:
            await first.aclose()
            await duplicate.aclose()
            await stale.aclose()
    asyncio.run(check())


def test_device_stream_protects_media_after_shared_queue_removal_until_disconnect(tmp_path):
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    store.update(job['id'], last_accessed_at='2000-01-01T00:00:00Z')

    async def check():
        one, two = room.events(job['id'], entry['id']), room.events(job['id'], entry['id'])
        try:
            event = await read_room_event(one)
            assert event['playback_entry']['job_id'] == job['id']
            assert event['playing_entry_ids'] == [entry['id']]
            await read_room_event(two)
            room.command(RoomCommand(action='finish', entry_id=entry['id'], playback_started_at=time.time()-5))
            assert not room.read()['entries']
            assert not room.snapshot()['playing_entry_ids']
            assert store.cache.leases[job['id']] == 2
            store.cache.limit = 1
            store.cache.enforce()
            assert store.read(job['id'])['status'] == 'completed'
            assert store.read(job['id'])['last_accessed_at'] == '2000-01-01T00:00:00Z'
            await one.aclose()
            assert store.cache.leases[job['id']] == 1
            assert 'playback_entry' in await read_room_event(two)
        finally:
            await one.aclose()
            await two.aclose()
        assert not store.cache.leases and not room.updates.subscribers
        store.cache.enforce()
        assert store.read(job['id'])['status'] != 'completed'
    asyncio.run(check())


def test_sung_history_keeps_performances_newest_first_across_restart_and_requeue(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    second = room.add(AddSong(job_id=job['id']))['entries'][1]
    room.command(RoomCommand(action='play'))
    room.command(RoomCommand(action='ready', entry_id=first['id']))
    now[0] = 105.
    room.command(RoomCommand(action='next', entry_id=first['id']))
    room.command(RoomCommand(action='ready', entry_id=second['id']))
    now[0] = 110.
    room.pause_on_shutdown()
    now[0] = 115.
    done = room.command(RoomCommand(action='next', entry_id=second['id']))

    assert done['entries'] == [] and done['current_id'] is None
    assert [entry['id'] for entry in done['history']] == [second['id'], first['id']]
    assert [entry['played_at'] for entry in done['history']] == [105., 100.]
    assert [entry['finished_at'] for entry in done['history']] == [115., 105.]
    assert all(entry['title'] == 'Test song' and entry['artist'] == 'Singer' and entry['url'] == URL
               for entry in done['history'])
    assert job['id'] not in store.cache.protected(store.list())
    with pytest.raises(HTTPException) as stale:
        room.command(RoomCommand(action='next', entry_id=second['id']))
    assert stale.value.status_code == 409

    restored = Room(store, app.state.worker, 'test', Updates())
    assert restored.snapshot()['history'] == done['history']
    again = restored.add(AddSong(job_id=job['id']))
    assert again['history'] == done['history']
    assert again['entries'][0]['id'] not in (first['id'], second['id'])
    assert again['entries'][0]['job_id'] == job['id'] and len(store.list()) == 1
    restored.command(RoomCommand(action='remove', entry_id=again['entries'][0]['id']))
    store.update(job['id'], status='evicted', result=None, song_info={})
    assert restored.snapshot()['history'] == done['history']


@pytest.mark.parametrize('mode', ['loading', 'unprepared', 'opening', 'paused_opening', 'seek_without_play'])
def test_unplayed_entries_do_not_enter_sung_history(tmp_path, monkeypatch, mode):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    job['result']['quality'] = dict(opening_countdown=dict(start=-4.))
    store.update(job['id'], result=job['result'], status='queued' if mode == 'unprepared' else 'completed')
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    room.command(RoomCommand(action='play'))
    if mode in ('opening', 'paused_opening', 'unprepared'):
        room.command(RoomCommand(action='ready', entry_id=entry['id']))
    now[0] = 101.
    if mode == 'paused_opening':
        room.command(RoomCommand(action='pause', entry_id=entry['id']))
        now[0] = 110.
    if mode == 'seek_without_play':
        room.command(RoomCommand(action='seek', entry_id=entry['id'], position=5.))
    done = room.command(RoomCommand(action='next', entry_id=entry['id']))
    assert [e['id'] for e in done['entries']] == ([entry['id']] if mode == 'unprepared' else [])
    assert done['history'] == []


def test_select_archives_sung_current_and_keeps_unplayed_entries_in_queue(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    for _ in range(3):
        room.add(AddSong(job_id=job['id']))
    first, second, third = room.read()['entries']
    room.command(RoomCommand(action='play'))
    room.command(RoomCommand(action='ready', entry_id=first['id']))
    now[0] = 101.
    selected = room.command(RoomCommand(action='select', entry_id=second['id']))
    assert [e['id'] for e in selected['entries']] == [second['id'], third['id']]
    assert [e['id'] for e in selected['history']] == [first['id']]
    selected = room.command(RoomCommand(action='select', entry_id=third['id']))
    assert [e['id'] for e in selected['entries']] == [third['id'], second['id']]
    assert [e['id'] for e in selected['history']] == [first['id']]
    room.command(RoomCommand(action='ready', entry_id=third['id']))
    now[0] = 102.
    removed = room.command(RoomCommand(action='remove', entry_id=third['id']))
    assert [e['id'] for e in removed['history']] == [third['id'], first['id']]
    assert removed['current_id'] == second['id']


def test_room_preserves_existing_entries_when_adding_past_one_hundred(tmp_path):
    (tmp_path/'preparation.pause').write_text('test')
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    user_job = ready_job(app.state.store)
    with TestClient(app) as client:
        original = client.post('/v1/room/entries', json={'job_id':user_job['id']}).json()['entries'][0]
        room = app.state.room.read()
        room['entries'] = [{**room['entries'][0], 'id':str(index)} for index in range(100)]
        app.state.room.save(room)
        response = client.post('/v1/room/entries', json={'job_id':user_job['id']})
        assert response.status_code == 201
        entries = response.json()['entries']
        assert [e['id'] for e in entries[:100]] == [str(index) for index in range(100)]
        assert len(entries) == 101 and entries[-1]['job_id'] == original['job_id']


def test_soak_adds_all_cases_to_room_and_only_removes_finished_owned_entries(tmp_path):
    from karaoke.tools.soak import queue_cases, release_finished_entries
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    user_job = ready_job(app.state.store)
    cases: list[dict[str, Any]] = [dict(url=f'https://www.youtube.com/watch?v={i:011d}',
        title=f'Test {i}', provider='youtube', attempts=[]) for i in range(3)]
    state = {'provider_backoff': {}, 'cases': cases}
    original_urls = [case['url'] for case in state['cases']]
    for case, status in zip(state['cases'], ('passed', 'failed', 'failed')):
        case['attempts'].append(dict(status=status, pipeline_version='previous'))
    with TestClient(app) as client:
        user = client.post('/v1/room/entries', json={'job_id': user_job['id']}).json()['entries'][0]
        state['provider_backoff']['openai'] = time.time()+60
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        assert len(client.get('/v1/room').json()['entries']) == 1
        state['provider_backoff'].clear()
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        entries = client.get('/v1/room').json()['entries']
        assert len(entries) == 4  # The abandoned queue=2 idea must not become an app limit.
        attempts = [c['attempts'][-1] for c in state['cases']]
        assert {a['entry_id'] for a in attempts} == {e['id'] for e in entries[1:]}
        assert [e['id'] for e in entries[1:]] == [attempts[i]['entry_id'] for i in (1, 2, 0)]
        assert [case['url'] for case in state['cases']] == original_urls
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        assert len(client.get('/v1/room').json()['entries']) == 4
        attempts[0]['status'] = 'passed'
        attempts[1]['status'] = 'failed'
        client.post('/v1/room/commands', json={'action':'select','entry_id':attempts[0]['entry_id']})
        release_finished_entries(client, 'http://testserver', state)
        remaining = {e['id'] for e in client.get('/v1/room').json()['entries']}
        assert remaining == {user['id'], attempts[0]['entry_id'], attempts[2]['entry_id']}


@pytest.mark.parametrize('by_id', [False, True])
def test_room_can_force_full_revalidation_without_reusing_a_completed_job(tmp_path, by_id):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    old = ready_job(app.state.store)
    with TestClient(app) as client:
        request = {'job_id':old['id']} if by_id else {'url':URL}
        entry = client.post('/v1/room/entries', json={**request,'force':True}).json()['entries'][0]
        assert entry['job_id'] != old['id']
        assert finished(client, entry['job_id'])['status'] == 'completed'


@pytest.mark.parametrize('by_id', [False, True])
@pytest.mark.parametrize('already_queued', [False, True])
def test_replaying_completed_song_keeps_media_by_id_or_url_across_versions(tmp_path, by_id, already_queued):
    app = create_app(tmp_path, worker_command=fake_worker, version='new')
    old = ready_job(app.state.store)
    if already_queued:
        state = app.state.room.read()
        state['entries'] = [dict(id='waiting',job_id=old['id'])]
        app.state.room.save(state)
    (tmp_path/'preparation.pause').touch()
    with TestClient(app) as client:
        request: dict[str, Any] = {'job_id':old['id']} if by_id else {'url':URL}
        request['if_absent'] = True
        first = client.post('/v1/room/entries',json=request).json()
        assert len(first['entries']) == 1
        entry = first['entries'][0]
        job = app.state.store.read(entry['job_id'])
        assert job['status']=='completed'
        assert job['pipeline_version']==old['pipeline_version']
        assert app.state.worker.queue.empty()
        assert client.post('/v1/room/entries',json=request).json()['entries']==first['entries']
        assert len(app.state.store.list(-1)) == 1
        if already_queued:
            assert entry['id']=='waiting'
        (tmp_path/'preparation.pause').unlink()
        assert finished(client,entry['job_id'])['status']=='completed'


def test_refresh_does_not_replace_current_or_leased_media(tmp_path):
    app = create_app(tmp_path, version='new')
    old = ready_job(app.state.store)
    state = app.state.room.read()
    state.update(entries=[dict(id='current',job_id=old['id'])],current_id='current',playing=True,position=4,started_at=time.time())
    app.state.room.save(state)
    first = app.state.room.add(AddSong(job_id=old['id'],if_absent=True))
    assert len(first['entries'])==1 and first['playing']
    second = app.state.room.add(AddSong(job_id=old['id']))
    assert second['entries'][0]['job_id']==second['entries'][1]['job_id']==old['id']
    assert app.state.store.read(old['id'])['status']=='completed'
    assert second['current_id']=='current' and second['playing']
    leased = ready_job(app.state.store,url='https://www.youtube.com/watch?v=2Vv-BfVoq4g')
    app.state.store.cache.acquire(leased['id'])
    fresh,_ = app.state.store.create(JobRequest(url=leased['input']['url']),'new')
    assert fresh['id']==leased['id'] and app.state.store.read(leased['id'])['status']=='completed'


def test_wait_for_preparation_then_play_pause_seek_mix_and_persist(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        data = client.post('/v1/room/entries', json={'url':URL,'title':'New song'}).json()
        entry = data['entries'][0]
        client.post('/v1/room/commands', json={'action':'play'})
        assert client.get('/v1/room').json()['started_at'] is None
        assert finished(client,entry['job_id'])['status'] == 'completed'
        # A ready audio renderer, not elapsed preparation time, starts playback.
        assert client.get('/v1/room').json()['started_at'] is None
        app.state.store.update(entry['job_id'], result={**app.state.store.read(entry['job_id'])['result'], 'duration_seconds':10})
        client.post('/v1/room/commands', json={'action':'ready','entry_id':entry['id']})
        client.post('/v1/room/commands', json={'action':'seek','entry_id':entry['id'],'position':6})
        paused = client.post('/v1/room/commands', json={'action':'pause','entry_id':entry['id']}).json()
        assert 6 <= paused['position'] < 7 and not paused['playing']
        mixed = client.post('/v1/room/commands', json={'action':'mix','mix':'instrumental'}).json()
        assert mixed['position'] == paused['position'] and mixed['mix'] == 'instrumental'
        assert client.post('/v1/room/commands', json={'action':'seek','entry_id':'old','position':9}).status_code == 409
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        restored = client.get('/v1/room').json()
        assert restored['current_id'] == entry['id'] and restored['position'] == paused['position']
        assert restored['mix'] == 'instrumental' and not restored['playing']
        empty = client.post('/v1/room/commands', json={'action':'remove','entry_id':entry['id']}).json()
        assert empty['entries'] == [] and empty['current_id'] is None


def test_buffering_freezes_room_until_ready_and_rejects_old_playback_events(tmp_path):
    app = create_app(tmp_path, version='test')
    app.state.store.cache.minimum_free = 0
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        def command(action, **fields):
            return client.post('/v1/room/commands', json={'action':action, 'entry_id':entry['id'], **fields})
        command('play')
        playing = command('ready').json()
        request = dict(position=4, playback_started_at=playing['started_at'])
        waiting = command('buffering', **request).json()
        assert waiting['playing'] and waiting['started_at'] is None and waiting['position'] == 4
        assert client.get('/v1/room').json()['position'] == 4
        resumed = command('ready', revision=waiting['revision']).json()
        assert resumed['started_at'] is not None and resumed['position'] == 4
        assert command('buffering', **request).status_code == 409
        seeked = command('seek', position=7).json()
        assert command('buffering', position=4, playback_started_at=resumed['started_at']).status_code == 409
        command('pause')
        assert command('buffering', position=7, playback_started_at=seeked['started_at']).status_code == 409
        assert command('buffering', entry_id='old', **request).status_code == 409


def test_independent_levels_preserve_playback_persist_and_presets_reset_them(tmp_path):
    app = create_app(tmp_path, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        client.post('/v1/room/commands', json={'action':'play'})
        before = client.post('/v1/room/commands', json={'action':'ready','entry_id':entry['id']}).json()
        changed = client.post('/v1/room/commands', json={'action':'levels','levels':{'backing_vocals':.35}}).json()
        assert changed['levels'] == dict(instrumental=1,backing_vocals=.35,lead_vocals=0)
        assert changed['started_at'] == before['started_at'] and changed['position'] == before['position']
        changed = client.post('/v1/room/commands', json={'action':'levels','levels':{'instrumental':.8,'lead_vocals':.2}}).json()
        assert changed['mix'] == 'custom' and changed['levels']['backing_vocals'] == .35
        assert client.post('/v1/room/commands', json={'action':'levels','levels':{'lead_vocals':1.1}}).status_code == 422
        assert client.post('/v1/room/commands', json={'action':'levels','levels':{}}).status_code == 422
        shifted = client.post('/v1/room/commands', json={'action':'pitch','pitch_semitones':3}).json()
        assert shifted['pitch_semitones'] == 3
        assert shifted['started_at'] == changed['started_at'] and shifted['position'] == changed['position']
        assert shifted['levels'] == changed['levels']
        for value in (-7, 7, .5, True, None):
            assert client.post('/v1/room/commands', json={'action':'pitch','pitch_semitones':value}).status_code == 422
    with TestClient(create_app(tmp_path, version='test')) as client:
        assert client.get('/v1/room').json()['levels'] == changed['levels']
        assert client.get('/v1/room').json()['pitch_semitones'] == 3
        original = client.post('/v1/room/commands', json={'action':'mix','mix':'original'}).json()
        assert original['levels'] == dict(instrumental=1,backing_vocals=1,lead_vocals=1)
        assert original['pitch_semitones'] == 3
        muted = client.post('/v1/room/commands', json={'action':'levels', 'levels':dict(instrumental=0,backing_vocals=0,lead_vocals=0)}).json()
        assert all(value == 0 for value in muted['levels'].values())


def test_pending_preparation_follows_playlist_without_interrupting_current(tmp_path):
    store = Store(tmp_path)
    worker = Worker(store)
    worker.processes['already-running'] = None
    jobs = {name:store.create(JobRequest(url=f'https://youtu.be/{i:011d}'),'test')[0]['id']
            for i,name in enumerate(['a','b','c','standalone'])}
    for job in [jobs['standalone'], *jobs.values(), jobs['a'], 'missing-job']:
        worker.queue.put_nowait(job)
    worker.prioritize([jobs[name] for name in ['c','c','a','a','b']])
    assert [worker.queue.get_nowait() for _ in range(4)] == [jobs[name] for name in ['c','a','b','standalone']]
    assert worker.queue.empty()
    assert worker.current == 'already-running'


def test_restored_playlist_priority_is_applied_before_starting_worker(tmp_path):
    app = create_app(tmp_path, worker_command=lambda folder: [sys.executable, '-c', 'import time;time.sleep(30)'], version='test')
    first, _ = app.state.store.create(JobRequest(url=URL), 'test')
    second, _ = app.state.store.create(JobRequest(url='https://youtu.be/2Vv-BfVoq4g'), 'test')
    state = app.state.room.read()
    state['entries'] = [dict(id='second',job_id=second['id']), dict(id='first',job_id=first['id'])]
    state['current_id'] = 'first'  # Persisted legacy transport cannot override the modern queue.
    app.state.room.save(state)
    with TestClient(app) as client:
        assert client.get('/health').json()['running_job'] == second['id']
        assert app.state.store.read(first['id'])['status'] == 'queued'


def test_reorder_and_cancel_recovery_use_queue_order_despite_legacy_transport(tmp_path):
    app = create_app(tmp_path, version='test')
    store, worker, room = app.state.store, app.state.worker, app.state.room
    jobs = [store.create(JobRequest(url=f'https://youtu.be/{i:011d}'), 'test')[0] for i in range(3)]
    for job in jobs:
        worker.queue.put_nowait(job['id'])
    state = room.read()
    state.update(entries=[dict(id=str(i), job_id=job['id']) for i, job in enumerate(jobs[:2])],
                 current_id='0', playing=True)
    room.save(state)
    room.command(RoomCommand(action='reorder', order=['1', '0']))
    asyncio.run(worker.cancel_unneeded(jobs[2]['id']))
    assert store.read(jobs[2]['id'])['status'] == 'cancelled'
    assert [worker.queue.get_nowait() for _ in range(2)] == [jobs[1]['id'], jobs[0]['id']]
    assert worker.queue.empty()
    assert room.read()['current_id'] == '0'  # Preparation changes do not rewrite legacy playback.


def test_removed_pending_song_does_not_create_a_new_job(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    pause = tmp_path/'preparation.pause'
    pause.touch()
    with TestClient(app) as client:
        assert client.post('/v1/room/entries',json={'url':'http://127.0.0.1/secret'}).status_code == 422
        assert client.post('/v1/room/entries',json={'job_id':'missing'}).status_code == 404
        data = client.post('/v1/room/entries',json={'url':URL}).json()
        entry = data['entries'][0]
        standalone = client.post('/v1/jobs', json={'url': 'https://youtu.be/2Vv-BfVoq4g'}).json()['id']
        client.post('/v1/room/commands',json={'action':'remove','entry_id':entry['id']})
        assert app.state.store.read(entry['job_id'])['status'] == 'cancelled'
        assert app.state.store.read(standalone)['status'] == 'queued'
        assert app.state.worker.queue.qsize() == 1
        assert client.get('/v1/room').json()['entries'] == []
        assert client.get('/').status_code == 200
        assert client.get('/assets/app.js').status_code == 200
    # Cancellation survives a restart. Re-adding explicitly resumes this job.
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        assert client.get('/health').json()['queued'] == 1
        resumed = client.post('/v1/room/entries', json={'url': URL}).json()['entries'][0]
        assert resumed['job_id'] == entry['job_id'] and resumed['status'] == 'queued'
        pause.unlink()
        assert finished(client, entry['job_id'])['status'] == 'completed'
        client.post('/v1/room/commands', json={'action': 'remove', 'entry_id': resumed['id']})
        cached = client.post('/v1/room/entries', json={'url': URL}).json()['entries'][0]
        assert cached['job_id'] == entry['job_id'] and cached['status'] == 'completed'


@pytest.mark.parametrize('spawned_first', [False, True])
def test_removing_during_process_creation_waits_for_cleanup(tmp_path, monkeypatch, spawned_first):
    import httpx

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        spawned = []
        original = asyncio.create_subprocess_exec
        async def delayed_spawn(*args, **kwargs):
            if spawned_first:
                spawned.append(await original(*args, **kwargs))
            entered.set()
            await release.wait()
            if not spawned_first:
                spawned.append(await original(*args, **kwargs))
            return spawned[-1]
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed_spawn)
        app = create_app(tmp_path, version='test',
            worker_command=lambda _: [sys.executable, '-c', 'import time; time.sleep(30)'])
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                entry = (await client.post('/v1/room/entries', json={'url': URL})).json()['entries'][0]
                await asyncio.wait_for(entered.wait(), 5)
                removing = asyncio.create_task(client.post('/v1/room/commands',
                    json={'action': 'remove', 'entry_id': entry['id']}))
                try:
                    for _ in range(100):
                        if app.state.worker.cancelled:
                            break
                        await asyncio.sleep(.01)
                    assert app.state.worker.cancelled and not removing.done()
                finally:
                    release.set()
                assert (await removing).status_code == 200
                assert len(spawned) == 1 and spawned[0].returncode is not None
                assert app.state.store.read(entry['job_id'])['status'] == 'cancelled'
                assert not app.state.worker.processes
                await asyncio.wait_for(app.state.worker.queue.join(), 1)
    asyncio.run(scenario())


@pytest.mark.parametrize('readd_during_cancel', [False, True])
@pytest.mark.parametrize('force', [False, True])
def test_remove_stops_descendants_and_resumes_only_completed_checkpoints(tmp_path, readd_during_cancel, force):
    import os
    import signal

    import httpx

    # A stubborn grandchild closes stdout, so waiting for the parent alone
    # cannot prove that all writers have stopped.
    child = """import pathlib,signal,sys,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
path=pathlib.Path(sys.argv[1])
while True:
    with path.open('a') as stream: stream.write('tick\\n')
    with pathlib.Path(sys.argv[2]).open('ab') as stream: stream.write(b'x'*1024)
    time.sleep(.02)
"""
    code = """import json,os,pathlib,signal,subprocess,sys,time
from karaoke.stage_cache import StageCache
p=pathlib.Path(sys.argv[1]); cache=StageCache(p)
source=p/'source.bin'; separated=p/'separated.bin'; aligned=p/'alignment.bin'
with cache.stage('decode',inputs=[p/'request.json'],outputs=[source]) as hit:
    if not hit: source.write_bytes(b'source')
with cache.stage('separate',inputs=[source],outputs=[separated]) as hit:
    if not hit: separated.write_bytes(b'expensive completed stage')
with cache.stage('align',inputs=[separated],outputs=[aligned]) as hit:
    if not hit:
        aligned.write_bytes(b'incomplete')
        child=subprocess.Popen([sys.executable,'-c',sys.argv[2],str(p.parent/'heartbeat'),str(p/'partial-audio.bin')],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (p/'started.json').write_text(json.dumps({'parent':os.getpid(),'child':child.pid}))
        while not (p.parent/'release').exists(): time.sleep(.02)
        child.kill(); child.wait()
        aligned.write_bytes(b'complete')
"""
    after = """result=json.loads((p/'result.json').read_text())
result['artifacts']['alignment']={'path':aligned.name,'bytes':aligned.stat().st_size,'sha256':hashlib.sha256(aligned.read_bytes()).hexdigest()}
(p/'result.json').write_text(json.dumps(dict(result,reused_stages=cache.reused)))
"""

    async def scenario():
        app = create_app(tmp_path, worker_command=lambda p: fake_worker(p, before=code, after=after, args=(child,)),
                         version='test', worker_concurrency=None)
        cached = ready_job(app.state.store) if force and readd_during_cancel else None
        worker = app.state.worker
        entered, release_cancel = asyncio.Event(), asyncio.Event()
        original = worker.terminate
        async def terminate(process):
            entered.set()
            if readd_during_cancel:
                await release_cancel.wait()
            await original(process)
        worker.terminate = terminate
        async def until(predicate):
            deadline = time.monotonic()+8
            while not predicate():
                assert time.monotonic() < deadline
                await asyncio.sleep(.02)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                entry = (await client.post('/v1/room/entries', json={'url': URL, 'force': force})).json()['entries'][0]
                folder = tmp_path/entry['job_id']
                await until(lambda: (tmp_path/'heartbeat').exists() and (tmp_path/'heartbeat').stat().st_size > 10)
                parent = json.loads((folder/'started.json').read_text())['parent']
                try:
                    removing = asyncio.create_task(client.post('/v1/room/commands',
                        json={'action': 'remove', 'entry_id': entry['id']}))
                    await entered.wait()
                    if readd_during_cancel:
                        again = (await client.post('/v1/room/entries', json={'job_id': entry['job_id']})).json()['entries'][0]
                        assert again['job_id'] == entry['job_id']
                        (tmp_path/'preparation.pause').touch()
                        release_cancel.set()
                    assert (await removing).status_code == 200
                    ticks = (tmp_path/'heartbeat').read_bytes()
                    await asyncio.sleep(.15)
                    assert (tmp_path/'heartbeat').read_bytes() == ticks
                    assert not (folder/'partial-audio.bin').exists()
                    assert not worker.processes
                    assert (folder/'separated.bin').read_bytes() == b'expensive completed stage'
                    assert (folder/'checkpoints/separate.json').is_file()
                    assert not (folder/'checkpoints/align.json').exists()
                    assert not (folder/'alignment.bin').exists()
                    assert app.state.store.read(entry['job_id'])['status'] == ('queued' if readd_during_cancel else 'cancelled')
                    assert (await client.get(f"/v1/jobs/{entry['job_id']}/artifacts/mv")).status_code == 409
                    if not readd_during_cancel:
                        again = (await client.post('/v1/room/entries', json={'job_id': entry['job_id']})).json()['entries'][0]
                    assert again['job_id'] == entry['job_id']
                    (tmp_path/'release').touch()
                    (tmp_path/'preparation.pause').unlink(missing_ok=True)
                    await until(lambda: app.state.store.read(entry['job_id'])['status'] == 'completed')
                    result = app.state.store.read(entry['job_id'])['result']
                    assert result['reused_stages'] == ['decode', 'separate']
                    assert (folder/'alignment.bin').read_bytes() == b'complete'
                    assert len(app.state.store.list()) == (2 if cached else 1)
                    if cached:
                        assert app.state.store.read(cached['id'])['status'] == 'completed'
                finally:
                    release_cancel.set()
                    try:
                        os.killpg(parent, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    asyncio.run(scenario())


@pytest.mark.parametrize('hint,initial,resolved', [('en','English','English'), ('zh','Chinese','Cantonese')])
def test_library_lists_queued_and_running_songs_and_reuses_their_jobs(tmp_path, hint, initial, resolved):
    app = create_app(tmp_path, version='test')
    store, room, worker = app.state.store, app.state.room, app.state.worker
    job, _ = store.create(JobRequest(url=URL, language=hint), 'test')
    worker.queue.put_nowait(job['id'])
    room.add(AddSong(job_id=job['id']))
    snapshot = room.add(AddSong(job_id=job['id']))
    song = next(s for s in snapshot['library'] if s['url'] == URL)
    assert song['status'] == 'queued' and song['language'] == initial
    assert song['job_id'] == job['id'] and worker.queue.qsize() == 1
    assert len(store.list()) == 1 and len(snapshot['entries']) == 2
    counters = {'download': dict(completed=50, total=100, unit='bytes')}
    store.update(job['id'], status='running', active_stages=['download'], stage_progress=counters,
                 song_info=dict(title='Video title', artist='Artist', language=resolved))
    song = next(s for s in room.library() if s['url'] == URL)
    assert song['title'] == 'Video title' and song['stage_progress'] == counters
    assert song['status'] == 'running'  # Reusable job ID does not mean ready to play.
    assert song['language'] == resolved and room.snapshot()['entries'][0]['language'] == resolved
    room.add(AddSong(job_id=song['job_id']))
    assert len(store.list()) == 1


def test_library_keeps_one_card_across_failed_jobs_and_retries(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    job, _ = store.create(JobRequest(url=URL, title='Song'), 'test')
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    store.update(job['id'], status='failed', song_info=dict(language='Japanese'),
                 error=dict(code='test', message='Failed'))
    song = next(s for s in room.library() if s['url'] == URL)
    assert song['status'] == 'failed' and song['language'] == 'Japanese'
    assert song['job_id'] is None
    snapshot = room.command(RoomCommand(action='retry', entry_id=first['id']))
    songs = [s for s in snapshot['library'] if s['url'] == URL]
    assert len(songs) == 1 and songs[0]['status'] == 'queued'
    assert songs[0]['job_id'] == job['id'] and snapshot['entries'][0]['id'] == first['id']
    assert store.read(job['id'])['attempt_count'] == 2


def test_retry_without_checkpoint_finishes_original_job_despite_a_newer_attempt(tmp_path):
    failures = []
    def worker(folder):
        if failures:
            return fake_worker(folder)
        failures.append(folder.name)
        return [sys.executable, '-c', "import sys; sys.exit(1)"]
    app = create_app(tmp_path, worker_command=worker, version='test')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][0]
        assert finished(client, entry['job_id'])['status'] == 'failed'
        rival, _ = app.state.store.create(JobRequest(url=URL, force=True), 'test')
        after = client.post('/v1/room/commands', json={'action':'retry', 'entry_id':entry['id']}).json()['entries'][0]
        assert (after['id'], after['job_id']) == (entry['id'], entry['job_id'])
        job = finished(client, entry['job_id'])
        assert job['status'] == 'completed' and job['attempt_count'] == 2
        assert app.state.store.read(rival['id'])['status'] == 'queued'
        for artifact in job['result']['artifacts'].values():
            assert hashlib.sha256(client.get(artifact['url']).content).hexdigest() == artifact['sha256']


def test_browser_remove_is_idempotent_and_only_sung_entries_enter_history(tmp_path):
    app = create_app(tmp_path, version='test')
    client = TestClient(app)
    waiting = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][0]
    ready = ready_job(app.state.store, 'https://youtu.be/00000000002')
    singing = client.post('/v1/room/entries', json={'job_id':ready['id']}).json()['entries'][1]
    for entry, started in ((waiting, None), (singing, 100.)):
        command = {'action':'remove','entry_id':entry['id'],'playback_started_at':started}
        first = client.post('/v1/room/commands', json=command).json()
        again = client.post('/v1/room/commands', json=command).json()
        assert again['revision'] == first['revision'] and again['history'] == first['history']
        assert [song['id'] for song in first['history']] == ([] if started is None else [singing['id']])
    assert again['entries'] == []


def test_manual_retry_resumes_waiting_job_but_preserves_provider_backoff(tmp_path):
    from karaoke.job_schema import download_backoff
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    job, _ = store.create(JobRequest(url=URL), 'test')
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    retry = dict(count=2, next_retry_at=time.time()+60, history=[dict(error='temporary')])
    store.update(job['id'], download_retry=retry, error=dict(code='download_failed'))
    download_backoff(URL, tmp_path/'stage-locks', until=time.time()+40)
    with pytest.raises(HTTPException) as error:
        room.command(RoomCommand(action='retry', entry_id=entry['id']))
    assert error.value.status_code == 429
    assert store.read(job['id'])['download_retry'] == retry
    for p in (tmp_path/'stage-locks').glob('*.retry.json'): p.unlink()
    snapshot = room.command(RoomCommand(action='retry', entry_id=entry['id']))
    assert snapshot['entries'][0]['job_id'] == job['id']
    assert snapshot['entries'][0]['next_retry_at'] is None
    assert store.read(job['id'])['download_retry']['history'] == retry['history']
    with pytest.raises(HTTPException):
        room.command(RoomCommand(action='retry', entry_id=entry['id']))
    assert len(store.list(-1)) == 1


def test_reprepare_keeps_old_media_and_retired_finish_cannot_remove_new_attempt(tmp_path):
    from karaoke.job_schema import download_backoff
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    old = ready_job(store)
    folder = tmp_path/old['id']
    (folder/'download').mkdir()
    (folder/'download/source.mkv').write_bytes(b'original source')
    (folder/'source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b'original source').hexdigest())))
    (folder/'download/source.info.json').write_text('{}')
    entry = room.add(AddSong(job_id=old['id']))['entries'][0]
    download_backoff(URL, tmp_path/'stage-locks', until=time.time()+60)
    store.cache.acquire(old['id'])
    try:
        snapshot = room.command(RoomCommand(action='reprepare', entry_id=entry['id']))
        fresh = snapshot['entries'][0]
        assert fresh['id'] != entry['id'] and fresh['job_id'] != old['id']
        assert fresh['status'] == 'queued'
        assert store.read(old['id'])['status'] == 'completed'
        assert (folder/'mv').read_bytes() == b'media'
        assert (tmp_path/fresh['job_id']/'download/source.mkv').read_bytes() == b'original source'
        assert room.command(RoomCommand(action='finish', entry_id=entry['id']))['entries'] == snapshot['entries']
        with pytest.raises(HTTPException):
            room.command(RoomCommand(action='reprepare', entry_id=entry['id']))
        assert len(store.list(-1)) == 2
    finally:
        store.cache.release(old['id'])


@pytest.mark.parametrize('new_status', ['failed', 'source_ready', 'audio_ready', 'queued', 'running'])
@pytest.mark.parametrize('video_evicted', [False, True])
def test_library_keeps_existing_media_after_playlist_removal_and_new_attempt(tmp_path, new_status, video_evicted):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    old = cache_fixture(store, '1', restorable=True)
    entry = room.add(AddSong(job_id=old['id']))['entries'][0]
    room.command(RoomCommand(action='remove', entry_id=entry['id']))
    folder = tmp_path/old['id']
    assert (folder/'mv.mp4').exists()
    if video_evicted:
        (folder/'mv.mp4').unlink()
        store.update(old['id'], status='video_evicted', restore_video=True)
    newer, _ = store.create(JobRequest.model_validate({**old['input'], 'force': True}), 'new-version')
    store.update(newer['id'], status=new_status)
    cards = [s for s in room.library() if s['url'] == old['input']['url']]
    assert len(cards) == 1
    assert cards[0]['job_id'] == old['id']
    assert cards[0]['status'] == ('video_evicted' if video_evicted else 'completed')
    assert room.read()['entries'] == []
    added = room.add(AddSong(job_id=cards[0]['job_id']))['entries'][0]
    assert added['job_id'] == old['id']
    assert (folder/'instrumental').read_bytes() == b'keep me'


def test_library_does_not_prefer_an_incomplete_old_result(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    old = cache_fixture(store, '1', restorable=True)
    (tmp_path/old['id']/'instrumental').unlink()
    store.update(old['id'], status='video_evicted', restore_video=True)
    newer, _ = store.create(JobRequest.model_validate({**old['input'], 'force': True}), 'new-version')
    card = next(s for s in room.library() if s['url'] == old['input']['url'])
    assert card['job_id'] == newer['id'] and card['status'] == 'queued'


def test_sse_initial_snapshot_push_coalescing_and_disconnect(monkeypatch):
    async def check():
        updates = Updates()
        data = {'revision':0}
        stream = updates.events(lambda: ('event: room\ndata: ' + json.dumps(data) + '\n\n',))
        assert '"revision": 0' in await anext(stream)
        timeout = asyncio.timeout
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, 'timeout', lambda _: timeout(.001))
            # Heartbeats reach EventSource listeners without re-rendering a
            # whole room or confusing a ping with an empty playlist.
            heartbeat = await anext(stream)
            assert heartbeat.startswith('event: ping\n')
            assert json.loads(heartbeat.split('data: ', 1)[1]) == {}
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        data['revision'] = 1
        updates.publish()
        assert '"revision": 1' in await asyncio.wait_for(waiting,1)
        for n in range(2,50):
            data['revision'] = n
            updates.publish()
        assert '"revision": 49' in await anext(stream)
        await stream.aclose()
        assert not updates.subscribers
    asyncio.run(check())


def test_sse_shares_snapshots_and_recovers_a_slow_viewer(tmp_path, monkeypatch):
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    calls = 0
    snapshot = room.snapshot

    def counted():
        nonlocal calls
        calls += 1
        return snapshot()

    monkeypatch.setattr(room, 'snapshot', counted)

    async def read(stream):
        text = (await anext(stream) + await anext(stream)).decode()
        return text.splitlines()[0], json.loads(text.split('data: ', 1)[1])

    async def check():
        viewers = [room.events() for _ in range(20)]
        try:
            initial = [await read(stream) for stream in viewers]
            assert calls == 1  # New subscribers do not each scan SQLite/the library.
            assert all(kind == 'event: room' for kind, _ in initial)
            store.update(job['id'], result={**job['result'], 'title':'Updated'})
            kind, first = await read(viewers[0])
            assert kind == 'event: room_patch'
            assert first['base'] == initial[0][1]['sequence']
            assert first['collections']['entries']['upsert'][0]['title'] == 'Updated'
            assert 'history' not in first['collections']
            assert calls == 2
            # This second update is consumed by one viewer but skipped by the others.
            store.update(job['id'], status='failed', error={'code':'test', 'message':'failed'})
            kind, next_update = await read(viewers[0])
            assert kind == 'event: room_patch'
            assert next_update['collections']['entries']['upsert'][0]['status'] == 'failed'
            assert next_update['collections']['entries']['upsert'][0]['id'] == entry['id']
            for viewer in viewers[1:]:
                kind, caught_up = await read(viewer)
                assert kind == 'event: room'
                assert caught_up['entries'][0]['status'] == 'failed'
                assert caught_up['sequence'] == next_update['sequence']
            assert calls == 3
        finally:
            for viewer in viewers:
                await viewer.aclose()
        assert not room.updates.subscribers
    asyncio.run(check())


def test_sse_preserves_reorder_removal_and_private_playback(tmp_path):
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    second = room.add(AddSong(job_id=job['id']))['entries'][-1]

    async def read(stream):
        return json.loads((await anext(stream) + await anext(stream)).decode().split('data: ', 1)[1])

    async def check():
        stream = room.events(job['id'], first['id'])
        try:
            initial = await read(stream)
            assert initial['playback_entry']['job_id'] == job['id']
            assert initial['playing_entry_ids'] == [first['id']]
            room.command(RoomCommand(action='remove', entry_id=first['id']))
            update = await read(stream)
            assert update['collections']['entries']['remove'] == [first['id']]
            assert update['collections']['entries']['order'] == [second['id']]
            assert update['playback_entry']['status'] == 'completed'
            assert store.cache.leases[job['id']] == 1
            room.command(RoomCommand(action='remove', entry_id=second['id']))
            update = await read(stream)
            assert update['collections']['entries']['order'] == []
            assert update['playback_entry']['job_id'] == job['id']
        finally:
            await stream.aclose()
        assert not store.cache.leases and not room.playing_connections
    asyncio.run(check())


def test_progress_is_received_before_worker_exits(tmp_path):
    info = dict(title='New video title', artist='Singer', duration_seconds=123)
    counters = {'separate_lead': dict(completed=3, total=10, unit='chunks')}
    def event_worker(folder):
        return [sys.executable,'-u','-c',
                'import sys,time;print("KARAOKE_PROGRESS "+sys.argv[1],flush=True);'
                'print("KARAOKE_PROGRESS "+sys.argv[2],flush=True);time.sleep(30)',
                json.dumps(dict(song_info=info)), json.dumps(dict(active_stages=['transcribe', 'separate_lead'],
                    stage_progress=counters, timings={}, song_info={'language': 'English'}))]
    with TestClient(create_app(tmp_path, worker_command=event_worker, version='test')) as client:
        entry = client.post('/v1/room/entries',json={'url':URL}).json()['entries'][0]
        end = time.monotonic()+3
        while time.monotonic() < end:
            state = client.get('/v1/jobs/'+entry['job_id']).json()
            if state.get('stage_progress') == counters:
                break
            time.sleep(.01)
        assert state['active_stages'] == ['transcribe', 'separate_lead'] and state['status'] == 'running'
        # Learning the language in a later stage must not erase earlier metadata.
        assert state['song_info'] == {**info, 'language': 'English'} and state['result'] is None
        assert state['input']['title'] is None and state['input']['artist'] is None
        visible = client.get('/v1/room').json()['entries'][0]
        assert (visible['title'], visible['artist'], visible['duration']) == ('New video title', 'Singer', 123)
        assert visible['stage_progress'] == counters


def test_song_info_updates_push_to_room_and_preserve_user_overrides(tmp_path):
    async def check():
        app = create_app(tmp_path, version='test')
        room, store = app.state.room, app.state.store
        job, _ = store.create(JobRequest(url=URL), 'test')
        state = room.read()
        state['entries'] = [dict(id='entry', job_id=job['id'])]
        room.save(state)
        stream = room.updates.events(lambda: ('event: room\ndata: ' + json.dumps(room.snapshot()) + '\n\n',))

        def entry(event):
            return json.loads(event.split('data: ', 1)[1])['entries'][0]

        queued = entry(await anext(stream))
        assert queued['title'] == '新点的歌' and queued['preparation_started_at'] is None
        started_at = '2026-10-05T22:00:00+00:00'
        thumbnail = 'https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg'
        store.update(job['id'], status='running', started_at=started_at, song_info=dict(thumbnail=thumbnail))
        preparing = entry(await anext(stream))
        assert preparing['preparation_started_at'] == started_at and preparing['poster'] == thumbnail
        for done, title in enumerate(('Video title', 'Resolved song title'), 1):
            counters = {'separate_music': dict(completed=done, total=4, unit='chunks')}
            store.update(job['id'], status='running', song_info=dict(title=title, artist='Singer', language='English'),
                         active_stages=['separate_music'], stage_progress=counters)
            event = await asyncio.wait_for(anext(stream), 1)
            visible = entry(event)
            assert visible['title'] == title and visible['stage_progress'] == counters
            assert visible['preparation_started_at'] == started_at
            assert visible['poster'] == thumbnail
            song = next(s for s in json.loads(event.split('data: ', 1)[1])['library'] if s['url'] == URL)
            assert song['title'] == title and song['language'] == 'English'
            assert song['stage_progress'] == counters
            assert song['preparation_started_at'] == started_at
            assert song['poster'] == thumbnail
        # Reconnecting/reloading reads the persisted start, not the last progress update.
        assert room.snapshot()['entries'][0]['preparation_started_at'] == started_at
        store.update(job['id'], status='failed', error=dict(code='test', message='Test failure'))
        visible = entry(await anext(stream))
        assert visible['title'] == 'Resolved song title' and visible['stage_progress'] == {}
        assert visible['preparation_started_at'] is None
        assert visible['poster'] == thumbnail
        # A queued retry must not inherit the previous attempt's clock.
        store.update(job['id'], status='queued')
        assert entry(await anext(stream))['preparation_started_at'] is None
        retry_started_at = '2026-10-05T22:10:00+00:00'
        store.update(job['id'], status='running', started_at=retry_started_at)
        assert entry(await anext(stream))['preparation_started_at'] == retry_started_at
        store.update(job['id'], status='completed', result=dict(title='Final title', artist='Final singer'))
        completed = entry(await anext(stream))
        assert completed['title'] == 'Final title' and completed['preparation_started_at'] is None
        await stream.aclose()

        manual, _ = store.create(JobRequest(url=URL, title='My title', artist='My singer'), 'test')
        store.update(manual['id'], song_info=dict(title='Detected title', artist='Detected singer'),
                     result=dict(title='Final title', artist='Final singer'))
        visible = room.describe(dict(id='manual', job_id=manual['id']))
        assert (visible['title'], visible['artist']) == ('My title', 'My singer')
    asyncio.run(check())


def test_search_title_is_immediate_but_does_not_override_metadata_or_cache(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'url':URL, 'source_title':'Video title (Official MV)'}).json()['entries'][0]
        assert entry['title'] == 'Video title (Official MV)'
        job = app.state.store.read(entry['job_id'])
        assert job['input']['title'] is None
        assert finished(client, job['id'])['status'] == 'completed'
        for partial in ({}, {'language': 'English'}, {'title': '', 'artist': None}, None):
            app.state.store.update(job['id'], song_info=partial)
            room = client.get('/v1/room').json()
            assert room['entries'][0]['title'] == 'Video title (Official MV)'
            assert room['library'][0]['title'] == 'Video title (Official MV)'
        app.state.store.update(job['id'], song_info={'title':'Clean song title', 'artist':'Singer'})
        room = client.get('/v1/room').json()
        assert room['entries'][0]['title'] == 'Clean song title'
        assert room['library'][0]['title'] == 'Clean song title'
        assert room['library'][0]['language'] == 'English'
        again = client.post('/v1/room/entries', json={'url':URL, 'source_title':'A different search title'}).json()
        assert again['entries'][-1]['job_id'] == job['id']
        assert again['entries'][-1]['title'] == 'Clean song title'
        assert len(app.state.store.list()) == 1


@pytest.mark.parametrize('thumbnail', [
    'https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg',
    'https://lh3.googleusercontent.com/cover',
    'http://i.ytimg.com/cover', 'https://ytimg.com.attacker.test/cover', 'https://localhost/cover',
])
def test_search_cover_is_immediate_persistent_and_replaced_by_local_artwork(tmp_path, thumbnail):
    app = create_app(tmp_path, version='test')
    (tmp_path/'preparation.pause').touch()
    expected = thumbnail if thumbnail.startswith(('https://i.ytimg.com/', 'https://lh3.googleusercontent.com/')) else None
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json=dict(url=URL, source_thumbnail=thumbnail)).json()['entries'][0]
        assert entry['status'] == 'queued' and entry['poster'] == expected
        store = app.state.store
        store.update(entry['job_id'], song_info=dict(title='Known song', thumbnail=None))
        assert client.get('/v1/room').json()['entries'][0]['poster'] == expected
    # Refresh/restart and a new attempt keep known artwork without fetching metadata again.
    app = create_app(tmp_path, version='new')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json=dict(url=URL)).json()['entries'][-1]
        assert entry['poster'] == expected and entry['title'] == 'Known song'
        (tmp_path/entry['job_id']/'poster.jpg').write_bytes(b'cover')
        app.state.store.update(entry['job_id'], status='completed', result=dict(artifacts=dict(
            poster=dict(path='poster.jpg', bytes=5, sha256='local'))))
        assert client.get('/v1/room').json()['entries'][-1]['poster'] == f"/v1/jobs/{entry['job_id']}/artifacts/poster?v=local"


@pytest.mark.parametrize('recreate', ['force', 'version', 'evicted'])
def test_recreated_song_keeps_known_metadata_before_any_worker_runs(tmp_path, recreate):
    app = create_app(tmp_path, version='new')
    (tmp_path/'preparation.pause').touch()
    old = ready_job(app.state.store)
    if recreate == 'evicted':
        import shutil
        shutil.rmtree(tmp_path/old['id'])
        app.state.store.update(old['id'], status='evicted')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json=dict(
            url=URL, force=recreate == 'force', source_title='Less useful search title')).json()['entries'][0]
        assert (entry['job_id'] == old['id']) == (recreate == 'version')
        assert entry['status'] == ('completed' if recreate == 'version' else 'queued')
        assert (entry['title'], entry['artist'], entry['duration']) == ('Test song', 'Singer', 10)
        if recreate == 'version':
            assert app.state.store.read(entry['job_id'])['pipeline_version'] == old['pipeline_version']
            return
        from karaoke.pipeline import Pipeline
        pipeline = Pipeline(dict(url=URL), tmp_path/entry['job_id'])
        assert pipeline.song_info['title'] == 'Test song'
        other = client.post('/v1/room/entries', json=dict(url='https://www.youtube.com/watch?v=abcdefghijk')).json()['entries'][-1]
        assert other['title'] == '新点的歌'
