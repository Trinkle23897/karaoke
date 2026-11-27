"""Migrate completed playback caches while the API worker is stopped.

No models or lyrics are recomputed. Old media survives until the new result is
fully checked and published. Job identities, versions and queue are unchanged.
"""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from karaoke.api import Store
from karaoke.exact_stems import compact_playback_audio, sha
from karaoke.job_schema import playback_stems
from karaoke.playback_video import mux_playback_video
from karaoke.tools.soak import verify_stems


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def video_hash(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
        '-map', '0:v:0', '-f', 'hash', '-hash', 'sha256', '-'], timeout=300).decode().strip()


def migrate(store, job):
    if job['status'] not in ('completed', 'video_evicted'):
        return dict(id=job['id'], status='skipped')
    folder = store.cache.folder(job['id'])
    original = job['result']
    if original.get('validation_scope') == 'audio_and_lyrics':
        return dict(id=job['id'], status='strict_audio_unchanged')
    before = store.cache.size(folder)
    required = {*playback_stems(original['artifacts']), 'subtitles'}
    if job['status'] == 'completed':
        required.add('mv')
    for name in required:
        artifact = original['artifacts'][name]
        path = folder/artifact['path']
        if (not path.resolve().is_relative_to(folder) or path.is_symlink()
                or path.stat().st_size != artifact['bytes'] or sha(path) != artifact['sha256']):
            raise ValueError('Original artifact failed verification: '+name)
    updated = compact_playback_audio(folder, original, combine=True)
    verify_stems({k: folder/a['path'] for k, a in updated['artifacts'].items()},
                 updated['duration_seconds'], storage=updated['audio_storage'])
    evidence = dict(id=job['id'], before_bytes=before, original_artifacts=original['artifacts'],
                    audio_storage=updated['audio_storage'])
    if job['status'] == 'completed' and original.get('video_audio') != 'silent_aac':
        old = folder/original['artifacts']['mv']['path']
        target = folder/'mv-video.mp4'
        temporary = folder/'mv-video.tmp.mp4'
        try:
            mux_playback_video(old, temporary)
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(temporary),
                '-f', 'null', '-'], check=True, timeout=300)
            evidence['video_before'] = video_hash(old)
            evidence['video_after'] = video_hash(temporary)
            if evidence['video_before'] != evidence['video_after']:
                raise ValueError('Video frames changed during storage migration')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        updated['artifacts']['mv'] = dict(original['artifacts']['mv'], path=target.name,
                                          bytes=target.stat().st_size, sha256=sha(target))
        updated.pop('video_only', None)
        updated['video_audio'] = 'silent_aac'
    save(folder/'storage-migration.json', evidence)
    # Disk and DB publication both precede compaction. An interrupted migration
    # can be rerun from the DB's still-valid old media until this commit succeeds.
    save(folder/'result.json', updated)
    fresh = store.update(job['id'], result=updated)
    store.cache.compact(fresh)
    return dict(id=job['id'], status='migrated', before_bytes=before,
                after_bytes=store.cache.size(folder), artifacts=updated['artifacts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--job', action='append')
    args = parser.parse_args()
    with (args.data/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(args.data)
        jobs = store.list(-1)
        if any(j['status'] == 'running' for j in jobs):
            raise RuntimeError('A job is still running; do not migrate active media')
        before = store.cache.total_size(store.cache.folder(j['id']) for j in jobs)
        results = [migrate(store, job) for job in jobs if not args.job or job['id'] in args.job]
        after = store.cache.total_size(store.cache.folder(j['id']) for j in jobs)
        print(json.dumps(dict(before_bytes=before, after_bytes=after, jobs=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
