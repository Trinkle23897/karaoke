"""Use publisher-supplied subtitle text from the same recording as lyric evidence."""
import hashlib
import inspect
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from .anchors_from_asr import text_units
from .job_schema import retry_after_seconds, song_language


def manual_track(metadata, language):
    if language is None:
        return None
    for tag, tracks in sorted((metadata.get('subtitles') or {}).items()):
        if song_language(tag) != language:
            continue
        for track in tracks:
            url = track.get('url', '')
            parsed = urlsplit(url)
            query = parse_qs(parsed.query)
            if (track.get('ext') == 'json3' and parsed.scheme == 'https' and
                    parsed.netloc in ('www.youtube.com', 'youtube.com') and
                    parsed.path == '/api/timedtext' and query.get('v') == [metadata.get('id')] and
                    query.get('fmt') == ['json3'] and song_language((query.get('lang') or [None])[0]) == language and
                    not query.get('kind') and not query.get('tlang')):
                return dict(language=tag, url=url)
    return None


def read_document(url, *, headers=None, timeout=20):
    with httpx.stream('GET', url, timeout=timeout, follow_redirects=False,
                      headers={'User-Agent': 'Mozilla/5.0', **(headers or {})}) as response:
        response.raise_for_status()
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > 2*1024*1024:
                raise ValueError('Subtitle response exceeds 2 MiB')
    return json.loads(body)


def fetch_manual_captions(metadata_path, language, cache):
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    track = manual_track(metadata, language)
    if track is None:
        return None
    path = cache.root/'source-captions.json'
    name = 'fetch_source_captions'
    try:
        old = json.loads(path.read_text())
        if old.get('error') and old.get('retry_at') is not None and old['retry_at'] <= time.time():
            (cache.directory/(name+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(name, inputs=[metadata_path], outputs=[path], config=dict(
            track=track, implementation=inspect.getsource(read_document))) as reused:
        if not reused:
            packet = dict(video_id=metadata['id'], language=track['language'])
            try:
                packet['response'] = read_document(track['url'])
            except (httpx.HTTPError, ValueError) as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                transient = isinstance(error, httpx.TransportError) or status == 429 or (status or 0) >= 500
                packet.update(error=dict(type=type(error).__name__, status=status),
                    retry_at=time.time()+max(300, retry_after_seconds(error) or 0) if transient else None)
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    return path


def caption_events(packet, language, duration):
    """Read source display events; their timestamps are not acoustic onsets."""
    if packet.get('error'):
        return None, dict(status='unavailable', error=packet['error'], retry_at=packet.get('retry_at'))
    if song_language(packet.get('language')) != language:
        return None, dict(status='wrong_caption_language')
    events = []
    try:
        for event in packet['response']['events']:
            text = ''.join(s.get('utf8', '') for s in event.get('segs', []))
            if not text.strip():
                continue
            start = event['tStartMs']/1000
            end = start+event['dDurationMs']/1000
            if (event.get('aAppend') or not all(math.isfinite(t) for t in (start, end)) or
                    not 0 <= start < end <= duration+1):
                raise ValueError('Unsupported or out-of-range caption event')
            if re.search(r'\[[^\]\n]+\]', text):
                # Closed captions may label speakers or describe sounds. Do not
                # sing those labels or guess which bracketed words to delete;
                # retain the audio transcript instead of this optional source.
                return None, dict(status='unsupported_caption_annotations')
            events.append(dict(text=text, start=start, end=end))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None, dict(status='invalid_caption_document')
    return events, None


def caption_lines(events):
    """Keep the publisher's lyric breaks; the renderer handles display width."""
    from .song_sources import join_punctuation_lines

    return join_punctuation_lines(line for event in events for line in event['text'].splitlines())


def caption_candidate(packet, transcription, language, duration):
    """Check source text against audio; caption display times are never word times."""
    from rapidfuzz.distance import Levenshtein

    events, error = caption_events(packet, language, duration)
    if events is None:
        return None, error
    try:
        lines = caption_lines(events)
        if not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            raise ValueError('Invalid lyric length')
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None, dict(status='invalid_caption_document')
    expected = text_units(''.join(lines), language)
    observed = [unit for word in transcription.get('words', []) for unit in text_units(word['word'], language)]
    matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
    support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
    accepted = bool(lines) and min(support, coverage) >= .55
    report = dict(status='accepted' if accepted else 'insufficient_audio_support',
        text_support=support, audio_word_coverage=coverage, language=packet['language'],
        video_id=packet['video_id'], response_sha256=hashlib.sha256(
            json.dumps(packet['response'], sort_keys=True).encode()).hexdigest())
    return lines if accepted else None, report


def prepare_caption_lyrics(request, metadata, duration, cache, model):
    """Use manual same-recording text only after every omission is confirmed as speech."""
    from .lyric_lookup import text_language
    from .song_sources import SongText, extract_song_metadata, video_song_info
    from .vocal_review import review_gaps, speech_intervals

    song = video_song_info(metadata)
    language = request.get('language') or song.get('language')
    if request.get('lyrics') or not language or not manual_track(metadata, language):
        return None
    path = fetch_manual_captions(cache.root/'download/source.info.json', language, cache)
    if path is None:
        return None
    packet = json.loads(path.read_text())
    if packet.get('video_id') != metadata.get('id'):
        return None
    events, error = caption_events(packet, language, duration)
    report_path = cache.root/'caption-lyrics-review.json'

    def reject(status, **details):
        report_path.write_text(json.dumps(dict(status=status, **details), ensure_ascii=False)+'\n')
        return None

    if events is None:
        return reject(**error)
    if not events or len(events) > 1000 or sum(len(e['text']) for e in events) > 50000:
        return reject('invalid_caption_document')
    evidence = {key:metadata.get(key) for key in ('title','track','artist','artists','uploader')}
    evidence['caption_lines'] = [dict(number=i+1, text=e['text']) for i,e in enumerate(events)]
    selection = cache.root/'caption-selection.json'
    with cache.stage('select_caption_lyrics', inputs=[path], outputs=[selection],
                     config=dict(evidence=evidence, model=model),
                     code=[Path(__file__), Path(__file__).with_name('song_sources.py')]) as reused:
        if not reused:
            selection.write_text(json.dumps(extract_song_metadata(evidence, model), ensure_ascii=False)+'\n')
    result = json.loads(selection.read_text()).get('parsed')
    if result is None:
        return reject('invalid_selection')
    parsed = SongText.model_validate(result)
    chosen = parsed.lyric_line_numbers
    if not chosen or chosen != sorted(set(chosen)) or any(not 1 <= i <= len(events) for i in chosen):
        return reject('invalid_selection')
    selected, omitted, groups = set(chosen), [], []
    for number, event in enumerate(events, 1):
        if number not in selected:
            omitted.append(event)
        elif omitted:
            groups.append(omitted)
            omitted = []
    if omitted:
        groups.append(omitted)
    ranges = [dict(start=min(e['start'] for e in g), end=max(e['end'] for e in g)) for g in groups]
    picked = [events[i-1] for i in chosen]
    if len(ranges) > 8 or any(r['end'] > duration or r['end']-r['start'] > 90 or
            any(e['start'] < r['end'] and r['start'] < e['end'] for e in picked) for r in ranges):
        return reject('unreviewable_omission', omitted_ranges=ranges)
    reviews = review_gaps(cache.root, ranges, [e for g in groups for e in g]) if ranges else []
    accepted = speech_intervals(cache.root, reviews)
    report = dict(status='accepted' if accepted == [(r['start'],r['end']) for r in ranges] else 'unverified_omission',
        video_id=packet['video_id'], language=language, lyric_line_numbers=chosen, omitted_ranges=ranges,
        response_sha256=hashlib.sha256(json.dumps(packet['response'], sort_keys=True).encode()).hexdigest(),
        provider_timestamps_used_for_alignment=False)
    report_path.write_text(json.dumps(dict(report, reviews=reviews), ensure_ascii=False)+'\n')
    if report['status'] != 'accepted':
        return None
    lines = caption_lines(picked)
    detected = text_language(''.join(lines))
    if (not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines) or
            detected != language and not (language == 'Cantonese' and detected == 'Chinese')):
        return reject('invalid_lyric_text', selection=report)
    return dict(song, id='song', title=request.get('title') or metadata.get('track') or parsed.title or song['title'],
        artist=request.get('artist') or song['artist'] or parsed.artist, language=language, lines=lines,
        source_url=request['url'], provenance=dict(kind='source_manual_subtitles', route='lyrics_first',
            needs_review=True, independent_asr_coverage=None, source_captions=report))
