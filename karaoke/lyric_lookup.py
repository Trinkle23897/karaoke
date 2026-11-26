"""Optional independent lyric text; provider timestamps never become onsets."""
import hashlib
import html
import json
import math
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache, partial
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import regex
import requests
from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

from .anchors_from_asr import text_units
from .job_schema import (
    language_code,
    provider_error,
    retry_after_seconds,
    retry_fields,
    song_language,
)
from .song_sources import (
    SongText,
    extract_song_metadata,
    find_verified_lyrics,
    join_punctuation_lines,
    video_song_info,
)
from .source_captions import read_document

_traditional, _simplified = OpenCC('s2t'), OpenCC('tw2s')
_versioned_title = re.compile(r'(.+?)\s*\(([^()]*(?:\b(?:version|ver\.?)|版本?)|[国國粤粵])\)\s*$', re.I)
_soundtrack_role = r'(?:人物)?(?:主题[歌曲]|片[头尾]曲|插曲)'
_soundtrack_credit = re.compile(r'(?:大型)?(?:纪录片|电影|电视剧|动画片|网络剧|网剧|剧集)?'
    r'\s*《[^《》]+》\s*'+_soundtrack_role+r'|from\s+["“][^"“”]+["”]', re.I)
_credit_separators = r'\s*[&,/×+、;]\s*|\s+·\s+|\s+\b(?:feat\.?|featuring|ft\.?|with)\s+'
_featured_title = re.compile(
    r'(.+?)\s*(?:[\[(](?:(?:feat\.?|featuring|ft\.?|with)\s+|合作音乐人\s*[:：]\s*)([^\])]+)[\])]|'
    r'\s+(?:feat\.?|featuring|ft\.?)\s+(.+))', re.I)
_section_label = re.compile(
    r'\[\s*(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|instrumental|interlude|refrain|(?:(?:piano|guitar)\s+)?solo|'
    r'主歌|副歌|前副歌|後副歌|后副歌|前奏|間奏|间奏|尾奏|橋段|桥段|'
    r'couplet|pr[ée][- ]?refrain|post[- ]?refrain|pont|ch(?:œ|oe)ur|strophe|'
    r'المقطع(?:\s+\S+)?|(?:بعد\s+)?اللازمة)'
    r'(?:\s*\d+)?(?:\s*[:：][^\]\n]+)?\s*\]', re.I)


def canonical_script(text):
    """Compare mixed Han scripts consistently; never rewrite the published body."""
    return _simplified.convert(_traditional.convert(unicodedata.normalize('NFKC', text)))


def comparison_key(text):
    return ''.join(c for c in canonical_script(text).casefold() if c.isalnum())


class LyricRequestError(httpx.HTTPError):
    def __init__(self, report):
        super().__init__(report.get('error', 'Lyric provider is unavailable'))
        self.report = report


def lyric_request_error(error, minimum_delay=30):
    if isinstance(error, LyricRequestError):
        return error.report
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    transient = isinstance(error, (httpx.TransportError, requests.ConnectionError, requests.Timeout)) or status == 429 or (status or 0) >= 500
    return dict(status='unavailable', error=type(error).__name__, http_status=status,
                retry_at=time.time()+max(minimum_delay, retry_after_seconds(error) or 0) if transient else None)


def lyric_retry_due(report, provider, cache):
    if report.get('retry_at') is not None and report['retry_at'] <= time.time():
        return True
    if report.get('http_status') == 401:
        try:
            hold = json.loads((cache.root.parent/'lyric-provider-backoff'/f'{provider}.json').read_text())
        except (OSError, ValueError):
            return True
        return not hold.get('requires_access')
    return False


def cached_lyric_request(provider, identity, cache, request, minimum_delay=30):
    """Cache provider responses independently of duration and lyric selection."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = cache.root/'lyric-responses'/f'{provider}-{key}.json'
    stage = f'lyric_response_{provider}_{key}'
    try:
        previous = json.loads(path.read_text()).get('error') or {}
        if lyric_retry_due(previous, provider, cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__).with_name('source_captions.py')], outputs=[path],
                     config=dict(provider=provider, request=identity, schema_version=1)) as reused:
        if not reused:
            hold = cache.root.parent/'lyric-provider-backoff'/f'{provider}.json'
            try:
                previous = json.loads(hold.read_text())
            except (OSError, ValueError, TypeError):
                previous = {}
            if previous.get('requires_access') or (previous.get('retry_at') or 0) > time.time():
                # A provider-wide hold carries no identity or text from its song.
                raise LyricRequestError({k: previous[k] for k in
                    ('status', 'error', 'http_status', 'retry_at', 'requires_access') if k in previous})
            try:
                packet = dict(data=request())
            except Exception as error:
                report = lyric_request_error(error, minimum_delay)
                packet = dict(error=report)
                if report['http_status'] in (401, 429):
                    import fcntl
                    hold.parent.mkdir(parents=True, exist_ok=True)
                    with hold.with_suffix('.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        try:
                            previous = json.loads(hold.read_text())
                        except (OSError, ValueError):
                            previous = {}
                        pending = dict(report, requires_access=bool(previous.get('requires_access')) or report['http_status'] == 401,
                            retry_at=max(previous.get('retry_at') or 0, report.get('retry_at') or time.time()+30))
                        temporary = hold.with_suffix(f'.{os.getpid()}.{time.time_ns()}.tmp')
                        temporary.write_text(json.dumps(pending)+'\n')
                        temporary.replace(hold)
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    packet = json.loads(path.read_text())
    if 'error' in packet:
        raise LyricRequestError(packet['error'])
    return packet['data']


def names(value, *, title=False):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').strip()
    result = [value] if value else []
    if title:
        # Upload-format labels describe the video, not the musical work.
        # Require a delimiter and retain the original spelling as an alias.
        label = (r'(?:official\s+(?:music\s+video|video|mv|audio|visualizer)|'
                 r'(?:official\s+)?lyrics?\s+video|with\s+lyrics|歌[詞词](?:付き|つき))'
                 r'(?:\s+(?:hd|4k))?')
        clean = re.sub(r'(?:\s*[\[(【「]\s*'+label+r'\s*[\])】」]|\s+'+label+r')\s*$',
                       '', value, flags=re.I)
        if clean and clean != value:
            return result + names(clean, title=True)
    # Split explicit bilingual names before parsing either side's annotations.
    pair = re.fullmatch(r'(.+?)\s+[-–—]\s+(.+)', value)
    if pair:
        native_side = lambda text: any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text)
        if (native_side(pair[1]) != native_side(pair[2]) or
                comparison_key(pair[1]) == comparison_key(pair[2])) and version_tags(pair[1]) == version_tags(pair[2]):
            for part in ((pair[2], pair[1]) if native_side(pair[1]) else (pair[1], pair[2])):
                result.extend(names(part, title=title))
            return list(dict.fromkeys(result))
    if title:
        # Featured-performer credits belong to recording metadata, not the
        # work name. Retain the full spelling alongside its title-only alias.
        featured = _featured_title.fullmatch(value)
        if featured:
            return result + names(featured[1].strip(), title=True)
        edition = _versioned_title.fullmatch(value)
        if edition:
            return result + names(edition[1], title=True)
        annotation = re.fullmatch(r'(.+?)\s*\(([^()]*)\)', value)
        if annotation:
            label = canonical_script(annotation[2]).casefold()
            release = re.fullmatch(r'(?:live|现场)(?:版本?)?(?:[ |/].*)?|cover(?: version)?|'
                r'(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?', label)
            if release or _soundtrack_credit.fullmatch(label) or re.fullmatch(_soundtrack_role, label):
                return result + names(annotation[1].strip(), title=True)
    # Platforms publish original names together with Latin-script aliases.
    # Version labels are not translations: never strip "Live", "Remix", etc.
    native = r'[^\W\d_\x00-\x7f]+(?:[ ,·、&/]+[^\W\d_\x00-\x7f]+)*'
    latin = r'[A-Za-z][\x20-\x7e]*?'
    match = re.fullmatch(rf'({native})(?:\s+(?:[-–—]\s+)?({latin})|\s*\(({latin})\))', value)
    reverse = re.fullmatch(rf'({latin})(?:\s+({native})|\s*\(({native})\))', value)
    if match or reverse:
        if match:
            local, english = match[1], match[2] or match[3]
        else:
            assert reverse is not None
            local, english = reverse[2] or reverse[3], reverse[1]
        if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in local) and not re.search(r'\b(live|remix|acoustic|instrumental|karaoke|remaster\w*)\b', english, re.I) and not re.search(
                r'라이브|리믹스|어쿠스틱|인스트루멘탈|버전', local):
            result.extend((english, local))
    return result


def song_names(song, key):
    """Keep source-confirmed spellings available after choosing a display name."""
    return list(dict.fromkeys(name for value in [song.get(key), *song.get(key+'_aliases', [])]
                              for name in names(value, title=key == 'title')))


def identity(value, *, title=False):
    aliases = [canonical_script(name).casefold() for name in names(value, title=title)]
    if not title:
        if aliases and re.search(r'[&,/×+]|\b(?:feat\.?|featuring|ft\.?|with)\s', aliases[0]):
            aliases = aliases[:1]  # A collaborator is not an alias for the whole credit.
        joined = re.fullmatch(r"([a-z][a-z.'-]*)([\u3040-\u30ff\u3400-\u9fff]{2,})(?:\s+[a-z][a-z .'-]*)?",
                              aliases[0] if aliases else '')
        if joined and not version_tags(joined[1]):
            aliases.append(joined[2])
    return {''.join(c for c in name if c.isalnum()) for name in aliases} - {''}


def artist_credit_matches(credit, artists, *, partial=False):
    """Match whole credited names, including reordered multi-artist credits."""
    expected = identity(credit)
    if not expected:
        return False
    if any(expected & identity(artist) for artist in artists):
        return True
    members = [identity(part) for artist in artists if isinstance(artist, str)
               for part in re.split(_credit_separators, artist, flags=re.I) if part.strip()]
    if partial:
        return any(identity(part) & member for part in re.split(_credit_separators, credit, flags=re.I) for member in members)
    if len(members) < 2 or not all(members):
        return False
    expected |= identity(''.join(re.split(_credit_separators, credit, flags=re.I)))
    expected.update(name for alias in names(credit) for name in identity(alias))

    @lru_cache(None)
    def consume(text, remaining):
        if not remaining:
            return not text
        return any(text.startswith(name) and consume(text[len(name):], remaining[:i]+remaining[i+1:])
                   for i, member in enumerate(remaining) for name in members[member])

    # Use provider-supplied name boundaries: splitting on spaces would break
    # multi-word English names. Every member must account for the whole credit.
    return any(consume(text, tuple(range(len(members)))) for text in expected)


def live_performance(value):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').casefold()
    match = re.search(r'(?:^|[\(\[]|\s[-–—]\s)\s*live\s+(?:from|at|in)\s+([^()\[\]|\n]+)', value)
    if match:
        venue = re.sub(r'^the\s+', '', match[1].strip())
        return ''.join(c for c in venue if c.isalnum()) or None
    return None


def fetch_lyrics(song, metadata_path, cache):
    titles, artists = song_names(song, 'title'), song_names(song, 'artist')
    if not titles or not artists or not metadata_path.is_file():
        return dict(status='missing_identity')
    queries = [dict(track_name=titles[0], artist_name=artists[0])]
    native = dict(track_name=titles[-1], artist_name=artists[-1])
    if native != queries[0]:
        queries.append(native)
    simplified = {key: canonical_script(value) for key, value in native.items()}
    if simplified not in queries:
        queries.append(simplified)
    # Search indexing may omit a combined performer credit. A broader query
    # still goes through the same title/performer/recording checks below.
    title_only = dict(track_name=simplified['track_name'])
    queries.append(title_only)
    path, stage = cache.root/'lyric-lookup.json', 'fetch_lyric_text'
    try:
        old = json.loads(path.read_text())
        if lyric_retry_due(old, 'lrclib', cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[metadata_path], outputs=[path], config=queries,
                     code=[Path(__file__), Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            packet: dict[str, Any] = dict(status='ok', queries=queries, results=[])
            try:
                for query in queries:
                    if query == simplified and simplified != native and packet['results']:
                        break
                    if query == title_only and packet['results']:
                        break
                    url = 'https://lrclib.net/api/search?'+urlencode(query)
                    response = cached_lyric_request('lrclib', url, cache, lambda: read_document(url), 300)
                    if not isinstance(response, list):
                        raise ValueError('Invalid lyric search response')
                    packet['results'].extend({key: row.get(key) for key in (
                        'id', 'trackName', 'artistName', 'albumName', 'duration', 'instrumental', 'plainLyrics')}
                        for row in response if isinstance(row, dict))
            except (httpx.HTTPError, ValueError) as error:
                packet.update(lyric_request_error(error, 300))
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())


def unpadded_duration(audio_path, duration):
    """Measure exact-zero padding in the original PCM, without trimming audio."""
    import numpy as np
    import soundfile as sf

    from .stage_cache import digest
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return None
    first, last, offset = None, 0, 0
    with sf.SoundFile(audio_path) as audio:
        rate, frames = audio.samplerate, audio.frames
        if abs(frames/rate-duration) > 1.:
            return None
        for block in audio.blocks(blocksize=rate*5, dtype='float64', always_2d=True):
            if not np.isfinite(block).all():
                return None
            nonzero = np.flatnonzero(np.any(block != 0, axis=1))
            if nonzero.size:
                if first is None:
                    first = offset+int(nonzero[0])
                last = offset+int(nonzero[-1])+1
            offset += len(block)
    if first is None:
        return None
    return dict(kind='nonzero_pcm', audio_sha256=digest(audio_path),
                recording_duration=frames/rate, nonzero_duration=(last-first)/rate,
                leading_zero_seconds=first/rate, trailing_zero_seconds=(frames-last)/rate)


@lru_cache(maxsize=128)
def text_language(text):
    # Formatting placeholders (including Hangul fillers) are not language
    # evidence. Keep the source text intact; only normalize this probe.
    text = regex.sub(r'\p{Default_Ignorable_Code_Point}', '', text)
    if re.search(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]', text):
        return 'Korean'
    if re.search(r'[\u3040-\u30ff]', text):
        return 'Japanese'
    if re.search(r'[\u3400-\u9fff]', text):
        return 'Chinese'
    if any(c.isalpha() for c in text):
        import langid
        return song_language(langid.classify(text)[0])
    return None


def published_lyric_lines(text, title, artists=(), *, album=None):
    """Keep published phrases intact; remove only explicit credit metadata."""
    instrument = (r'(?:钢琴|键盘|(?:低音)?吉他|贝斯|鼓|打击乐|二胡|小号|'
        r'(?:第[一二三四\d]+)?(?:小提琴|中提琴|大提琴))(?:手|独奏|\d+)?')
    production = (r'(?:音乐|音响|灯光|舞台秀?|乐队|声乐|演唱|歌曲)'
        r'(?:总监|设计|指导|统筹|队长|混音|编辑|发行|营销|策划|导演团队)')
    role = (rf'{instrument}|{production}|歌名|歌曲|(?:中文|英文|改编)?词|曲|词曲|作词|作曲|(?:改编)?编曲(?:协力)?|原唱|演唱|专辑|发行(?:日|公司)|出品人?|音乐制作(?:人|助理)?|(?:定位)?制作人|制作(?:助理|团队)|监制|'
        r'弦乐(?:编写|编曲|制谱|监制|录音)?|(?:人声|配唱|伴唱|主唱|合唱|童声|合声|合音|和声|和音|念白)(?:编写|编曲|编配|制作|录音|录制|编辑)?(?:人|师|指导)?|器乐|'
        r'(?:特别)?鸣谢(?:\s*\([^()]*\))?|'
        r'录音(?:师|助理|(?:工作)?室)?|(?:前置)?混音(?:师|(?:工作)?室)?|母带(?:后期(?:处理|制作))?(?:人|工程师|录音室)?|音频(?:助理|工程师|编辑)|'
        r'(?:song )?title|song name|artist|album|composer|lyricist|producer|production|lyrics(?: by)?|music(?: production| distributed by)?|'
        r'guitar|bass|drums|piano|strings|instrumental|(?:written|composed|arranged|produced|recorded|mixed|mastered) by|audio engineer|assistant engineer|'
        r'乐团|乐队|orchestra|band|arrange(?:ment|r)|(?:backing |lead )?vocals?(?: arrangement)?|(?:recording|mixing|mastering)(?: engineer| studio)?(?: \(vocals\))?|'
        r'production assistant|program(?:ming)?|PGM|[OS]P|O[AC](?:\s*/\s*O[AC])?|publisher|copyright')
    # A translated role is still metadata: "编曲 Arrangement: ...".
    # Require complete recognized labels before the colon, not arbitrary prose.
    label = rf'(?:首席|执行|联合|chief\s+|executive\s+|co[- ])?(?:{role})'
    separator = (r'\s+|\s*[/／]\s*|(?<=[\u3400-\u9fff])(?=[A-Za-z])|'
                 r'(?<=[A-Za-z])(?=[\u3400-\u9fff])')
    credit = re.compile(rf'^{label}(?:(?:{separator}){label})*\s*[：:]', re.I)
    people = [identity(part) for artist in artists if isinstance(artist, str)
              for part in re.split(_credit_separators, artist, flags=re.I) if part.strip()]
    rows, metadata, timestamps = [], [], {}
    for i, line in enumerate(text.splitlines(), 1):
        line = html.unescape(line).strip()
        if re.fullmatch(r'\[(?:ar|ti|al|by|offset|re|ve|id):[^\]]*\]', line, re.I):
            metadata.append(i)
            continue
        # Provider timestamps describe text order/repetitions, never our onsets.
        # Defer repeated LRC rows until they can be ordered with the other rows.
        prefix = re.match(r'^(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?(?:-\d+)?\]\s*)+', line)
        if prefix:
            stamps = re.findall(r'\[(\d{1,3}):(\d{2}(?:[.:]\d{1,3})?)(?:-\d+)?\]', prefix[0])
            timestamps[i] = [60*int(m)+float(s.replace(':', '.')) for m,s in stamps]
            line = line[prefix.end():]
        # Some LRC publishers copy a serialized credit header into each
        # payload. Decode only a recognized prefix followed by one lyric;
        # ordinary escapes and multi-line bodies must remain untouched.
        if r'\n' in line:
            parts = line.split(r'\n')
            cleaned, excluded = published_lyric_lines('\n'.join(parts), title, artists, album=album)
            if len(cleaned) == 1 and excluded == list(range(1, len(parts))):
                line = cleaned[0]
        # An empty attribution row is a cue to the credited performers, not a
        # sung name. Resolve abbreviations only when one credited name fits.
        speaker = re.fullmatch(r'([^:：]+)[:：]\s*', line)
        if speaker:
            labels = identity(speaker[1])
            matches = [person for person in people if any(name.startswith(label) for name in person for label in labels)]
            if (any(labels & person for person in people) or len(matches) == 1 or
                    len(people) > 1 and labels & {'合', '合唱', 'all'}):
                metadata.append(i)
                continue
        # These are publisher section headings, not inferred content. Retain
        # unrecognized brackets and all inline backing-vocal responses.
        quoted_titles = re.findall(r'["“„《「『]([^"“”》」』]+)["“”》」』]', line)
        lyric_header = (line.startswith('[') and line.endswith(']') and
            re.search(r'lyrics?|歌詞|歌词|가사|paroles|songtext', line, re.I) and any(
                identity(name, title=True) & identity(title, title=True) for name in quoted_titles))
        if _section_label.fullmatch(line) or lyric_header:
            metadata.append(i)
            continue
        if line:
            rows.append((i, line))
    is_credit = lambda line: bool(credit.match(canonical_script(line.replace('︰', ':'))) or
                                 re.fullmatch(r'credits?(?:\s+list)?\s*[:：]?', line, re.I))
    def title_artist(line):
        return any(identity(candidate_title, title=True) & identity(title, title=True) and
                   any(artist_credit_matches(name, artists) for name in
                       (candidate_artist, re.split(r'[（(]', candidate_artist)[0]))
                   for match in re.finditer(r'[-–—]', line)
                   for candidate_title, candidate_artist in (
                       (line[:match.start()], line[match.end():]), (line[match.end():], line[:match.start()])))
    first = 0
    while first < len(rows):
        line = rows[first][1]
        # Credits and copied page headers may appear in either order. A title
        # and its matching byline form one header; either alone may be sung.
        if first+1 < len(rows) and identity(line, title=True) & identity(title, title=True):
            byline = re.fullmatch(r'by\s+(.+)', rows[first+1][1], re.I)
            if byline and any(artist_credit_matches(name, artists) for name in
                    (byline[1], re.sub(r'\s*Play$', '', byline[1], flags=re.I))):
                first += 2
                continue
        note = canonical_script(line).strip('()').strip()
        album_header = identity(line) & identity(album) and any(
            title_artist(rows[i][1]) for i in (first-1, first+1) if 0 <= i < len(rows))
        heading_note = first and (_soundtrack_credit.fullmatch(note) or re.fullmatch(
            r'(?:又名|别名|also known as|a\.?k\.?a\.?)\s*[:：].+', note, re.I) or re.fullmatch(
            r'【未经[^【】]*(?:许可|授权)[^【】]*不得[^【】]+】', note))
        if is_credit(line) or title_artist(line) or album_header or heading_note or (identity(line) & identity(title) and
                first+1 < len(rows) and is_credit(rows[first+1][1])):
            first += 1
        else:
            break
    stop = next((i for i in range(first, len(rows)) if is_credit(rows[i][1])), len(rows))
    body = rows[first:stop]
    marker = re.sub(r'[^\w]', '', canonical_script(''.join(line for _, line in body))).casefold()
    if marker in {'纯音乐请欣赏', '纯音乐请您欣赏', '此歌曲为没有填词的纯音乐请您欣赏',
                  '暂无歌词', '暂无该歌曲歌词', '歌词暂未提供', 'nolyricsavailable',
                  'lyricsnotavailable', 'instrumental'}:
        return [], sorted(metadata+[i for i, _ in rows])
    if body and all(i in timestamps for i, _ in body):
        body = [(i, line) for _, i, line in sorted((at, i, line) for i, line in body for at in timestamps[i])]
    elif any(len(timestamps.get(i, [])) > 1 for i, _ in body):
        raise ValueError('Repeated LRC lines need a fully timed body to preserve lyric order')
    return join_punctuation_lines([line for _, line in body]), sorted(metadata+[i for i, _ in rows[:first]+rows[stop:]])


def description_lyrics(metadata, reference_lines=()):
    """Select an explicitly labelled or corroborated block, never generate text."""
    from rapidfuzz.fuzz import partial_ratio
    description = (metadata.get('description') or '')[:30000]
    rows = list(enumerate(description.splitlines(), 1))
    normalize = lambda text: ''.join(c for c in canonical_script(text).casefold() if c.isalnum())
    reference = normalize(''.join(reference_lines))
    song = video_song_info(metadata)
    sections, section, labelled = [], [], False
    for number, line in [*rows, (len(rows)+1, '#end')]:
        if re.fullmatch(r'[\[【<＜#\s]*(?:lyrics?|歌詞|歌词|가사)[\]】>＞\s:：]*', line, re.I):
            section, labelled = [], True
        elif re.search(r'https?://|www\.|^#|^[-_=━─]{2,}$', line.strip()):
            if section:
                sections.append((section, labelled))
            section, labelled = [], False
        else:
            section.append((number, line.strip()))
    candidates = []
    for section, labelled in sections:
        if not labelled:
            if len(reference) < 40:
                continue
            matching = [i for i, (_, line) in enumerate(section) if len(normalize(line)) >= 4 and
                        partial_ratio(normalize(line), reference) >= 85]
            if len(matching) < 3:
                continue
            first, last = matching[0], matching[-1]
            # Retain whole opening/closing stanzas, including orthographic
            # variants that did not match a particular reference spelling.
            while first and section[first-1][1]: first -= 1
            while last+1 < len(section) and section[last+1][1]: last += 1
            # A repeated lyric already retained inside this block remains
            # supported at its edges, even when the reference spells it
            # differently. Blank lines format lyrics; unrelated prose stops us.
            retained = {normalize(line) for _, line in section[first:last+1] if line}
            for i in range(first-1, -1, -1):
                if section[i][1] and normalize(section[i][1]) not in retained:
                    break
                first = i
            for i in range(last+1, len(section)):
                if section[i][1] and normalize(section[i][1]) not in retained:
                    break
                last = i
            section = section[first:last+1]
        selected = [(n, line) for n, line in section if line]
        text, excluded = published_lyric_lines('\n'.join(line for _, line in selected),
            song['title'], [song['artist']], album=song.get('album'))
        body = normalize(''.join(text))
        overlap = sum(b.size for b in SequenceMatcher(None, body, reference, autojunk=False).get_matching_blocks())
        if not body or (not labelled and overlap < max(len(reference), len(body))*.7):
            continue
        candidates.append((labelled, overlap, text, [n for i, (n, _) in enumerate(selected, 1) if i not in excluded]))
    if not candidates:
        return None
    labelled, _, text, numbers = max(candidates, key=lambda item: item[:2])
    return text, dict(provider='video_description', publisher='youtube', kind='video_description',
        language=text_language(''.join(text)), url='https://www.youtube.com/watch?v='+metadata.get('id', ''),
        description_sha256=hashlib.sha256(description.encode()).hexdigest(), lyric_line_numbers=numbers,
        text_sha256=hashlib.sha256('\n'.join(text).encode()).hexdigest(),
        extraction='explicit_lyric_heading' if labelled else 'published_text_overlap',
        provider_timestamps_used=False)


def version_tags(text):
    text = canonical_script(text or '').casefold()
    return {kind for kind, pattern in (
        ('live', r'\blive(?:版本?)?\b|演唱会|现场'), ('remix', r'\bremix\b|混音'),
        ('acoustic', r'\bacoustic\b'), ('instrumental', r'\binstrumental\b|纯音乐'),
        ('karaoke', r'\bkaraoke\b|伴奏版')) if re.search(pattern, text)}


def recording_mismatches(song, source_title, title, album):
    expected = version_tags(source_title) | version_tags(song['title'])
    actual = version_tags((title or '')+' '+(album or ''))
    # Live/acoustic releases may supply text for alignment to our own audio.
    # Instrumentals, karaoke edits and remixes are not interchangeable lyrics.
    if (expected ^ actual) - {'live', 'acoustic'}:
        return None
    def edition_key(label):
        name = re.sub(r'\s*(?:\b(?:version|ver\.?)|版本?)$', '', label, flags=re.I).strip().casefold()
        language = language_code({'国': 'Mandarin', '国语': 'Mandarin', '粤': 'Cantonese'}.get(canonical_script(name), name))
        return language or canonical_script(label).casefold()
    editions = lambda values: {edition_key(m[2]) for value in values
        if (m := _versioned_title.fullmatch(unicodedata.normalize('NFKC', value or '')))}
    mismatch = ['version'] if expected != actual or editions((source_title, song['title'])) != editions((title, album)) else []
    features = lambda values: {name for value in values
        if (match := _featured_title.fullmatch(unicodedata.normalize('NFKC', value or '')))
        for name in identity(match[2] or match[3], title=True)}
    expected_features, actual_features = features((source_title, song['title'])), features((title,))
    if expected_features and actual_features and not expected_features & actual_features:
        mismatch.append('featured_performers')
    performance = live_performance(source_title)
    if performance and performance not in (live_performance(title), live_performance(album)):
        mismatch.append('live_performance')
    return mismatch


def short_lyric_text(lines, duration):
    """Reject obvious scraps, not certify completeness or acoustic coverage."""
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return None
    text = unicodedata.normalize('NFKC', ' '.join(lines)).casefold()
    units = len(re.findall(r"[a-z\u00c0-\u024f]+(?:['’][a-z\u00c0-\u024f]+)*|\d+|[^\W_]", text))
    # Deliberately sparse: one Latin word / other character per ten seconds.
    # Count repetitions and ignore line wrapping; neither is evidence of junk.
    minimum = math.ceil(duration / 10)
    if units < minimum:
        return dict(reason='lyrics_too_short', units=units, minimum_units=minimum, duration=duration)
    return None


def published_candidate(rows, song, source_title, *, read_lyrics=None, allow_recording_mismatch=False,
                        duration=None, reference_lines=(), source_url=None):
    """Select published text; recording onsets still come entirely from our audio."""
    reference = text_units(''.join(reference_lines), text_language(''.join(reference_lines)) or 'English')
    usable_reference = len(reference) >= 40 and len(set(reference)) >= 10
    candidates, identity_candidates = [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get('trackTitle')
        if not isinstance(title, str) or not isinstance(row.get('artists'), list) or not isinstance(row.get('album'), dict):
            continue
        artists = [a.get('artistName') for a in row['artists'] if isinstance(a, dict)]
        artist_aliases = [name for a in row['artists'] if isinstance(a, dict)
                          for name in a.get('artistAliases') or [] if isinstance(name, str)]
        album = row['album'].get('albumTitle', '')
        if not isinstance(album, str):
            continue
        full_credit = any(artist_credit_matches(credit, artists) or artist_credit_matches(credit, artist_aliases)
                          for credit in song_names(song, 'artist'))
        partial_credit = allow_recording_mismatch and any(artist_credit_matches(credit, artists+artist_aliases, partial=True)
                                                          for credit in song_names(song, 'artist'))
        if not (full_credit or partial_credit):
            continue
        title_aliases = [title, *(name for name in row.get('trackAliases') or [] if isinstance(name, str))]
        title_match = any(identity(name, title=True) & identity(alias, title=True)
                          for name in song_names(song, 'title') for alias in title_aliases)
        if not title_match:
            identity_candidates.append(dict(title=title, track_id=str(row.get('trackId', ''))))
            if not usable_reference:
                continue
        mismatch = recording_mismatches(song, source_title, title, album)
        if mismatch is None:
            continue
        if not full_credit:
            mismatch.append('performers')
        seconds = row.get('duration_seconds')
        seconds = seconds if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0 else None
        difference = abs(seconds-duration) if seconds is not None and duration is not None else math.inf
        if duration is not None and math.isfinite(difference) and difference > max(3., min(10., duration*.02)):
            mismatch = [*mismatch, 'duration']
        if mismatch and not allow_recording_mismatch:
            continue
        album_match = bool(identity(song.get('album')) & identity(album))
        rank = (not title_match, len(mismatch), not album_match, difference)
        candidates.append((rank, mismatch, row, title, artists, album, seconds, title_match))
    # A title can have different release edits. Use known recording metadata
    # before provider order, while retaining complete text from other releases.
    rejected, identity_attempts = [], 0
    for _, mismatch, row, title, artists, album, seconds, title_match in sorted(candidates, key=lambda c: c[0]):
        if not title_match:
            if identity_attempts >= 5:
                continue
            identity_attempts += 1
        try:
            text = row.get('lyrics')
            if text is None and read_lyrics is not None:
                text = read_lyrics(row)
            if not isinstance(text, str) or not text.strip() or len(text) > 30000:
                continue
            lines, excluded = published_lyric_lines(text, title, artists, album=album)
        except Exception as error:
            failure = lyric_request_error(error)
            if failure.get('requires_access') or failure.get('http_status') in (401, 403, 429):
                raise
            rejected.append(dict(reason='candidate_unavailable', track_id=str(row.get('trackId', '')),
                                 **failure))
            continue
        if not lines:
            rejected.append(dict(reason='empty_lyric_body', track_id=str(row.get('trackId', '')),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded_credit_lines=excluded))
            continue
        language = song.get('language') or text_language('\n'.join(lines))
        identity_support = None
        if not title_match:
            left = text_units(''.join(lines), language)
            right = text_units(''.join(reference_lines), language)
            matched = sum(o.src_end-o.src_start for o in Levenshtein.opcodes(left, right) if o.tag == 'equal')
            if min(len(left), len(right)) < 40 or min(len(set(left)), len(set(right))) < 10 or (
                    matched < .9*min(len(left), len(right)) or matched < .65*max(len(left), len(right))):
                rejected.append(dict(reason='unconfirmed_title_alias', track_id=str(row.get('trackId', ''))))
                continue
            identity_support = dict(matched_units=matched, candidate_units=len(left), reference_units=len(right),
                reference_sha256=hashlib.sha256('\n'.join(reference_lines).encode()).hexdigest())
        track_id = str(row.get('trackId', ''))
        url = source_url(track_id) if source_url else ('https://music-tw.line.me/track/'+track_id if track_id.isdigit() else None)
        if not url or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            continue
        if issue := short_lyric_text(lines, duration):
            rejected.append(dict(issue, track_id=track_id, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            continue
        return lines, dict(status='candidate', provider='line_music', language=language,
            url=url, track=title, artists=artists, album=album,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded_credit_lines=excluded,
            recording_mismatches=mismatch, duration_seconds=seconds,
            title_match='title' if title_match else 'corroborated_lyric_text', identity_support=identity_support,
            rejected_candidates=rejected,
            text_transform='html_entities_decoded_credit_header_footer_removed_then_whitespace_only',
            provider_timestamps_used=False)
    return None, dict(status='no_matching_published_text', provider='line_music', rejected_candidates=rejected,
                      identity_candidates=identity_candidates)


def fetch_published_lyrics(song, source_title, cache, duration=None, *, reference_lines=()):
    if not song.get('title') or not song.get('artist'):
        return None, dict(status='missing_identity', provider='line_music')
    path, stage = cache.root/'published-lyrics.json', 'fetch_published_lyrics'
    title = song_names(song, 'title')[-1]
    queries = list(dict.fromkeys([' '.join((title, song_names(song, 'artist')[-1])), title]))
    try:
        old = json.loads(path.read_text())
        if lyric_retry_due(dict(old.get('report') or {}, retry_at=old.get('retry_at')), 'line_music', cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__)], outputs=[path],
                     config=dict(song=song, source_title=source_title, duration=duration, reference_lines=reference_lines),
                     code=[Path(__file__), Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            try:
                attempted, rejected = [], []
                for query in queries:
                    url = 'https://music-tw.line.me/api/search/v4/track.json?'+urlencode(
                        dict(query=query, start=1, display=30, sort='RELEVANCE'))
                    response = cached_lyric_request('line_music', url, cache,
                        lambda: read_document(url, headers={'CHANNEL-ID': '1581655766'}))
                    rows = response['response']['result']['tracks']
                    if not isinstance(rows, list):
                        raise ValueError('Invalid published lyric response')
                    attempted.append(dict(query=query, results=rows))
                    lines, report = published_candidate(rows, song, source_title,
                        allow_recording_mismatch=True, duration=duration, reference_lines=reference_lines)
                    rejected.extend(report.get('rejected_candidates', []))
                    if lines:
                        break
                report['rejected_candidates'] = rejected
                packet = dict(lines=lines, report=report, query_results=attempted)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                report = lyric_request_error(error)
                packet = dict(lines=None, report=dict(report, provider='line_music', rejected_candidates=rejected),
                    query_results=attempted, retry_at=report.get('retry_at'))
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    packet = json.loads(path.read_text())
    return packet['lines'], dict(packet['report'], **({'retry_at':packet['retry_at']} if packet.get('retry_at') else {}))


def catalogue_lyric_identity(song, duration, cache):
    """Ground an upload label in catalogue credits, never in its channel name."""
    titles = names(song.get('title'), title=True)
    if not titles or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return None, dict(status='missing_title_or_duration')
    # Delimiters supply a search hint, not a trusted song identity. The
    # catalogue must still account for both work and credit in the upload.
    quoted = regex.findall(r'[\p{Ps}\p{Pi}"]([^\p{Ps}\p{Pe}\p{Pi}\p{Pf}"\n]+)[\p{Pe}\p{Pf}"]', titles[0])
    # Search engines can treat adjacent bracketed credits as one literal
    # token. Keep every word, but separate the title and credit for search.
    label_query = ' '.join(regex.sub(r'[\p{Ps}\p{Pe}\p{Pi}\p{Pf}"]', ' ', titles[-1]).split())
    queries = list(dict.fromkeys(([quoted[0]] if len(set(quoted)) == 1 else []) + [label_query]))
    upload = canonical_script(titles[0]).casefold()

    def named_span(value, text, *, title=False):
        for alias in sorted(names(value, title=title), key=len, reverse=True):
            units = re.findall(r'[a-z\u00c0-\u024f]+|\d+|[^\W_]', canonical_script(alias).casefold())
            if units:
                match = re.search(r'(?<![a-z\u00c0-\u024f0-9])'+r'\W*'.join(map(re.escape, units))+
                                  r'(?![a-z\u00c0-\u024f0-9])', text)
                if match:
                    return match

    def rank(candidate):
        delta = candidate['duration_difference']
        return candidate['identity_basis'] != 'upload_title_and_credit', delta if delta is not None else math.inf

    attempts, lyric_candidates = [], {}
    searches = [*product(queries, ('netease', 'line_music')),
                *((query, 'youtube_music') for query in queries)]
    for query, provider in searches:
        provider_query = _traditional.convert(query) if provider == 'line_music' else query
        try:
            if provider == 'netease':
                url = 'https://music.163.com/api/cloudsearch/pc?'+urlencode(dict(s=provider_query, type=1, limit=30, offset=0))
                packet = cached_lyric_request(provider, url, cache, lambda: read_document(url, timeout=10))
                if packet.get('code') != 200:
                    raise ValueError('NetEase catalogue search rejected the request')
                rows = [dict(title=r.get('name'), artists=[a.get('name') for a in r.get('ar') or [] if isinstance(a, dict)],
                    seconds=r['dt']/1000 if isinstance(r.get('dt'), (int, float)) else None,
                    url='https://music.163.com/#/song?id='+str(r.get('id')))
                    for r in packet.get('result', {}).get('songs', []) if isinstance(r, dict)]
            elif provider == 'line_music':
                url = 'https://music-tw.line.me/api/search/v4/track.json?'+urlencode(
                    dict(query=provider_query, start=1, display=30, sort='RELEVANCE'))
                packet = cached_lyric_request(provider, url, cache,
                    lambda: read_document(url, headers={'CHANNEL-ID': '1581655766'}))
                rows = [dict(title=r.get('trackTitle'), artists=[a.get('artistName') for a in r.get('artists') or [] if isinstance(a, dict)],
                    seconds=None, url='https://music-tw.line.me/track/'+str(r.get('trackId')))
                    for r in packet['response']['result']['tracks'] if isinstance(r, dict)]
            else:
                from ytmusicapi import YTMusic
                with requests.Session() as session:
                    session.request = partial(session.request, timeout=10)  # ty: ignore[invalid-assignment] # Bind a default timeout to this third-party session only.
                    session.hooks['response'].append(lambda response, *a, **kw: response.raise_for_status())
                    music = YTMusic(language='en', requests_session=session)
                    packet = cached_lyric_request(provider, dict(method='search', query=provider_query, locale='en'), cache,
                        lambda: music.search(provider_query, filter='songs', limit=20))
                rows = [dict(title=r.get('title'), artists=[a.get('name') for a in r.get('artists') or [] if isinstance(a, dict)],
                    seconds=r.get('duration_seconds'), url='https://music.youtube.com/watch?v='+r['videoId'])
                    for r in packet if isinstance(r, dict) and re.fullmatch(r'[A-Za-z0-9_-]{11}', r.get('videoId') or '')]
        except Exception as error:
            attempts.append(dict(lyric_request_error(error), provider=provider, query=provider_query))
            continue
        candidates = {}
        for row in rows:
            title, artists, seconds = row['title'], row['artists'], row['seconds']
            if not isinstance(title, str) or not artists or not all(isinstance(a, str) and a.strip() for a in artists):
                continue
            seconds = seconds if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0 else None
            span = named_span(title, upload, title=True)
            outside_title = upload[:span.start()]+' '+upload[span.end():] if span else ''
            grounded = bool(span and all(named_span(a, outside_title) for a in artists))
            delta = abs(seconds-duration) if seconds is not None else None
            key = (comparison_key(names(title, title=True)[-1]), tuple(sorted(tuple(sorted(identity(a))) for a in artists)))
            candidate = dict(title=title, artist=' · '.join(artists), duration_seconds=seconds,
                duration_difference=delta, url=row['url'], identity_basis='upload_title_and_credit' if grounded else 'title_and_duration')
            title_match = bool(identity(song['title'], title=True) & identity(title, title=True))
            if title_match and (key not in lyric_candidates or rank(candidate) < rank(lyric_candidates[key])):
                lyric_candidates[key] = dict(candidate, identity_basis='work_title_only')
            if not grounded and not (title_match and delta is not None and delta <= max(3., min(10., duration*.02))):
                continue
            if key not in candidates or rank(candidate) < rank(candidates[key]):
                candidates[key] = candidate
        ordered = sorted(candidates.values(), key=rank)
        report = dict(status='candidate' if ordered else 'no_match', provider=provider, query=provider_query, candidates=ordered,
                      basis='Catalogue identity; candidate text still requires acoustic alignment')
        attempts.append(report)
        if ordered:
            # Distinct complete credits in one label can describe a medley.
            # Duration rounding alone also cannot disambiguate two singers.
            if len(ordered) > 1 and rank(ordered[0])[0] == rank(ordered[1])[0] and (
                    not rank(ordered[0])[0] or rank(ordered[1])[1] < rank(ordered[0])[1]+1):
                return None, dict(report, status='ambiguous', attempts=attempts)
            return ordered[0], dict(report, status='candidate', attempts=attempts)
    # A different edit length prevents a recording-identity claim, not a
    # lookup of its published words. Keep these explicitly unconfirmed and
    # use the same five-candidate budget as alternate Music lyric releases.
    return None, dict(attempts[-1], lyric_candidates=sorted(lyric_candidates.values(), key=rank)[:5], attempts=attempts)


def catalogue_lyric_text(candidates, metadata, duration, cache):
    """Propose published work text without assigning its singer to an upload."""
    from .lyric_consensus import compare_lyrics, fetch_lyric_consensus
    texts, attempts, seen = [], [], set()
    for candidate in candidates:
        # Keep the job's shared response cache and provider backoff. Each
        # lookup is identity-bound; retain its whole report before the next.
        song = {k:candidate[k] for k in ('title', 'artist')}
        lines, report = fetch_lyric_consensus(song, metadata, duration, cache, preserve_title=True)
        attempts.append(dict(candidate=candidate, lookup=report))
        review = report.get('crosscheck', {})
        rejected = {row['source'] for row in review.get('rejected_sources', [])}
        rejected.update(row['source'] for row in review.get('serialized_reading_variants', []))
        available = list(zip(report.get('candidate_texts', []), review.get('sources', [])))
        order = dict.fromkeys([review.get('selected_source'), *review.get('eligible_sources', []), *range(len(available))])
        for i in order:
            if not isinstance(i, int) or i not in range(len(available)) or i in rejected:
                continue
            body, source = available[i]
            identity_key = (source.get('provider'), source.get('url'), tuple(body))
            if identity_key not in seen:
                seen.add(identity_key)
                texts.append((body, dict(source, catalogue_reference=candidate, performer_unconfirmed=True)))
    _, review = compare_lyrics(texts)
    if not texts:
        return None, dict(status='unavailable', attempts=attempts)
    index = next(iter(review['agreeing_sources']), 0)
    review.update(selected_source=index, selection='catalogue_work_text', eligible_sources=list(range(len(texts))),
                  rejected_sources=[], unavailable_sources=[])
    lines, source = texts[index]
    return lines, dict(source, status='candidate', crosscheck=review, candidate_texts=[body for body, _ in texts],
                       catalogue_attempts=attempts)


def fetch_reading_lyrics(song, cache, duration=None, reference_lines=(), source_title=None):
    """Reuse the published ruby parser for original Japanese text as well."""
    from .lyric_readings import ReadingPage, read_page
    title, artist = song_names(song, 'title')[-1], song_names(song, 'artist')[-1]
    fetch = lambda url: cached_lyric_request('utaten_readings', url, cache, lambda: read_page(url))
    attempts, seen, reading_evidence = [], set(), []
    reference = ''.join(reference_lines)
    queries = [dict(title=title, artist_name=artist), dict(title=title)]
    phrase = next((line.strip() for line in reference_lines if re.search('[\u3400-\u9fff]', line)), '')
    if phrase:
        queries.append(dict(body=phrase))
    for query in queries:
        url = 'https://utaten.com/search?'+urlencode(query)
        listing = ReadingPage(fetch(url))
        for link in listing.links[:5]:
            if link in seen:
                continue
            seen.add(link)
            url = 'https://utaten.com'+link
            page = ReadingPage(fetch(url))
            actual = page.identity
            if page.includes_translation:
                # This publisher labels combined originals/translations in
                # the page metadata, but does not mark their body boundaries.
                attempts.append(dict(url=url, identity=actual, reason='published_translation_included'))
                continue
            title_match = any(identity(name, title=True) & identity(alias, title=True)
                              for name in song_names(song, 'title') for alias in song_names(actual, 'title'))
            artist_match = any(artist_credit_matches(name, [alias]) for name in song_names(song, 'artist')
                               for alias in song_names(actual, 'artist'))
            partial_credit = any(artist_credit_matches(name, [alias], partial=True) for name in song_names(song, 'artist')
                                 for alias in song_names(actual, 'artist'))
            mismatch = recording_mismatches(song, source_title, actual.get('title'), None)
            text, excluded = published_lyric_lines(page.text, actual.get('title') or '', [actual.get('artist')])
            left, right = text_units(''.join(text), 'Japanese'), text_units(reference, 'Japanese')
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(left, right) if b.tag == 'equal')
            corroborated = (min(len(left), len(right)) >= 40 and min(len(set(left)), len(set(right))) >= 10 and
                            matched >= min(len(left), len(right))*.9 and matched >= max(len(left), len(right))*.65)
            attempts.append(dict(url=url, identity=actual, title_match=title_match, artist_match=artist_match,
                                 corroborated=corroborated))
            if title_match or corroborated:
                reading_evidence.append(dict(url=url, pairs=[dict(text=page.text[s['first']:s['stop']],
                    reading=s['reading']) for s in page.spans]))
            if text and mismatch is not None and (title_match and (artist_match or partial_credit) or corroborated) and not short_lyric_text(text, duration):
                if partial_credit and not artist_match:
                    mismatch.append('performers')
                readings = [dict(after=len(comparison_key(page.text[:span['stop']])),
                                 text=comparison_key(span['reading'])) for span in page.spans
                            if comparison_key(span['reading'])] if comparison_key(page.text) == comparison_key(''.join(text)) else []
                return text, dict(status='candidate', provider='utaten', publisher='utaten', language=text_language('\n'.join(text)),
                    url=url, track=actual.get('title'), artist=actual.get('artist'), attempts=attempts,
                    text_sha256=hashlib.sha256(page.text.encode()).hexdigest(), excluded_credit_lines=excluded,
                    text_transform='published_ruby_body_without_readings', recording_mismatches=mismatch,
                    provider_timestamps_used=False, reading_annotations=readings, reading_evidence=reading_evidence)
    return None, dict(status='no_matching_published_text', provider='utaten', attempts=attempts,
                      reading_evidence=reading_evidence)


def structured_recordings(document):
    """Read schema.org recording data, never the surrounding editorial text."""
    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.recordings, self.script = [], None
        def handle_starttag(self, tag, attrs):
            if tag == 'script' and dict(attrs).get('type') == 'application/ld+json':
                self.script = ''
        def handle_data(self, data):
            if self.script is not None:
                self.script += data
        def handle_endtag(self, tag):
            if tag != 'script' or self.script is None:
                return
            try:
                data = json.loads(self.script)
                records = data if isinstance(data, list) else data.get('@graph', [data])
                self.recordings.extend(r for r in records if isinstance(r, dict) and r.get('@type') == 'MusicRecording')
            except (ValueError, AttributeError, TypeError):
                pass  # Another script may contain the recording's valid data.
            self.script = None
    page = Page()
    page.feed(document)
    return page.recordings


def read_lyric_catalogue(url):
    """Discard repeated biographies/editorial data before caching search metadata."""
    with httpx.stream('GET', url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 16*1024*1024:
                raise ValueError('Lyric catalogue response exceeds 16 MiB')
    data = json.loads(body).get('data') or {}
    rows = []
    for row in (data.get('global_song_matches') or []) + (data.get('artist_songs') or []):
        if not isinstance(row, dict):
            continue
        credit = row.get('artist')
        artists = [credit] if isinstance(credit, dict) else credit if isinstance(credit, list) else []
        rows.append(dict(id=row.get('_id'), title=row.get('song_name'), path=row.get('public_song_url'),
                         artists=[a.get('artist_name') for a in artists if isinstance(a, dict)]))
    return dict(rows=rows, response_sha256=hashlib.sha256(body).hexdigest(), response_bytes=len(body))


def fetch_structured_lyrics(song, cache, duration=None, source_title=None, *, reference_lines=()):
    """Read a publisher's explicit lyric field with the existing identity matcher."""
    from .lyric_readings import read_page
    if not song.get('title') or not song.get('artist'):
        return None, dict(status='missing_identity', provider='lyricsweb')
    # This endpoint searches song names and artist names separately. Combining
    # them returns an artist catalogue, which can omit the requested song.
    # The provider accepts 2–120 characters. Keep short/long work names intact
    # for matching; query a declared alias or the artist's catalogue instead.
    query = next((name for key in ('title', 'artist') for name in reversed(song_names(song, key))
                  if 2 <= len(name) <= 120), None)
    if query is None:
        return None, dict(status='unsupported_query', provider='lyricsweb')
    url = 'https://lyricsweb.com/api/v1/user/search-result?'+urlencode(dict(query=query))
    packet = cached_lyric_request('lyricsweb', dict(url=url, representation='candidate_metadata_v1'), cache,
                                  lambda: read_lyric_catalogue(url))
    rows, links, publishers = [], {}, {}
    for row in packet['rows']:
        if not row.get('id'):
            continue
        path = row.get('path') or ''
        if not isinstance(path, str) or not re.fullmatch(r'/song/[\w%/-]+', path):
            continue
        key = str(row['id'])
        links[key] = 'https://lyricsweb.com'+path
        rows.append(dict(trackId=key, trackTitle=row.get('title'),
            artists=[dict(artistName=a) for a in row['artists']], album={}))
    def lyrics(row):
        url = links[row['trackId']]
        document = cached_lyric_request('lyricsweb', url, cache, lambda: read_page(url))
        for recording in structured_recordings(document):
            credit = recording.get('byArtist') or {}
            credits = [credit] if isinstance(credit, dict) else credit if isinstance(credit, list) else []
            artists = [a.get('name') for a in credits if isinstance(a, dict)]
            if (not identity(recording.get('name'), title=True) & identity(row['trackTitle'], title=True) or
                    not any(artist_credit_matches(a['artistName'], artists, partial=True) for a in row['artists'])):
                continue
            composition = recording.get('recordingOf')
            body = composition.get('lyrics') if isinstance(composition, dict) else None
            if not isinstance(body, dict):
                continue
            text = body.get('text')
            if not isinstance(text, str) or len(text) > 30000:
                continue
            provider = body.get('provider')
            publisher = provider.get('name') if isinstance(provider, dict) else None
            publishers[url] = publisher.casefold() if isinstance(publisher, str) and publisher else 'lyricsweb'
            # This publisher serializes line breaks inside the JSON text field.
            return text.replace('\\n', '\n')
        return None
    lines, report = published_candidate(rows, song, source_title, read_lyrics=lyrics,
        allow_recording_mismatch=True, duration=duration, source_url=links.get, reference_lines=reference_lines)
    if lines:
        report.update(publisher=publishers[report['url']],
                      text_transform='schema.org.lyrics_text_credit_metadata_removed')
    return lines, dict(report, provider='lyricsweb')


def prepare_online_lyrics(request, metadata, duration, cache, model, catalog_root):
    """Find supplied text before ASR; this is a candidate, not independent audio validation."""
    song = video_song_info(metadata)
    # Upload/default-track tags are hints, not the language of published text.
    # Keep explicit user choices; otherwise choose from the actual lyric body.
    metadata_language = song.pop('language', None)
    song.update({key: request[key] for key in ('title', 'artist', 'language') if request.get(key)})
    lines, provenance = request.get('lyrics'), dict(kind='user_supplied')
    metadata_error = None
    if not lines:
        catalog = find_verified_lyrics(request['url'], duration, catalog_root)
        if catalog:
            return dict(catalog, id='song', source_url=request['url'], provenance=dict(
                catalog['provenance'], route='lyrics_first', needs_review=True, independent_asr_coverage=None))
        description = (metadata.get('description') or '')[:30000]
        description_lines = description.splitlines()
        evidence = {key: metadata.get(key) for key in ('title', 'track', 'artist', 'artists', 'uploader')}
        evidence['description_lines'] = [dict(number=i+1, text=line) for i, line in enumerate(description_lines)]
        from openai import OpenAIError
        preserve_title = bool(request.get('title') or metadata.get('track'))
        metadata_read, parsed, metadata_failure = False, None, None
        metadata_retries = []
        def read_description(resolve_identity=(), *, retry=False):
            nonlocal metadata_error, metadata_read, parsed, metadata_failure
            if not resolve_identity and not description.strip():
                return None
            if retry and metadata_failure and not metadata_retries:
                error, failed_at = metadata_failure
                fields = retry_fields(error, '', 1, failed_at=failed_at)
                deadline = fields.get('next_retry_at')
                delay = max(0, deadline-time.time()) if deadline else None
                metadata_retries.append(dict(error=error, next_retry_at=deadline,
                                             attempted=delay is not None and delay <= 60))
                if delay is not None and delay <= 60:
                    time.sleep(delay)
                    metadata_read = False
            try:
                if not metadata_read:
                    metadata_read = True
                    packet = extract_song_metadata(evidence, model, cache)
                    parsed = SongText.model_validate(packet['parsed']) if packet['parsed'] is not None else None
            except (OpenAIError, ValueError) as error:
                metadata_error = type(error).__name__
                classified = provider_error(error)
                metadata_failure = (classified, time.time()) if classified else None
                parsed = None
            if parsed is None:
                return None
            if 'title' in resolve_identity:
                song['title'] = parsed.title or song['title']
            if 'artist' in resolve_identity:
                song['artist'] = parsed.artist or song['artist']
            selected = parsed.lyric_line_numbers
            if not selected or selected != sorted(set(selected)) or not all(1 <= i <= len(description_lines) for i in selected):
                return None
            text = [description_lines[i-1].strip() for i in selected if description_lines[i-1].strip()]
            if not text:
                return None
            return text, dict(kind='video_description', provider='video_description', publisher='youtube',
                description_sha256=hashlib.sha256(description.encode()).hexdigest(), lyric_line_numbers=selected,
                language=song.get('language') or text_language(''.join(text)), url=request['url'],
                text_sha256=hashlib.sha256('\n'.join(text).encode()).hexdigest())
        from .lyric_consensus import fetch_lyric_consensus, resolve_recording_artists
        lines, report = fetch_lyric_consensus(song, metadata, duration, cache, read_description, preserve_title)
        review = report.get('crosscheck', {})
        recording = next((r.get('recording_identity') or {} for r in
            [*review.get('sources', []), *review.get('unavailable_sources', [])]
            if r.get('provider') == 'youtube_music' and
            (r.get('recording_identity') or {}).get('video_id') == metadata.get('id')), {})
        channel_name = metadata.get('channel') or metadata.get('uploader')
        # A channel is not a performer credit. Only reuse its bilingual name
        # when Music identifies that exact channel as this recording's artist.
        artist_alias = channel_name if channel_name and any(
            a.get('id') and a['id'] == metadata.get('channel_id') for a in recording.get('artists') or []) and (
            identity(channel_name) & identity(recording.get('artist'))) and not (
            identity(channel_name) <= identity(song.get('artist'))) else None
        confirmed_names = dict(title=recording.get('title'), artist=artist_alias or recording.get('artist'))
        unused_alias = any(name and not request.get(key) and not
            identity(name, title=key == 'title') <= identity(song.get(key), title=key == 'title')
            for key, name in confirmed_names.items())
        needs_support = review.get('status') != 'corroborated' and review.get('selection') != 'similarity_support'
        if not lines or (unused_alias and needs_support) or (parsed and review.get('status') == 'insufficient_sources'):
            retained_lines, retained_report, retained_song = lines, report, dict(song)
            original = {key: song.get(key) for key in ('title', 'artist')}
            previous, attempts, accepted = report, [], False
            artist_resolution = None
            # Retain a platform-confirmed performer while resolving an upload
            # title. A description can also credit producers or session players.
            for key in ('title', 'artist'):
                if request.get(key):
                    continue
                before = dict(song)
                review = report.get('crosscheck', {})
                confirmed_alias = confirmed_names.get(key)
                # Try the recording's unused credit before an artist page can
                # replace it with the same spelling that already found no text.
                if (key == 'artist' and not artist_alias and not retained_lines and recording.get('artists')
                        and identity(confirmed_alias) <= identity(song.get(key))):
                    confirmed_alias, artist_resolution = resolve_recording_artists(recording, cache)
                if confirmed_alias and not identity(confirmed_alias, title=key == 'title') <= identity(song.get(key), title=key == 'title'):
                    song[key] = confirmed_alias
                    description_text = None
                elif not retained_lines or parsed:
                    description_text = read_description([key], retry=True)
                    if parsed is None and metadata_failure and not metadata_retries:
                        description_text = read_description([key], retry=True)
                else:
                    continue
                if song == before:
                    continue
                blocked = {r['provider']: (None, r) for r in report.get('crosscheck', {}).get('unavailable_sources', [])
                           if r.get('provider') and r.get('error') and not lyric_retry_due(r, r['provider'], cache)}
                blocked.update({r['provider']: (text, r) for text, r in
                    zip(report.get('candidate_texts', []), review.get('sources', [])) if r['provider'] == 'youtube_music'})
                lines, report = fetch_lyric_consensus(song, metadata, duration, cache, description_text, True,
                                                      prefetched=blocked)
                attempts.append(dict(identity={key: song.get(key) for key in original}, lookup=report))
                # A source can attach another song's text to valid metadata.
                # Reuse already-read native credits to seek corroboration;
                # a second isolated guess cannot replace the first candidate.
                review = report.get('crosscheck', {})
                accepted = bool(lines) and (not retained_lines or review.get('status') == 'corroborated' or
                                           review.get('selection') == 'similarity_support')
                if accepted:
                    break
            if retained_lines and not accepted:
                lines, report = retained_lines, retained_report
                song.clear()
                song.update(retained_song)
            if attempts:
                report: dict[str, Any] = dict(report, identity_recovery=dict(original=original,
                    resolved={key: song.get(key) for key in original}, recovered=accepted,
                    attempts=attempts, previous_lookup=previous))
            if artist_resolution:
                report = dict(report, artist_identity_lookup=artist_resolution)
        # Watch-page "artists" can merely repeat a video's upload channel.
        # Upload-title parsing can reverse artist/title order. Only actual
        # music tags or explicit user/model credits exclude catalogue recovery.
        if (not lines and not request.get('artist') and not metadata.get('artist') and not metadata.get('artists')
                and not (parsed and parsed.artist)):
            upload = dict(song, title=request.get('title') or metadata.get('title') or song.get('title'))
            candidate, recovery = catalogue_lyric_identity(upload, duration, cache)
            original_report, original_song = report, dict(song)
            if candidate:
                song.update({key:candidate[key] for key in ('title','artist')})
                # A missing lyric result belongs to the old search identity.
                # Only live provider errors carry over after credits change.
                reused = {r['provider']:(None,r) for r in original_report.get('crosscheck', {}).get('unavailable_sources', [])
                          if r.get('provider') and r.get('error') and not lyric_retry_due(r, r['provider'], cache)}
                lines, report = fetch_lyric_consensus(song, metadata, duration, cache, None, True, prefetched=reused)
                if not lines:
                    recovery = dict(recovery, rejected_lookup=report)
                    lines, report = None, original_report
                    song.clear()
                    song.update(original_song)
            elif recovery.get('lyric_candidates'):
                lines, work_report = catalogue_lyric_text(recovery['lyric_candidates'], metadata, duration, cache)
                if lines:
                    song.update(title=names(upload['title'], title=True)[-1], artist='')
                    report = work_report
                    recovery = dict(recovery, performer_unconfirmed=True)
                else:
                    recovery = dict(recovery, rejected_lookup=work_report)
            method = 'catalogue_work_text' if recovery.get('performer_unconfirmed') else 'catalogue_title_and_duration'
            report = dict(report, identity_recovery=dict(method=method,
                          catalogue=recovery, recovered=bool(lines), previous_lookup=original_report,
                          original={k:original_song.get(k) for k in ('title','artist')},
                          resolved={k:song.get(k) for k in ('title','artist')}))
        if report.get('language') == 'Chinese' and metadata_language == 'Cantonese' and not request.get('language'):
            report = dict(report, language='Cantonese')
        if metadata_language and report.get('language') and metadata_language != report['language']:
            report = dict(report, metadata_language_corrected=dict(original=metadata_language, resolved=report['language']))
        if metadata_error:
            report = dict(report, metadata_error=metadata_error)
        if metadata_retries:
            report = dict(report, metadata_retries=metadata_retries)
        (cache.root/'lyrics-first-lookup.json').write_text(json.dumps(report, ensure_ascii=False)+'\n')
        if not lines:
            return None
        song['language'] = report.get('language')
        provenance = dict(kind='online_lyric_text', lyric_lookup=report,
                          text_transform=report.get('text_transform', 'whitespace_only'))
    language = song.get('language') or text_language(''.join(lines))
    # Han script does not distinguish Mandarin from Cantonese. Retain a
    # specific recording tag when it is compatible with the published text.
    if language == 'Chinese' and metadata_language == 'Cantonese' and not request.get('language'):
        language = 'Cantonese'
    if len(lines) > 1000 or any(not line.strip() or len(line) > 200 for line in lines):
        raise ValueError('Found lyric text has invalid line lengths')
    return dict(song, id='song', source_url=request['url'], language=language, lines=lines,
                provenance=dict(provenance, needs_review=True, route='lyrics_first', independent_asr_coverage=None,
                                **({'metadata_error': metadata_error} if metadata_error else {})))


def prepared_lyrics_identity(url, metadata):
    """Bind split-host lookup to the same media metadata and lookup implementation."""
    root = Path(__file__).parent
    return dict(url=url, metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                code_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('lyric_lookup.py', 'lyric_consensus.py', 'song_sources.py', 'source_captions.py',
                                          'lyric_readings.py')})


def read_prepared_lyrics(path, url, metadata):
    packet = json.loads(Path(path).read_text())
    if packet['identity'] != prepared_lyrics_identity(url, metadata):
        raise RuntimeError('Prepared lyric lookup does not match this source or implementation')
    lyrics = packet['lyrics']
    if lyrics is not None and (lyrics['source_url'] != url or lyrics['provenance'].get('route') != 'lyrics_first'):
        raise RuntimeError('Prepared lyrics have the wrong source or route')
    return lyrics


def lyric_candidate(packet, song, transcription, duration, *, audio_path=None, source_title=None,
                    allow_recording_mismatch=False):
    if allow_recording_mismatch and transcription is not None:
        raise ValueError('Text-only candidates cannot validate independent ASR coverage')
    report = {key: value for key, value in packet.items() if key != 'results'}
    if packet.get('status') != 'ok':
        return None, report
    performance = live_performance(source_title)
    if performance:
        report['required_live_performance'] = performance
    candidates, padding, padding_checked = [], None, False
    report['rejected_candidates'] = []
    for row in packet['results']:
        # A text search can return covers, live cuts and unrelated recordings.
        # Identity and duration must agree before using any audio-text score.
        full_credit = any(artist_credit_matches(name, [row.get('artistName')]) for name in song_names(song, 'artist'))
        partial_credit = allow_recording_mismatch and any(artist_credit_matches(name, [row.get('artistName')], partial=True)
                                                          for name in song_names(song, 'artist'))
        if (not any(identity(name, title=True) & identity(row.get('trackName'), title=True) for name in song_names(song, 'title')) or
                not (full_credit or partial_credit) or row.get('instrumental') is not False):
            continue
        basis = source_title if allow_recording_mismatch or performance else song['title']
        mismatch = recording_mismatches(song, basis, row.get('trackName'), row.get('albumName'))
        if mismatch is None or mismatch and not allow_recording_mismatch:
            continue
        if not full_credit:
            mismatch.append('performers')
        try:
            reference_duration, basis = duration, None
            difference = abs(float(row['duration'])-reference_duration)
            if difference > max(3., min(10., duration*.02)) and audio_path is not None:
                if not padding_checked:
                    padding = unpadded_duration(audio_path, duration)
                    padding_checked = True
                if padding is not None:
                    reference_duration, basis = padding['nonzero_duration'], padding
                    difference = abs(float(row['duration'])-reference_duration)
            if not math.isfinite(difference):
                continue
            if difference > max(3., min(10., reference_duration*.02)):
                if not allow_recording_mismatch:
                    continue
                mismatch.append('duration')
            raw_text = row['plainLyrics']
            if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > 30000:
                continue
            clean_lines, excluded = published_lyric_lines(raw_text, row['trackName'], [row['artistName']],
                                                       album=row.get('albumName'))
            text = '\n'.join(clean_lines)
            if not text:
                report['rejected_candidates'].append(dict(reason='empty_lyric_body', id=row.get('id'),
                    text_sha256=hashlib.sha256(raw_text.encode()).hexdigest(), excluded_credit_lines=excluded))
                continue
            language = song.get('language') or text_language(text)
            lines = clean_lines
            if transcription is None and (issue := short_lyric_text(lines, duration)):
                report['rejected_candidates'].append(dict(issue, id=row.get('id'),
                    text_sha256=hashlib.sha256(raw_text.encode()).hexdigest()))
                continue
            observed = [u for w in (transcription or {}).get('words', []) for u in text_units(w['word'], language)]
            expected = text_units(''.join(lines), language)
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
            support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
            if (not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines) or
                    (transcription is not None and min(support, coverage) < .55)):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        candidates.append((not mismatch, min(support, coverage), -difference, lines, dict(
            id=row['id'], url=f"https://lrclib.net/api/get/{row['id']}",
            track=row['trackName'], artist=row['artistName'], album=row.get('albumName'),
            duration=row['duration'], language=language,
            text_support=support if transcription is not None else None,
            audio_word_coverage=coverage if transcription is not None else None,
            duration_basis=basis or dict(kind='recording', recording_duration=duration),
            recording_mismatches=mismatch,
            excluded_credit_lines=excluded, provider_timestamps_used=False,
            text_sha256=hashlib.sha256(raw_text.encode()).hexdigest())))
    if not candidates:
        return None, dict(report, status='no_supported_recording')
    best = max(candidates, key=lambda c: c[:3])
    return best[3], dict(report, status='accepted' if transcription is not None else 'candidate', provider='lrclib', **best[4])
