"""Source-bound Japanese readings and cached Cantonese dictionary readings."""
import hashlib
import inspect
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode


def hiragana(text):
    return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c
                   for c in unicodedata.normalize('NFKC', text))


def separate_inline_readings(lines, evidence):
    """Separate parenthetical ruby only when a publisher confirms the exact pair."""
    from .lyric_lookup import comparison_key
    pairs = {(comparison_key(p['text']), hiragana(p['reading'])): source['url']
             for source in evidence for p in source['pairs']}
    if not pairs:
        return lines, []
    pattern = re.compile(r'(?:(?P<opening>[《≪「『])(?P<quoted>[^《》≪≫「」『』\n]+)(?P<closing>[》≫」』])'
                         r'|(?P<plain>[\u3400-\u9fff]+))(?P<gap>\s*)[（(](?P<reading>[ぁ-ゖァ-ヶー]+)[）)]')
    cleaned, annotations = [], []
    for line in lines:
        parts, spans, cursor = [], [], 0
        for match in pattern.finditer(line):
            group = 'quoted' if match['opening'] else 'plain'
            base, reading = match[group], hiragana(match['reading'])
            source = pairs.get((comparison_key(base), reading))
            if ((match['opening'] and dict(zip('《≪「『', '》≫」』'))[match['opening']] != match['closing'])
                    or (group == 'plain' and match['gap'])
                    or not re.search('[\u3400-\u9fff]', base) or not source):
                continue
            parts.append(line[cursor:match.start(group)])
            first = sum(map(len, parts))
            parts.append(base + (match['closing'] or ''))
            spans.append(dict(first=first, stop=first+len(base), text=base, reading=reading, url=source))
            cursor = match.end()
        parts.append(line[cursor:])
        cleaned.append(''.join(parts))
        if spans:
            annotations.append(dict(text=cleaned[-1], source_text=line, spans=spans))
    return cleaned, annotations


class ReadingPage(HTMLParser):
    """Read UtaTen's ruby body; navigation, romanization and comments are excluded."""
    page_url: str | None
    script: str | None
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.text, self.spans, self.links, self.identity = '', [], [], {}
        self.depth, self.stack, self.ruby, self.in_reading = 0, [], None, False
        self.page_url, self.script, self.compositions = None, None, []
        self.search_depth = 0
        self.includes_translation = False
        self.feed(html)
        # The display title mixes artist and soundtrack credits. Use the
        # composition explicitly attached to this lyric page when available.
        for record in self.compositions:
            lyric = record.get('Lyrics', record.get('lyrics'))
            recording = record.get('recordedAs')
            if not isinstance(lyric, dict) or not isinstance(recording, dict):
                continue
            page = lyric.get('mainEntityOfPage')
            if not self.page_url or not isinstance(page, dict) or page.get('url') != self.page_url:  # ty: ignore[redundant-condition] # HTMLParser.feed populates page_url.
                continue
            artist = recording.get('byArtist')
            if not isinstance(artist, dict) or not all(isinstance(v, str) and v.strip()
                    for v in (record.get('name'), artist.get('name'))):
                continue
            aliases = lambda value: [s for s in (value if isinstance(value, list) else [value])
                                     if isinstance(s, str) and s.strip()]
            self.identity = dict(title=record['name'], artist=artist['name'],
                title_aliases=aliases(record.get('alternateName')),
                artist_aliases=aliases(artist.get('alternateName')))
            break

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'table' and (self.search_depth or 'searchResult' in classes):
            self.search_depth += 1
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.script = ''
        if tag == 'meta' and attrs.get('property') == 'og:url':
            self.page_url = attrs.get('content')
        if tag == 'meta' and attrs.get('property') == 'og:title':
            self.includes_translation = bool(re.search(r'【[^】]*和訳[^】]*】\s*ふりがな付', attrs.get('content', '')))
            match = re.fullmatch(r'(.+?) 歌詞 (.+?) ふりがな付(?: - .+)?', attrs.get('content', ''))
            if match:
                self.identity = dict(title=match[1].strip(), artist=match[2].strip())
        if self.search_depth and tag == 'a' and re.fullmatch(r'/lyric/[A-Za-z0-9]+/', attrs.get('href', '')):
            if attrs['href'] not in self.links:
                self.links.append(attrs['href'])
        if tag == 'div':
            if self.depth:
                self.depth += 1
            elif 'hiragana' in classes and not self.text:
                self.depth = 1
        if not self.depth:
            return
        if tag == 'br':
            self.text += '\n'
        elif tag == 'span':
            role = next((c for c in classes if c in ('ruby', 'rb', 'rt')), '')
            self.stack.append(role)
            if role == 'ruby':
                self.ruby = dict(first=len(self.text), reading='')
            elif role == 'rt':
                self.in_reading = True

    def handle_endtag(self, tag):
        if tag == 'table' and self.search_depth:
            self.search_depth -= 1
        if tag == 'script' and self.script is not None:
            try:
                data = json.loads(self.script)
                records = data if isinstance(data, list) else data.get('@graph', [data])
                self.compositions.extend(r for r in records
                    if isinstance(r, dict) and r.get('@type') == 'MusicComposition')
            except (ValueError, AttributeError, TypeError):
                pass
            self.script = None
        if not self.depth:
            return
        if tag == 'div':
            self.depth -= 1
        elif tag == 'span' and self.stack:
            role = self.stack.pop()
            if role == 'rt':
                self.in_reading = False
            elif role == 'ruby' and self.ruby is not None:
                self.spans.append(dict(self.ruby, stop=len(self.text)))
                self.ruby = None

    def handle_data(self, data):
        if self.script is not None:
            self.script += data
        if self.depth:
            if self.in_reading and self.ruby is not None:
                self.ruby['reading'] += data
            else:
                self.text += data


def text_positions(text):
    """Match typography without losing the original character boundaries."""
    normalized, positions = '', []
    for i, character in enumerate(text):
        for c in unicodedata.normalize('NFKC', character):
            if c.isalnum():
                normalized += c
                positions.append(i)
    return normalized, positions


def match_readings(lines, page):
    expected, positions = text_positions('\n'.join(lines))
    published, source_positions = text_positions(page.text)
    blocks = SequenceMatcher(None, published, expected, autojunk=False).get_matching_blocks()
    matched = sum(b.size for b in blocks)
    coverage = matched / max(1, len(expected), len(published))
    # A shared short phrase cannot establish which song these readings belong to.
    if not expected or coverage < .9:
        return None, coverage
    mapping = {b.a+i: b.b+i for b in blocks for i in range(b.size)}
    by_source = {raw: i for i, raw in enumerate(source_positions)}
    rows, owners = [], []
    for line in lines:
        rows.append(dict(text=line, spans=[]))
        owners.extend((len(rows)-1, i) for i in range(len(line)+1))
    for span in page.spans:
        reading = hiragana(span['reading'].strip())
        indexes = [by_source[i] for i in range(span['first'], span['stop']) if i in by_source]
        if any(i not in mapping for i in indexes):
            continue
        target = [mapping[i] for i in indexes]
        if (not target or target != list(range(target[0], target[0]+len(target)))
                or not re.fullmatch('[ぁ-ゖー]+', reading)):
            continue
        line, first = owners[positions[target[0]]]
        last_line, stop = owners[positions[target[-1]]]
        if line == last_line:
            rows[line]['spans'].append(dict(first=first, stop=stop+1,
                text=lines[line][first:stop+1], reading=reading))
    return rows, coverage


def read_page(url):
    import httpx
    with httpx.stream('GET', url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        response.raise_for_status()
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > 2*1024*1024:
                raise ValueError('Reading page exceeds 2 MiB')
    return body.decode('utf-8')


def prepare_readings(lines, source_url, cache, language='Japanese'):
    """Optional, cached presentation evidence; unavailable readings never lose lyrics."""
    from .lyric_lookup import cached_lyric_request, lyric_request_error
    path = cache.root/'lyric-readings.json'
    identity = dict(lines=lines, source_url=source_url)
    inline = []
    if language == 'Japanese':
        try:
            lyrics = json.loads((cache.root/'lyrics.json').read_text())
            if lyrics.get('source_url') == source_url:
                inline = lyrics.get('provenance', {}).get('lyric_lookup', {}).get('inline_readings', [])
        except (OSError, ValueError):
            pass
        if inline:
            identity['inline_readings'] = inline
    if language == 'Japanese':
        identity['parser_sha256'] = hashlib.sha256(inspect.getsource(ReadingPage).encode()).hexdigest()
    if language == 'Cantonese':
        from importlib.metadata import PackageNotFoundError, version

        from subtitle_style import cantonese_readings
        try:
            installed = version('pycantonese')
        except PackageNotFoundError:
            installed = None
        identity.update(language=language, dictionary='pycantonese', version=installed,
            mapper_sha256=hashlib.sha256(inspect.getsource(cantonese_readings).encode()).hexdigest())
    try:
        old = json.loads(path.read_text())
        if old['identity'] == identity and (old.get('error', {}).get('retry_at') or float('inf')) > time.time():
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    packet: dict[str, Any] = dict(identity=identity, status='unavailable', lines=[], attempts=[])
    try:
        if language == 'Cantonese':
            for text in lines:
                spans, warnings = cantonese_readings(text)
                packet['lines'].append(dict(text=text, spans=[dict(first=a, stop=b, text=text[a:b], reading=r)
                    for a, b, r in spans], warnings=list(warnings)))
            packet['status'] = 'dictionary'
        query = next((line.strip() for line in lines if re.search('[\u3400-\u9fff]', line)), '') if language == 'Japanese' else ''
        if query:
            url = 'https://utaten.com/search?'+urlencode(dict(body=query))
            fetch = lambda url: cached_lyric_request('utaten_readings', url, cache, lambda: read_page(url))
            listing = ReadingPage(fetch(url))
            for link in listing.links[:3]:
                url = 'https://utaten.com'+link
                html = fetch(url)
                rows, coverage = match_readings(lines, ReadingPage(html))
                packet['attempts'].append(dict(url=url, text_coverage=coverage))
                if rows and any(row['spans'] for row in rows):
                    packet.update(status='published', lines=rows, url=url,
                                  page_sha256=hashlib.sha256(html.encode()).hexdigest())
                    break
    except Exception as error:
        packet['error'] = lyric_request_error(error)
    if inline:
        supplied = {row['text']: row['spans'] for row in inline}
        if not packet['lines']:
            packet['lines'] = [dict(text=text, spans=[]) for text in lines]
        for row in packet['lines']:
            additions = supplied.get(row['text'], [])
            row['spans'] = sorted([*additions, *(s for s in row['spans'] if not any(
                s['first'] < a['stop'] and a['first'] < s['stop'] for a in additions))], key=lambda s:s['first'])
        if any(row['spans'] for row in packet['lines']):
            packet['status'] = 'published'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    return path


def resolve_readings(text, annotations=(), language='Japanese'):
    from subtitle_style import cantonese_readings, japanese_readings
    if language not in ('Japanese', 'Cantonese'):
        return (), ()
    fallback, warnings = (japanese_readings if language == 'Japanese' else cantonese_readings)(text)
    supplied = []
    for item in annotations:
        try:
            first, stop = item['first'], item['stop']
            reading = hiragana(item['reading']) if language == 'Japanese' else item['reading']
            pattern = '[ぁ-ゖー]+' if language == 'Japanese' else r'[a-z]+[1-6](?: [a-z]+[1-6])*'
            if (not isinstance(first, int) or not isinstance(stop, int) or not 0 <= first < stop <= len(text)
                    or text[first:stop] != item['text'] or not re.fullmatch(pattern, reading)
                    or supplied and first < supplied[-1][1]):
                raise ValueError('Reading does not match lyric span')
            supplied.append((first, stop, reading))
        except (KeyError, TypeError, ValueError):
            warnings += ('Ignored a published reading that does not match this lyric span',)
    return tuple(sorted([*supplied, *(s for s in fallback if not any(
        s[0] < stop and first < s[1] for first, stop, _ in supplied))])), warnings


def _same_passage(start, end, observed_start, observed_end):
    # Recognition times locate the occurrence. They never drive the sweep.
    return abs(start-observed_start) <= .5 and abs(end-observed_end) <= .5


def _spoken_variant(cue, words, offset):
    """Require an intact phrase with one independently heard replacement."""
    from subtitle_style import KANJI, japanese_readings
    text = cue['text']
    written, positions = text_positions(text)
    parts = [text_positions(w['word'])[0] for w in words]
    heard = ''.join(parts)
    edits = SequenceMatcher(None, written, heard, autojunk=False).get_opcodes()
    if [edit[0] for edit in edits] != ['equal', 'replace', 'equal']:
        return None
    _, left, right, begin, end = edits[1]
    if left < 2 or len(written)-right < 2:
        return None
    # Match the whole dictionary span and whole ASR words on either side
    # of the spelling change. A fragment of a compound has no such reading.
    first, stop = positions[left], positions[right-1]+1
    default = next((r for a, b, r in japanese_readings(text)[0] if (a, b) == (first, stop)), None)
    if not default or not re.fullmatch(KANJI, text[first:stop]):
        return None
    owners = [i for i, part in enumerate(parts) for _ in part]
    if owners[begin-1] == owners[begin] or owners[end-1] == owners[end]:
        return None
    spoken = words[owners[begin]:owners[end-1]+1]
    if not all(.9 <= (w.get('probability') or 0) <= 1 for w in spoken):
        return None
    units = [u for u in cue.get('units', []) if u['first'] < stop and first < u['stop']]
    start, finish = offset+spoken[0]['start'], offset+spoken[-1]['end']
    if (not units or units[0]['first'] != first or units[-1]['stop'] != stop
            or not _same_passage(units[0]['start'], units[-1]['end'], start, finish)):
        return None
    variant = heard[begin:end]
    if re.fullmatch(KANJI, variant):
        sentence = text[:first]+variant+text[stop:]
        reading = next((r for a, b, r in japanese_readings(sentence)[0]
                        if (a, b) == (first, first+len(variant))), None)
    else:
        reading = hiragana(variant)
    if not reading or reading == default or not re.fullmatch('[ぁ-ゖー]+', reading):
        return None
    return dict(first=first, stop=stop, text=text[first:stop], reading=reading,
                source='local_asr', recognized=variant, observed_range=[start, finish],
                probability=min(w['probability'] for w in spoken))


def _recording_readings(cues, blocks):
    # Forced alignment repeats the supplied lyrics. Only the unprompted
    # recognition windows can contain a different spelling of what was sung.
    views = {}
    for block in blocks:
        for window in block.get('asr_localizations', []):
            recognition = window.get('recognition') or {}
            if recognition.get('language') not in ('ja', 'Japanese'):
                continue
            offset, end = window['source_range']
            for segment in recognition.get('segments', []):
                words = segment.get('words')
                if (not words or not 0 <= segment.get('no_speech_prob', 0) < .6
                        or not all(0 <= w['start'] < w['end'] <= end-offset+.001 for w in words)
                        or any(a['end'] > b['start']+.001 for a, b in zip(words, words[1:]))):
                    continue
                start, finish = offset+words[0]['start'], offset+words[-1]['end']
                matches = [i for i, cue in enumerate(cues)
                           if _same_passage(cue['start'], cue['end'], start, finish)]
                if len(matches) != 1:
                    continue
                index = matches[0]
                span = _spoken_variant(cues[index], words, offset)
                key = (span['first'], span['stop'], span['reading']) if span else None
                views.setdefault(index, {})[key] = span
    # A competing reading, including the ordinary spelling, vetoes the change.
    return {i: next(iter(choices.values())) for i, choices in views.items()
            if len(choices) == 1 and None not in choices}


def load_readings(path, cues, language='Japanese', *, alignment=None):
    """Refuse stale annotations without changing any lyric or timing data."""
    try:
        packet = json.loads(path.read_text())
        if (packet['identity'].get('language', 'Japanese') != language
                or packet['identity']['lines'] != [cue['text'] for cue in cues] or not isinstance(packet['lines'], list)
                or not all(isinstance(row, dict) and isinstance(row.get('text'), str) and isinstance(row.get('spans'), list)
                           for row in packet['lines'])):
            return [], dict(status='stale_lyric_body')
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [], dict(status='unavailable')
    if language == 'Japanese' and alignment is not None:
        try:
            raw = alignment.read_bytes()
            variants = _recording_readings(cues, json.loads(raw))
            rows = packet['lines'] or [dict(text=cue['text'], spans=[]) for cue in cues]
            if variants:
                packet['alignment_sha256'] = hashlib.sha256(raw).hexdigest()
            for index, span in variants.items():
                row = rows[index]
                if row['text'] != cues[index]['text'] or any(
                        s['first'] < span['stop'] and span['first'] < s['stop'] for s in row['spans']):
                    continue
                row['spans'] = sorted([*row['spans'], span], key=lambda s: s['first'])
                packet['lines'] = rows
                if packet.get('status') == 'unavailable':
                    packet['status'] = 'local_asr'
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            packet['recognition_error'] = type(error).__name__
    return packet['lines'], packet
