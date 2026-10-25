"""Map aligned text units back to source lyric lines and write timed captions."""

import json
import math
import re
import unicodedata
from bisect import bisect_right
from pathlib import Path
from typing import Any

LATIN_LOOKALIKES = str.maketrans('АВЕКМНОРСТХаеорсхуіј', 'ABEKMHOPCTXaeopcxyij')


def latin_word_text(text):
    """Use Latin lookalikes for mixed words without changing source offsets."""
    if not text.isascii() and any(ord(c) in LATIN_LOOKALIKES for c in text):
        def latin_token(match):
            token = match[0]
            candidate = token.translate(LATIN_LOOKALIKES)
            latin = lambda c: unicodedata.name(c, '').startswith('LATIN ')
            return candidate if any(map(latin, token)) and all(latin(c) or c in "'’" for c in candidate) else token
        text = re.sub(r"[^\W\d_]+(?:['’][^\W\d_]+)*", latin_token, text)
    return text


def normalized(text):
    return ''.join(c for c in latin_word_text(unicodedata.normalize('NFKC', text))
                   if unicodedata.category(c)[0] in ('L', 'N') or c == "'")


def highlight_spans(text, language=None):
    """Display atoms: Unicode words and Chinese characters, in source offsets.

    Japanese reading groups are chosen by the pronunciation model, not here.
    Apostrophes and hyphens inside a Latin word do not create another onset.
    """
    text = latin_word_text(text)
    def latin(char):
        return unicodedata.name(char, '').startswith('LATIN ') or char.isdecimal()

    def word_letter(char):
        return char.isalpha() and not unicodedata.name(char, '').startswith(
            ('CJK ', 'HIRAGANA ', 'KATAKANA ', 'HALFWIDTH KATAKANA '))

    index = 0
    while index < len(text):
        first = index
        char = text[index]
        index += 1
        if latin(char):
            while index < len(text):
                char = text[index]
                if latin(char) or unicodedata.category(char).startswith('M'):
                    index += 1
                elif char in "'’‐-" and index+1 < len(text) and latin(text[index+1]):
                    index += 1
                else:
                    break
            yield first, index
        elif language in ('Chinese', 'Cantonese') and unicodedata.name(char, '').startswith(
                ('CJK UNIFIED IDEOGRAPH', 'CJK COMPATIBILITY IDEOGRAPH')):
            yield first, index
        elif word_letter(char):
            while index < len(text):
                char = text[index]
                if word_letter(char) or char.isdecimal() or unicodedata.category(char).startswith('M'):
                    index += 1
                elif char in "'’‐-" and index+1 < len(text) and text[index+1].isalpha():
                    index += 1
                else:
                    break
            yield first, index


def latin_word_spans(text):
    """Words representable by the Latin phonetic adapters, not display policy."""
    text = latin_word_text(text)
    for first, stop in highlight_spans(text):
        if all(not c.isalpha() or unicodedata.name(c, '').startswith('LATIN ')
               for c in text[first:stop]):
            yield first, stop


def highlight_coverage_errors(cues, language):
    """Reject merged/missing/split display atoms, independent of timing accuracy."""
    errors = []
    for line, cue in enumerate(cues, 1):
        text, units = cue['text'], cue.get('units', [])
        if any(u.get('timing_source') == 'linear_fallback' for u in units):
            errors.append(dict(line=line, reason='estimated playback timing'))
        if any(not isinstance(u.get('first'), int) or not isinstance(u.get('stop'), int)
               or not 0 <= u['first'] < u['stop'] <= len(text)
               or u['text'] != text[u['first']:u['stop']] for u in units):
            errors.append(dict(line=line, reason='invalid display span'))
            continue
        for first, stop in highlight_spans(text, language):
            covering = [u for u in units if u['first'] < stop and u['stop'] > first]
            if len(covering) == 1:
                unit = covering[0]
                outside = text[unit['first']:first]+text[stop:unit['stop']]
                if unit['first'] <= first and unit['stop'] >= stop and not any(
                        char.isalnum() for char in outside):
                    continue
            errors.append(dict(line=line, first=first, stop=stop, text=text[first:stop],
                               reason='word or character lacks an independent highlight'))
    return errors


def collapsed_phrase(cue, language):
    """Detect gross phrase compression, not normal variation in singing speed."""
    text, atoms = latin_word_text(cue['text']), []
    for first, stop in highlight_spans(text, language):
        # Keep the existing non-Latin spelling measure when a whole word
        # becomes one highlight; changing its display must not hide collapse.
        if any(c.isalpha() and not unicodedata.name(c, '').startswith('LATIN ') for c in text[first:stop]):
            atoms.extend((i,i+1) for i in range(first,stop) if text[i].isalnum())
        else:
            atoms.append((first,stop))
    # Display tokenization intentionally leaves Japanese reading groups to
    # the acoustic model. Compression still has to cover those scripts,
    # including a failed group that swallowed a complete sentence.
    covered = {i for first, stop in atoms for i in range(first, stop)}
    atoms.extend((i, i+1) for i, char in enumerate(cue['text'])
                 if i not in covered and unicodedata.category(char)[0] in ('L', 'N'))
    spans: list[dict[str, Any]] = [dict(first=0, stop=len(cue['text']),
                  start=cue.get('aligned_start', cue.get('start')),
                  end=cue.get('aligned_end', cue.get('end'))), *(cue.get('units') or [])]
    for span in spans:
        if not isinstance(span, dict):
            continue
        first, stop, start, end = (span.get(k) for k in ('first', 'stop', 'start', 'end'))
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (first, stop, start, end))):
            continue
        count = sum(first <= a and b <= stop for a, b in atoms)
        # More than 25 complete words/characters per second is a collapsed
        # alignment, even if a tiny positive interval survived token merging.
        if count >= 3 and end-start < .04*count:
            return True
    return False


def playback_units(cue, language):
    """Preserve usable onsets and sweep only unresolved spans between known bounds."""
    if collapsed_phrase(cue, language):
        raise ValueError('Collapsed lyric phrase timing')
    text = cue['text']
    begin, end = cue.get('aligned_start', cue['start']), cue.get('aligned_end', cue['end'])
    if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (begin, end)) or not (
            cue['start'] <= begin < end <= cue['end']):
        begin, end = cue['start'], cue['end']

    def sweep(first, stop, start, finish, source):
        return dict(text=text[first:stop], first=first, stop=stop, start=start, end=finish,
                    timing_source='linear_fallback', bounds_source=source)

    known = []
    atoms = list(highlight_spans(text, language))
    for unit in cue.get('units') or []:
        if not isinstance(unit, dict):
            continue
        first, stop, start, finish = (unit.get(k) for k in ('first', 'stop', 'start', 'end'))
        if (not isinstance(first, int) or not isinstance(stop, int) or
                not 0 <= first < stop <= len(text) or unit.get('text') != text[first:stop] or
                not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, finish)) or
                not begin <= start < finish <= end or
                any(a < first < b or a < stop < b for a, b in atoms) or
                known and (first < known[-1]['stop'] or start <= known[-1]['start'])):
            continue
        single = dict(text=unit['text'], units=[dict(unit, first=0, stop=stop-first)])
        known.append(sweep(first, stop, start, finish, 'word_group')
                     if highlight_coverage_errors([single], language) else dict(unit))

    result, cursor = [], 0
    # A sentinel lets the final missing span use the measured phrase end.
    for unit in [*known, dict(first=len(text), stop=len(text), start=end, end=end)]:
        if normalized(text[cursor:unit['first']]):
            left = result[-1]['end'] if result else begin
            right = unit['start']
            if left >= right and result:
                # Overlapping word tails leave no interval: expand this fallback
                # by one group, keeping its first onset and the next good onset.
                previous = result.pop()
                cursor, left = previous['first'], previous['start']
            if left < right:
                result.append(sweep(cursor, unit['first'], left, right, 'neighbor_or_phrase'))
            else:
                # A missing prefix shares the first measured onset. Only this
                # first group must join the sweep, not the rest of the line.
                unit = sweep(cursor, unit['stop'], begin, unit['end'], 'neighbor_or_phrase')
        if unit['first'] < unit['stop']:
            result.append(unit)
        cursor = unit['stop']
    return result


def lyric_cues(lines, units, duration):
    expected = ''.join(normalized(line) for line in lines)
    observed = ''.join(normalized(unit['text']) for unit in units)
    if expected != observed:
        raise ValueError('Aligned units do not exactly cover the supplied lyric text')
    spans = []
    position = 0
    for unit in units:
        length = len(normalized(unit['text']))
        if length:
            spans.append((position, position+length, unit))
        position += length
    cues = []
    position = 0
    for line in lines:
        length = len(normalized(line))
        if not length:
            continue
        matched = [unit for begin, end, unit in spans if begin < position+length and end > position]
        start = min(u['start'] for u in matched)
        end = max(u['end'] for u in matched)
        valid = 0 <= start < end <= duration+.001
        cues.append({'text': line, 'start': start, 'end': end,
                     'status': 'candidate' if valid else 'unresolved_timing'})
        position += length
    return cues


def presentation_cues(cues, duration, lead=.2, tail=.4, bridge_gap=1.5):
    """Make readable line holds without changing the recorded acoustic timings."""
    adjusted = [dict(cue) for cue in cues]
    changes = []
    for i, cue in enumerate(adjusted):
        if cue['status'] != 'candidate' or not 0 <= cue['start'] < cue['end'] <= duration+.001:
            raise ValueError(f'Unresolved alignment on line {i+1}')
        cue['aligned_start'], cue['aligned_end'] = cue['start'], cue['end']
    for i, (left, right) in enumerate(zip(adjusted, adjusted[1:])):
        if not left['start'] < right['start'] or (left.get('units') and right.get('units') and
                left['units'][-1]['start'] >= right['units'][0]['start']):
            raise ValueError(f'Unordered alignment at line {i+1}')
        overlap = left['end']-right['start']
        if overlap > 0:
            # A held note can outlive the next line's entrance. End its display
            # hold there; neither acoustic boundary nor the next onset moves.
            boundary = right['start']
            left['end'] = boundary
            changes.append({'lines': [i+1, i+2], 'overlap_seconds': round(overlap, 3), 'boundary': boundary})
    starts = [max(0., cue['start']-lead) for cue in adjusted]
    for i, cue in enumerate(adjusted):
        end = min(duration, cue['end']+tail)
        if i+1 < len(adjusted):
            next_start = starts[i+1]
            end = next_start if next_start-cue['end'] <= bridge_gap else min(end, next_start)
        cue['start'], cue['end'] = round(starts[i], 3), round(end, 3)
        if cue['end'] <= cue['start']:
            raise ValueError(f'No display interval for line {i+1}')
    return adjusted, {'lead_seconds': lead, 'tail_seconds': tail,
                      'bridge_gap_seconds': bridge_gap, 'overlap_adjustments': changes}


def read_ahead_state(cues, instant):
    """Keep the current and next lines visible; advance only at sung onsets."""
    if not cues:
        return None, None
    starts = [c['units'][0]['start'] if c.get('units')
              else c.get('aligned_start', c['start']) for c in cues]
    current = max(0, bisect_right(starts, instant)-1)
    following = current+1
    return current, following if following < len(cues) else None


def acoustic_end(unit):
    """Display holds and CTC pulses do not measure the end of singing."""
    if 'acoustic_end' in unit:
        return unit['acoustic_end']
    if (unit.get('timing_evidence') == 'ctc_event' or unit.get('display_end_source')
            or unit.get('timing_source') == 'linear_fallback'):
        return None
    return unit['end']


def countdown_events(cues, minimum_silence=4., dots=3, step_seconds=1., settle_seconds=1.):
    """Prepare the opening and long vocal gaps; a negative start needs playback pre-roll."""
    duration = dots*step_seconds+settle_seconds
    if dots < 1 or step_seconds <= 0 or settle_seconds < 0 or minimum_silence < duration:
        raise ValueError('Countdown must fit inside the minimum vocal gap')
    events = []
    previous_end = 0.
    last_end_known = True
    for line, cue in enumerate(cues):
        units = cue.get('units') or [{'start': cue.get('aligned_start', cue['start']),
                                     'end': cue.get('aligned_end', cue['end'])}]
        for index, unit in enumerate(units):
            onset, end = unit['start'], acoustic_end(unit)
            if not (math.isfinite(onset) and onset >= 0 and (end is None or
                    math.isfinite(end) and onset < end)):
                raise ValueError('Countdown requires valid acoustic intervals')
            if (line == 0 and index == 0) or last_end_known and onset-previous_end >= minimum_silence:
                events.append(dict(line=line, unit=index, start=onset-duration,
                                   onset=onset, previous_end=previous_end,
                                   dots=dots, step_seconds=step_seconds, settle_seconds=settle_seconds))
            last_end_known = end is not None
            if end is not None:
                previous_end = max(previous_end, end)
    return events


def countdown_state(events, instant):
    """Stateless countdown: seeking and pausing always preserve the audio clock."""
    index = bisect_right([event['start'] for event in events], instant)-1
    if index < 0 or instant >= events[index]['onset']:
        return None, 0
    event = events[index]
    remaining = math.ceil((event['onset']-event.get('settle_seconds', 0.)-instant)/event['step_seconds']-1e-9)
    return event, max(0, min(event['dots'], remaining))


def timestamp(seconds, decimal=','):
    value = round(seconds*1000)
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    seconds, milliseconds = divmod(value, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}{decimal}{milliseconds:03}'


def write_captions(cues, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'cues.json').write_text(json.dumps(cues, ensure_ascii=False, indent=2)+'\n')
    valid = [cue for cue in cues if cue['status'] == 'candidate']
    for extension, decimal, header in [('srt', ',', ''), ('vtt', '.', 'WEBVTT\n\n')]:
        body = '\n\n'.join(f"{i}\n{timestamp(c['start'], decimal)} --> "
                           f"{timestamp(c['end'], decimal)}\n{c['text']}"
                           for i, c in enumerate(valid, 1))
        (directory/f'lyrics.{extension}').write_text(header+body+'\n')
    return {'total_lines': len(cues), 'exported_lines': len(valid),
            'unresolved_lines': [i+1 for i, c in enumerate(cues) if c['status'] != 'candidate'],
            'overlap_pairs': [[i, i+1] for i in range(1, len(valid))
                              if valid[i-1]['end'] > valid[i]['start']]}
