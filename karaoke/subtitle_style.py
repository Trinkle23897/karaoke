"""Shared, transparent subtitle typography for previews and exported videos."""

import io
import re
import subprocess
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont

if __package__:
    from .subtitle_config import font_files, load_styles
else:
    from subtitle_config import font_files, load_styles

TOKENS = load_styles()
STYLES = TOKENS['styles']
DEFAULT_STYLE = TOKENS['default_style']
KANJI = r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff々〆]+'
WRAP_TOKEN = re.compile(r"[A-Za-z0-9’'\-]+|[^\S\n]+|.", re.DOTALL)


class CaptionOverflow(ValueError):
    pass


def contrast_scrim(picture, text_top, colors, minimum=4.5):
    """The least black shading that protects every pixel in the lyric region."""
    import numpy as np

    def luminance(rgb):
        value = np.asarray(rgb, dtype=np.float64)/255
        linear = np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)
        return linear @ np.array([.2126, .7152, .0722])

    text_top = max(0, min(picture.height-1, int(text_top)))
    pixels = np.asarray(picture.convert('RGB'), dtype=np.uint16)[text_top:]
    foreground = luminance([ImageColor.getrgb(color) for color in colors])

    def contrast(alpha):
        background = luminance((pixels*(255-alpha)+127)//255)
        return min(float(np.min((np.maximum(background, value)+.05)/(np.minimum(background, value)+.05)))
                   for value in foreground)

    before = contrast(0)
    report = dict(minimum=minimum, before=before, text_top=text_top, colors=list(colors))
    if before >= minimum:
        return None, dict(report, after=before, opacity=0., status='already_readable')
    if contrast(255) < minimum:
        return None, dict(report, status='unsupported_foreground')
    low, high = 1, 255
    while low < high:
        middle = (low+high)//2
        if contrast(middle) >= minimum:
            high = middle
        else:
            low = middle+1
    # Keep the upper artwork untouched; ease smoothly into a constant lyric backdrop.
    fade_start = max(0, text_top-round(picture.height/6))
    ramp = np.clip((np.arange(picture.height)-fade_start)/max(1, text_top-fade_start), 0, 1)
    alpha = np.rint(low*ramp*ramp*(3-2*ramp)).astype('uint8')
    scrim = Image.new('RGBA', picture.size)
    scrim.putalpha(Image.fromarray(np.repeat(alpha[:, None], picture.width, axis=1)))
    return scrim, dict(report, after=contrast(low), opacity=low/255,
                       fade_start=fade_start, status='shaded')


def static_video_scrim(video, output, text_top, colors):
    """Sample the original picture, never the rendered captions; keep motion intact."""
    import numpy as np
    started = time.perf_counter()
    report: dict[str, Any] = dict(status='unchecked', sample_times=[], max_mean_difference=0., max_pixel_difference=0)
    try:
        duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=nw=1:nk=1', str(video)], stderr=subprocess.PIPE, timeout=10))
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid picture duration')

        def frame(instant, width, height, raw):
            scale = (f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                     f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2')
            return subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
                '-ss', str(instant), '-i', str(video), '-frames:v', '1', '-an', '-sn',
                '-vf', scale, '-threads', '1', '-pix_fmt', 'rgb24',
                *(['-f', 'rawvideo'] if raw else ['-f', 'image2pipe', '-c:v', 'png']), '-'],
                stderr=subprocess.PIPE, timeout=10)

        reference = None
        for fraction in (.02, .09, .19, .32, .46, .61, .75, .88, .98):
            instant = min(fraction*duration, max(0., duration-.1))
            sampled = np.frombuffer(frame(instant, 128, 72, True), dtype=np.uint8).reshape(72, 128, 3)
            report['sample_times'].append(instant)
            if reference is not None:
                difference = np.abs(sampled.astype('int16')-reference)
                mean, maximum = float(difference.mean()), int(difference.max())
                report['max_mean_difference'] = max(report['max_mean_difference'], mean)
                report['max_pixel_difference'] = max(report['max_pixel_difference'], maximum)
                if mean > .75 or maximum > 16 or np.percentile(difference, 99.9) > 4:
                    return None, dict(report, status='moving_picture', seconds=time.perf_counter()-started)
            else:
                reference = sampled.astype('int16')
        picture = Image.open(io.BytesIO(frame(duration*.46, 1280, 720, False))).convert('RGB')
        scrim, contrast = contrast_scrim(picture, text_top, colors)
        report.update(status='sampled_static', contrast=contrast)
        if scrim is not None:
            scrim.save(output)
            return output, dict(report, seconds=time.perf_counter()-started)
    except Exception as error:
        report.update(status='unavailable', reason=f'{type(error).__name__}: {error}')
    return None, dict(report, seconds=time.perf_counter()-started)


@lru_cache(maxsize=4096)
def japanese_readings(text):
    """Dictionary readings of whole words, anchored to the unchanged lyric text."""
    if not re.search(KANJI, text):
        return (), ()
    def hira(value):
        return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c
                       for c in unicodedata.normalize('NFKC', value))

    try:
        from lyric_progress import japanese_tokenization
        normalized, positions, words = japanese_tokenization(text)
    except Exception as error:
        # A missing optional reading must never discard valid, timed lyrics.
        return (), (f'{type(error).__name__}: {error}',)
    spans, cursor = [], 0
    for word in words:
        written = word['string']
        first = normalized.find(written, cursor)
        if first < 0 or not written:
            continue
        cursor = first+len(written)
        if not re.search(KANJI, written):
            continue
        parts = list(re.finditer(KANJI+r'|[^\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff々〆]+', written))
        pattern = ''.join('([ぁ-ゖー]+?)' if re.fullmatch(KANJI, p[0]) else re.escape(hira(p[0]))
                          for p in parts)
        # "read" preserves spelling (きょう); "pron" can replace it with きょー.
        matched = re.fullmatch(pattern, hira(word.get('read', '')))
        if not matched:
            continue
        readings = iter(matched.groups())
        for part in parts:
            if re.fullmatch(KANJI, part[0]):
                a, b = positions[first+part.start()][0], positions[first+part.end()-1][1]
                reading = next(readings)
                if spans and a < spans[-1][1]:
                    previous = spans.pop()
                    spans.append((previous[0], max(previous[1], b), previous[2]+reading))
                else:
                    spans.append((a, b, reading))
    unresolved = tuple(f'No dictionary reading for {m[0]!r} at {m.start()}'
                       for m in re.finditer(KANJI, text)
                       if any(not any(a <= i < b for a, b, _ in spans)
                              for i in range(m.start(), m.end())))
    return tuple(spans), unresolved


@lru_cache(maxsize=4096)
def cantonese_readings(text):
    """Resolve whole-phrase Jyutping, then map syllables to the original characters."""
    if not re.search(KANJI, text):
        return (), ()
    try:
        import pycantonese
        from opencc import OpenCC
        normalized, positions = '', []
        for i, character in enumerate(text):
            # PyCantonese joins tokens across whitespace; keep that same input
            # while retaining offsets into the untouched lyric text.
            part = ''.join(c for c in unicodedata.normalize('NFKC', character) if not c.isspace())
            normalized += part
            positions.extend([i]*len(part))
        traditional = OpenCC('s2hk').convert(normalized)
        if len(traditional) != len(normalized):
            return (), ('Cantonese spelling conversion changed character boundaries',)
        words = pycantonese.characters_to_jyutping(traditional)
    except Exception as error:
        return (), (f'{type(error).__name__}: {error}',)
    spans, warnings, cursor = [], [], 0
    for written, reading in words:
        first = traditional.find(written, cursor)
        if first < 0 or not written:
            continue
        cursor = first+len(written)
        syllables = (reading or '').split()
        for i, character in enumerate(written):
            if not re.fullmatch(KANJI, character):
                continue
            index = positions[first+i]
            if len(syllables) == len(written) and re.fullmatch('[a-z]+[1-6]', syllables[i]):
                spans.append((index, index+1, syllables[i]))
            else:
                warnings.append(f'No Cantonese dictionary reading for {text[index]!r} at {index}')
    return tuple(spans), tuple(warnings)


def caption_height(language, style=DEFAULT_STYLE):
    height = STYLES[style].get('strip_height', TOKENS['strip_height'])
    return height+(TOKENS['furigana']['extra_height'] if language in ('Japanese', 'Cantonese') else 0)


def annotated_lines(text, readings, font, ruby_font, tracking, maximum, balanced):
    """Wrap whole reading groups, reserving their width instead of squeezing kana."""
    spans = {first: (stop, reading) for first, stop, reading in readings}
    items, cursor = [], 0
    while cursor < len(text):
        stop, reading = spans.get(cursor, (None, None))
        if stop is None:
            match = WRAP_TOKEN.match(text, cursor)
            assert match is not None  # The alternatives cover every character.
            stop = match.end()
        token = text[cursor:stop]
        width = max(line_width(token, font, tracking),
                    ruby_font.getlength(reading)+2*ruby_font.size*TOKENS['furigana']['side_padding_em'] if reading else 0)
        items.append(dict(text=token, first=cursor, stop=stop, reading=reading, width=width))
        cursor = stop

    def trim(row):
        while row and row[-1]['text'].isspace():
            row = row[:-1]
        while row and row[0]['text'].isspace():
            row = row[1:]
        return row

    def width(row):
        return sum(item['width'] for item in row)+max(0, len(row)-1)*tracking

    rows, row = [], []
    for item in items:
        if item['text'] == '\n':
            rows.append(trim(row))
            row = []
        elif row and width(row+[item]) > maximum:
            rows.append(trim(row))
            row = [] if item['text'].isspace() else [item]
        else:
            row.append(item)
    rows.append(trim(row))
    if balanced and len(rows) == 2 and '\n' not in text:
        candidates = []
        for i in range(1, len(items)):
            left, right = trim(items[:i]), trim(items[i:])
            if not left or not right or right[0]['text'][0] in '、。，．？！!?：:；;）)]」』ぁぃぅぇぉっゃゅょァィゥェォッャュョー':
                continue
            a, b = width(left), width(right)
            if max(a, b) <= maximum:
                semantic = items[i-1]['text'].isspace() or items[i]['text'].isspace() or items[i-1]['text'][-1] in '、，,。.!?？！;；'
                candidates.append((abs(a-b)+(0 if semantic else maximum*.15), i, left, right))
        if candidates:
            rows = list(min(candidates)[2:])
    return rows


@lru_cache(maxsize=32)
def system_font(name):
    path = Path('/System/Library/Fonts')/name
    if path.exists():
        return path
    # macOS stores some Japanese font filenames in decomposed Unicode.
    for candidate in path.parent.iterdir():
        if unicodedata.normalize('NFC', candidate.name) == path.name:
            return candidate
    raise FileNotFoundError(path)


def script_for(text, language='auto'):
    if language in ('Chinese', 'Japanese', 'Korean', 'Latin'):
        return language
    if re.search('[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]', text):
        return 'Korean'
    if re.search('[\u3040-\u30ff]', text):
        return 'Japanese'
    if re.search('[\u3400-\u9fff]', text):
        return 'Chinese'
    return 'Latin'


@lru_cache(maxsize=32)
def coverage(path, index):
    try:
        with TTFont(str(path), fontNumber=index, lazy=True) as font:
            # OpenType flag 14 marks last-resort glyphs, not real text coverage.
            if font['head'].flags & (1 << 14):  # ty: ignore[unresolved-attribute]  # fontTools loads the OpenType flags field dynamically.
                return set()
            return set(font.getBestCmap() or {})
    except TTLibError:
        # Fontconfig can return Type 1 fonts, which have no OpenType cmap.
        return set()


def font_inputs(style=DEFAULT_STYLE, font_file=None):
    return font_files(STYLES[style], font_file)


def resolve_font(text, style=DEFAULT_STYLE, language='auto', font_file=None):
    spec = STYLES[style]
    script = script_for(text, language)
    # Formatting controls (e.g. BOM / zero-width joiner) have no visible glyph.
    # Keep the source string and its timing indices intact.
    required = {ord(c) for c in text if not c.isspace() and unicodedata.category(c) != 'Cf'}
    for path in font_inputs(style, font_file):
        index = spec['asset_faces'].get(script, 0) if path.name == spec.get('font_asset') else 0
        if required <= coverage(path, index):
            return path, index, script
    name, index = spec['fonts'][script]
    try:
        path = system_font(name)
        if required <= coverage(path, index):
            return path, index, script
    except FileNotFoundError:
        pass
    # Select by actual glyph coverage, independently of the song's language.
    candidates = []
    try:
        family = spec.get('fallback_family', '')
        fonts = subprocess.check_output(['fc-match', '-s', '--format', '%{file}\t%{index}\n',
            family+':charset='+' '.join(f'{c:x}' for c in sorted(required))], text=True, timeout=5)
        candidates.extend((Path(path), int(index)) for path, index in
                          (line.rsplit('\t', 1) for line in fonts.splitlines()))
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    candidates.append((Path('/Library/Fonts/Arial Unicode.ttf'), 0))
    for path, index in candidates:
        if path.is_file() and required <= coverage(path, index):
            return path, index, script
    raise ValueError('No installed font covers all lyric characters: '+''.join(map(chr, sorted(required))))


def line_width(text, font, tracking):
    return sum(font.getlength(c) for c in text)+max(0, len(text)-1)*tracking


def wrap_lines(text, font, tracking, maximum):
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        # Keep Latin words intact; CJK can wrap between characters.
        for token in WRAP_TOKEN.findall(paragraph):
            if line and line_width(line+token, font, tracking) > maximum:
                lines.append(line.rstrip())
                line = token.lstrip()
            else:
                line += token
        lines.append(line.rstrip())
    return lines


def balanced_lines(text, font, tracking, maximum):
    lines = wrap_lines(text, font, tracking, maximum)
    if '\n' in text or len(lines) != 2:
        return lines
    candidates = []
    for index in range(1, len(text)):
        left, right = text[:index].rstrip(), text[index:].lstrip()
        if not left or not right or right[0] in '、。，．？！!?：:；;）)]」』ぁぃぅぇぉっゃゅょァィゥェォッャュョー':
            continue
        if text[index-1].isascii() and text[index-1].isalnum() and text[index].isascii() and text[index].isalnum():
            continue
        a, b = line_width(left, font, tracking), line_width(right, font, tracking)
        if max(a, b) <= maximum:
            semantic_break = text[index-1].isspace() or text[index].isspace() or text[index-1] in '、，,。.!?？！;；'
            candidates.append((abs(a-b)+(0 if semantic_break else maximum*.15), left, right))
    return list(min(candidates)[1:]) if candidates else lines


def outline_mask(mask, radius):
    """Dilate only the occupied region, including the filter's full halo."""
    outline = Image.new('L', mask.size)
    bounds = mask.getbbox()
    if bounds is not None:
        left, top, right, bottom = bounds
        bounds = (max(0, left-radius), max(0, top-radius),
                  min(mask.width, right+radius), min(mask.height, bottom+radius))
        outline.paste(mask.crop(bounds).filter(ImageFilter.MaxFilter(2*radius+1)), bounds)
    return outline


def caption_image(text, width=1280, height=None, font_file=None,
                  style=DEFAULT_STYLE, tone='light', language='auto', layout_out=None,
                  style_overrides=None, readings=()):
    """Return a transparent strip. No box, gradient, or background repainting."""
    scale = width/TOKENS['reference_width']
    height = height or round(caption_height(language, style)*scale)
    supersample = 2
    image = Image.new('RGBA', (width*supersample, height*supersample))
    if not text.strip():
        return image.resize((width, height))
    spec = {**STYLES[style], **(style_overrides or {})}
    path, index, script = resolve_font(text, style, language, font_file)
    font_scale = scale*supersample
    tracking_em = spec['latin_tracking_em'] if script == 'Latin' else spec['tracking_em']
    margin = image.width*spec.get('safe_x', TOKENS['safe_x'])
    maximum = image.width-2*margin
    maximum_lines = spec.get('maximum_lines', 2)
    sizes = [(size, maximum_lines == 1)
             for size in range(spec['size'], spec.get('minimum_size', TOKENS['minimum_font_size'])-1, -2)]
    if maximum_lines > 1 and 'single_line_floor' in spec:
        sizes = [(size, True) for size in range(spec['size'], spec['single_line_floor']-1, -2)]+sizes
    for size, require_single in sizes:
        font = ImageFont.truetype(str(path), round(size*font_scale), index=index)
        tracking = size*font_scale*tracking_em
        balanced = spec.get('balanced_wrap') and not require_single
        if readings:
            ruby_size = spec.get('reading_size_em', TOKENS['furigana']['size_em'])
            ruby_font = font.font_variant(size=max(1, round(size*font_scale*ruby_size)))
            rows = annotated_lines(text, readings, font, ruby_font, tracking, maximum, balanced)
            lines = [''.join(item['text'] for item in row) for row in rows]
            widths = [sum(item['width'] for item in row)+max(0, len(row)-1)*tracking for row in rows]
        else:
            lines = (balanced_lines if balanced else wrap_lines)(text, font, tracking, maximum)
            widths = [line_width(line, font, tracking) for line in lines]
        if require_single and len(lines) > 1:
            continue
        if len(lines) <= maximum_lines and all(length <= maximum for length in widths):
            break
    else:
        raise CaptionOverflow('Caption exceeds the readable line limit; split the lyric cue')
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)
    ruby_mask = Image.new('L', image.size)
    ruby_draw = ImageDraw.Draw(ruby_mask)
    if spec.get('rim_width'):
        edge = Image.new('L', image.size)
        rim = Image.new('L', image.size)
        edge_draw, rim_draw = ImageDraw.Draw(edge), ImageDraw.Draw(rim)

    def draw_ink(pen, position, value, face):
        weight = 0
        if spec.get('rim_width'):
            # A single dark contour keeps small reading marks legible when scaled down.
            weight = max(1, round(face.size*spec['weight_em']))
            factor = face.size/STYLES[style]['size']
            inner = weight+max(1, round(factor*spec['outline_width']))
            outer = inner+max(1, round(factor*spec['rim_width']))
            edge_draw.text(position, value, font=face, fill=255, anchor='ls',
                           stroke_width=outer if pen is ruby_draw else inner)
            rim_draw.text(position, value, font=face, fill=255, anchor='ls', stroke_width=outer)
        pen.text(position, value, font=face, fill=255, anchor='ls', stroke_width=weight)
        return pen.textbbox(position, value, font=face, anchor='ls', stroke_width=weight)

    baseline = (height-spec.get('baseline_inset', 36)*scale)*supersample
    characters, ruby_boxes = [], []
    cursor = 0
    line_height = size*font_scale*spec['line_height']
    if readings:
        body = [font.getbbox(line, anchor='ls') for line in lines]
        ruby = [ruby_font.getbbox(item['reading'], anchor='ls') for row in rows for item in row if item['reading']]
        if 'reading_gap' in spec:
            ruby_offset = max(r[3] for r in ruby)-min(b[1] for b in body)+spec['reading_gap']*font_scale
        else:
            ruby_offset = size*font_scale+TOKENS['furigana']['gap']*font_scale+ruby_font.getmetrics()[1]
        body_bottom = max(b[3] for b in body)
        ruby_top = min(r[1] for r in ruby)
        line_height = max(line_height+ruby_font.size, body_bottom+ruby_offset-ruby_top)
        line_height += TOKENS['furigana']['line_gap']*font_scale
    for i, line in enumerate(lines):
        length = widths[i]
        x = {'left': margin, 'center': (image.width-length)/2,
             'right': image.width-margin-length}[spec['alignment']]
        y = baseline-(len(lines)-i-1)*line_height
        offset = text.find(line, cursor)
        if offset < 0:
            raise ValueError('Wrapped caption no longer matches its source text')
        row: list[dict[str, Any]] = rows[i] if readings else [dict(text=line, first=offset, width=length, reading=None)]
        for item in row:
            glyph_x = x+(item['width']-line_width(item['text'], font, tracking))/2
            for j, character in enumerate(item['text']):
                position = (round(glyph_x), round(y))
                bounds = draw_ink(draw, position, character, font)
                characters.append({'index': item['first']+j, 'line': i, 'character': character,
                    'left': min(glyph_x, bounds[0])/supersample,
                    'right': max(glyph_x+font.getlength(character), bounds[2])/supersample,
                    'top': bounds[1]/supersample, 'bottom': bounds[3]/supersample,
                    'baseline': y/supersample})
                glyph_x += font.getlength(character)+tracking
            if item['reading']:
                bounds = ruby_font.getbbox(item['reading'], anchor='ls')
                position = (round(x+item['width']/2-(bounds[0]+bounds[2])/2),
                            round(y-ruby_offset))
                box = draw_ink(ruby_draw, position, item['reading'], ruby_font)
                ruby_boxes.append(dict(first=item['first'], stop=item['stop'], reading=item['reading'], line=i,
                                       bounds=tuple(value/supersample for value in box),
                                       baseline=position[1]/supersample))
            x += item['width']+tracking
        cursor = offset+len(line)
    visible_mask = ImageChops.lighter(mask, ruby_mask)
    if tone not in ('light', 'dark'):
        raise ValueError('tone must be light or dark')
    color = spec['color'] if tone == 'light' else '#172028'
    blur = ImageFilter.GaussianBlur(spec.get('shadow_blur', 2.4)*font_scale)
    if spec.get('rim_width'):
        glow = rim.filter(blur).point(lambda a: min(255, round(a*1.8)))
        layers = [(spec['progress_color'], glow), (color, rim),
                  ('#22202a' if tone == 'light' else '#ffffff', edge)]
    else:
        contrast = '#000000' if tone == 'light' else '#ffffff'
        edge_radius = max(1, round(spec.get('outline_width', .5)*font_scale))
        layers = [(contrast, visible_mask.filter(blur).point(lambda a: round(a*spec.get('shadow_alpha', .6)))),
                  (contrast, outline_mask(visible_mask, edge_radius).point(lambda a: round(a*spec.get('outline_alpha', .34))))]
    for paint, alpha in [*layers, (color, visible_mask)]:
        layer = Image.new('RGBA', image.size, paint)
        layer.putalpha(alpha)
        image.alpha_composite(layer)
    if layout_out is not None:
        layout_out.update(mask=mask.resize((width, height), Image.Resampling.LANCZOS),
                          characters=characters, font_size=size, lines=lines, ruby=ruby_boxes,
                          font=' '.join(part for part in font.getname() if part),
                          wrap_height=round((len(lines)-1)*line_height/supersample),
                          content_top=min([c['top'] for c in characters]+[r['bounds'][1] for r in ruby_boxes]))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def with_opacity(image, opacity):
    result = image.copy()
    result.putalpha(image.getchannel('A').point(lambda a: round(a*opacity)))
    return result


def compact_caption_image(text, **options):
    """Fit auxiliary text; the current lyric caption always uses caption_image."""
    try:
        return caption_image(text, **options)
    except CaptionOverflow:
        # Prefer a whole last word. Even a single oversized title can be abbreviated.
        stops = [m.end() for m in WRAP_TOKEN.finditer(text)]
        stops = list(range(1, stops[0])) + stops[:-1]
        low, high = 0, len(stops)-1
        row = caption_image('…', **dict(options, readings=()))
        while low <= high:
            middle = (low+high)//2
            candidate_text = text[:stops[middle]].rstrip()+'…'
            candidate_options = dict(options)
            if options.get('readings'):
                candidate_options['readings'] = tuple(
                    (a,b,kana) for a,b,kana in options['readings'] if b <= len(candidate_text)-1)
            try:
                candidate = caption_image(candidate_text, **candidate_options)
            except CaptionOverflow:
                high = middle-1
            else:
                row, low = candidate, middle+1
        return row


def song_intro_image(info, width=1280, style=DEFAULT_STYLE):
    """Transparent song credits, using the same typefaces as the lyrics."""
    title = info.get('title', '').strip()
    if not title:
        raise ValueError('Song information requires a title')
    spec = TOKENS['intro']
    scale = width/TOKENS['reference_width']
    image = Image.new('RGBA', (width, round(spec['height']*scale)))
    rows = [(title, 0, 240, spec['title_size'], spec['title_minimum_size'], 40, 2)]
    artist = (info.get('artist') or info.get('original_artist') or '').strip()
    if artist:
        rows.append((f'演唱 · {artist}', 242, 72, spec['artist_size'], 24, 22, 1))
    version = info.get('version', '').strip()
    if version:
        rows.append((version, 310, 52, spec['version_size'], 20, 18, 1))
    for text, top, height, size, minimum, inset, lines in rows:
        row = compact_caption_image(text, width=width, height=round(height*scale), style=style,
            style_overrides={'size': size, 'minimum_size': minimum, 'single_line_floor': minimum,
                'maximum_lines': lines, 'baseline_inset': inset, 'alignment': 'center',
                'balanced_wrap': True})
        image.alpha_composite(row, (0, round(top*scale)))
    color = ImageColor.getrgb(STYLES[style].get('progress_color', '#e59a24'))+(255,)
    ImageDraw.Draw(image).rectangle((round(width/2-24*scale), round(229*scale),
                                    round(width/2+24*scale), round(230*scale)), fill=color)
    return image


def cue_opacity(elapsed, duration, style=DEFAULT_STYLE):
    if not 0 <= elapsed < duration:
        return 0.
    spec = STYLES[style]
    ingress = min(spec['fade_in_ms']/1000, duration/4)
    egress = min(spec['fade_out_ms']/1000, duration/4)
    return max(0., min(1., elapsed/ingress if ingress else 1.,
                       (duration-elapsed)/egress if egress else 1.))
