"""Render line captions with Pillow and FFmpeg, without requiring libass."""

import argparse
import json
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from subtitle_style import (
    DEFAULT_STYLE,
    STYLES,
    TOKENS,
    caption_height,
    caption_image,
    cue_opacity,
    japanese_readings,
    resolve_font,
    script_for,
    with_opacity,
)


def video_frame_rate(video):
    """Keep the source cadence, including fractional broadcast frame rates."""
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,r_frame_rate', '-of', 'json', str(video)]))
    stream = next(iter(probe.get('streams', [])), {})
    for field in ('avg_frame_rate', 'r_frame_rate'):
        try:
            rate = Fraction(stream.get(field, '0/0'))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return Fraction(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    background = parser.add_mutually_exclusive_group(required=True)
    background.add_argument('--video', type=Path)
    background.add_argument('--title', help='Use a title card when the supplied source is audio only')
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--cues', required=True, type=Path)
    parser.add_argument('--duration', required=True, type=float)
    parser.add_argument('--gain', type=float, default=1.)
    parser.add_argument('--font', type=Path, help='Font covering the lyric language')
    parser.add_argument('--style', choices=STYLES, default=DEFAULT_STYLE)
    parser.add_argument('--tone', choices=['light', 'dark'], default='light')
    parser.add_argument('--language', default='auto')
    parser.add_argument('--animate', action='store_true', help='Apply the style fade at cue boundaries')
    parser.add_argument('--bottom-margin', type=int, default=TOKENS['bottom_margin'],
                        help='Caption strip offset in output pixels; adjust for existing picture titles')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cues = json.loads(args.cues.read_text())
    if any(c['status'] != 'candidate' for c in cues):
        raise ValueError('Resolve invalid captions before producing a complete preview')
    folder = args.output.parent/(args.output.stem+'-caption-images')
    folder.mkdir(parents=True)
    blank = folder/'blank.png'
    height = max((caption_height(script_for(cue['text'], args.language), args.style) for cue in cues),
                 default=caption_height(args.language, args.style))
    styling = dict(font_file=args.font, style=args.style, tone=args.tone, language=args.language, height=height)
    caption_image('', **styling).save(blank)
    entries = []
    cursor = 0.
    for i, cue in enumerate(cues):
        start, end = cue['start'], cue['end']
        if start < cursor-.00001 or end <= start or end > args.duration+.001:
            raise ValueError(f'Overlapping or invalid cue: {i+1}')
        if start > cursor:
            entries.append((blank, start-cursor))
        path = folder/f'{i:04d}.png'
        readings = japanese_readings(cue['text'])[0] if script_for(cue['text'], args.language) == 'Japanese' else ()
        picture = caption_image(cue['text'], **styling, readings=readings)
        picture.save(path)
        duration = end-start
        if args.animate:
            step = 1/30
            fade_in = min(STYLES[args.style]['fade_in_ms']/1000, duration/4)
            fade_out = min(STYLES[args.style]['fade_out_ms']/1000, duration/4)
            boundaries = sorted({0., duration, fade_in, duration-fade_out} |
                {min(fade_in, k*step) for k in range(1, 7)} |
                {max(duration-fade_out, duration-k*step) for k in range(1, 5)})
            for j, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
                opacity = cue_opacity((a+b)/2, duration, args.style)
                frame = path if opacity >= 1 else folder/f'{i:04d}-{j:02d}.png'
                if frame != path:
                    with_opacity(picture, opacity).save(frame)
                entries.append((frame, b-a))
        else:
            entries.append((path, duration))
        cursor = end
    if cursor < args.duration:
        entries.append((blank, args.duration-cursor))
    concat = folder/'frames.ffconcat'
    script = ['ffconcat version 1.0']
    for path, duration in entries:
        script.extend([f"file '{path.name}'", 'option framerate 1000', f'duration {duration:.6f}'])
    script.extend(["file 'blank.png'", 'option framerate 1000'])
    concat.write_text('\n'.join(script)+'\n')
    if args.video:
        video_input = ['-i', str(args.video)]
        fps = video_frame_rate(args.video)
    else:
        fps = Fraction(30)
        still = folder/'background.png'
        picture = Image.new('RGB', (1280, 720), '#11141b')
        title_font, index, _ = resolve_font(args.title, args.style, args.language, args.font)
        ImageDraw.Draw(picture).text((640, 285), args.title,
            font=ImageFont.truetype(str(title_font), 64, index=index), fill='#f5f1e9', anchor='mm')
        picture.save(still)
        video_input = ['-loop', '1', '-framerate', '30', '-i', str(still)]
    command = ['ffmpeg', '-v', 'error', '-nostdin']+video_input+[
        '-f', 'concat', '-safe', '0', '-i', str(concat), '-i', str(args.audio),
        '-filter_complex', f'[0:v]scale=1280:-2,fps={fps},setpts=PTS-STARTPTS[base];'
        f'[1:v]fps={fps},setpts=PTS-STARTPTS,format=rgba[text];'
        f'[base][text]overlay=0:H-h-{args.bottom_margin}:eof_action=repeat:format=auto,format=yuv420p[v]',
        '-map', '[v]', '-map', '2:a:0', '-af', f'volume={args.gain:.12g},asetpts=PTS-STARTPTS',
        '-t', str(args.duration), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-threads', '4', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(args.output)]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter()-started
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(args.output), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height', '-of', 'json', str(args.output)]))
    if abs(float(probe['format']['duration'])-args.duration) > .1:
        raise ValueError('Rendered duration differs from the complete recording')
    metadata = {'command': command, 'render_seconds': elapsed, 'decode_check': 'passed',
                'frame_rate': str(fps),
                'duration_seconds': args.duration, 'caption_count': len(cues), 'probe': probe,
                'style': args.style, 'tone': args.tone, 'language': args.language,
                'animated': args.animate, 'style_version': TOKENS['version'],
                'bottom_margin': args.bottom_margin}
    args.output.with_suffix('.render.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
