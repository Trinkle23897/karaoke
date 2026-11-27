"""Subtitle settings shared by the renderer and the lightweight jobs service."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_styles():
    return json.loads((ROOT/'subtitle-styles.json').read_text())


def font_files(spec, font_file=None):
    """Files shared by font selection and render checkpoints, in priority order."""
    override = font_file or os.getenv('KARAOKE_FONT_FILE')
    paths = [Path(override).resolve()] if override else []
    asset = spec.get('font_asset')
    if asset:
        path = (ROOT/'models'/'fonts'/asset).resolve()
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths
