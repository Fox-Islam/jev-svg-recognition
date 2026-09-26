"""An SVG rendered to pixels and written as a grid of characters, the state Jev is given in the pixel-grid runs.

The grids depend on the renderer, so the manifest records a sha1 of each, and scripts/fetch.py checks them.
"""

import io
import re

import numpy as np

from .svg import NS, prepare

PALETTE = {
    '.': ('white', (255, 255, 255)),
    'k': ('black', (20, 20, 20)),
    'g': ('grey', (128, 128, 128)),
    'r': ('red', (200, 30, 30)),
    'o': ('orange', (240, 140, 30)),
    'y': ('yellow', (240, 210, 40)),
    'n': ('green', (40, 150, 60)),
    'b': ('blue', (40, 90, 200)),
    'p': ('purple', (130, 60, 160)),
    'i': ('pink', (240, 150, 180)),
    'w': ('brown', (120, 70, 35)),
    's': ('beige', (225, 195, 160)),
}
KEYS = list(PALETTE)
COLOURS = np.array([PALETTE[k][1] for k in KEYS], float)
RAMP = '.:-=+*#%@'  # light to dark: the white background is '.', solid ink '@'


def render(svg_text, size):
    """RGB array of size x size: drawn at 4x on white, then reduced with a Lanczos filter."""
    import resvg_py
    from PIL import Image

    s = svg_text if 'xmlns=' in svg_text[:300] else svg_text.replace('<svg', NS, 1)
    s = re.sub(r'\s(width|height)="[^"]*"', '', s[: s.index('>')]) + s[s.index('>') :]
    s = s.replace('<svg', f'<svg width="{size * 4}" height="{size * 4}"', 1)
    try:
        im = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=s)))).convert('RGBA')
    except Exception:  # noqa: BLE001 - an SVG resvg cannot render is shown as a blank image
        im = Image.new('RGBA', (size * 4, size * 4), (255, 255, 255, 255))
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    return np.asarray(bg.convert('RGB').resize((size, size), Image.LANCZOS), float)


def colour_cells(svg_text):
    """Index into PALETTE of each cell of a 24 x 24 rendering, nearest colour by RGB distance."""
    return ((render(svg_text, 24)[:, :, None, :] - COLOURS[None, None]) ** 2).sum(-1).argmin(-1)


def brightness_cells(svg_text):
    """Index into RAMP of each cell of a 32 x 32 rendering, 0 for white."""
    lum = render(svg_text, 32) @ np.array([0.299, 0.587, 0.114]) / 255
    return np.minimum(len(RAMP) - 1, ((1 - lum) * len(RAMP)).astype(int))


def colour_grid(svg_text):
    grid = '\n'.join(''.join(KEYS[j] for j in row) for row in colour_cells(svg_text))
    used = [k for k in KEYS if k in grid]
    return {
        'image_grid': grid,
        'legend': 'each character is one cell of the image, 24 by 24, row 1 at the top; '
        + ', '.join(f'{k} = {PALETTE[k][0]}' for k in used),
    }


def brightness_grid(svg_text):
    grid = '\n'.join(''.join(RAMP[j] for j in row) for row in brightness_cells(svg_text))
    return {
        'image_grid': grid,
        'legend': f'each character is one cell of the image, 32 by 32, row 1 at the top; from light to dark: {" ".join(RAMP)}',
    }


GRIDS = {'colour_grid': colour_grid, 'brightness_grid': brightness_grid}


def state(raw, recipe, version):
    """The state Jev is given: {'svg': ...} for the SVG versions, the grid and its legend for the grid versions."""
    if version in GRIDS:
        return GRIDS[version](prepare(raw, recipe))
    return {'svg': prepare(raw, recipe, version)}
