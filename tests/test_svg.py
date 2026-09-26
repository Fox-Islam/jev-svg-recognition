import io

import numpy as np
import pytest

from jevsvg.svg import NS, flatten_gradients, greyed, leaks, luma, normalise, sanitise

LEAKY = """<?xml version="1.0"?><svg viewBox="0 0 10 10"><!-- Bottle cap --><title>water bottle</title>
<defs><linearGradient id="bottleGrad"><stop stop-color="#000"/></linearGradient></defs>
<style>.cap{fill:#00f}</style><rect class="cap" width="4" height="4" fill="url(#bottleGrad)"/><text>bottle</text>
<use href="#bottleGrad"/></svg>"""


def render(svg, size=48):
    resvg_py = pytest.importorskip('resvg_py')
    from PIL import Image

    s = svg.replace('<svg', f'{NS} width="{size}" height="{size}"', 1)
    im = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=s)))).convert('RGBA')
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    return np.asarray(bg.convert('RGB'), float) / 255


def test_sanitise_removes_every_name_and_keeps_references_working():
    out = sanitise(LEAKY)
    assert leaks(out, 'a bottle of water') == []
    assert 'cap' not in out and 'bottleGrad' not in out
    assert 'id="i0"' in out and 'url(#i0)' in out and 'href="#i0"' in out
    assert '.c0{' in out and 'class="c0"' in out


def test_sanitise_leaves_a_clean_svg_as_it_is():
    svg = '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="#f00"/></svg>'
    assert sanitise(svg) == svg


def test_gradients_become_the_mean_of_their_stops_through_href():
    svg = (
        '<svg><defs><linearGradient id="a"><stop stop-color="#000000"/><stop stop-color="#ffffff"/></linearGradient>'
        '<radialGradient id="b" href="#a"/></defs><rect fill="url(#b)"/></svg>'
    )
    out = flatten_gradients(svg)
    assert 'Gradient' not in out and 'fill="#808080"' in out


def test_grey_keeps_black_line_art_and_luma_keeps_lightness():
    svg = '<svg><path fill="#ff0000" stroke="#000"/></svg>'
    assert greyed(svg) == '<svg><path fill="#9b9b9b" stroke="#000"/></svg>'
    assert luma(svg) == '<svg><path fill="#4c4c4c" stroke="#000000"/></svg>'


def test_normalise_keeps_paint_order_and_the_picture():
    # the black disc is drawn first and covered by the grey one: sorting paths would put black on top
    svg = (
        '<svg viewBox="0 0 72 72"><circle cx="36" cy="36" r="28" fill="#000"/><circle cx="36" cy="36" r="28" fill="#3f3f3f"/>'
        '<path d="M10 60 L62 60" stroke="#e00" stroke-width="3"/></svg>'
    )
    out = normalise(svg)
    assert out.startswith('<svg viewBox="0 0 100 100">')
    assert out.index('#000') < out.index('#3f3f3f')
    assert np.abs(render(svg) - render(out)).mean() < 0.01
