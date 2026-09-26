"""SVG text preparation: what Jev is shown for each item.

Every function here must give byte-identical output for the answers in data/answers to apply to it, so the
manifests record a sha1 of each prepared string and scripts/fetch.py checks them.
"""

import re

GREY = '#9b9b9b'
NS = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
STOP = {'make', 'svg', 'icon', 'for', 'with', 'the', 'and', 'of', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'its', 'it', 'is'}


def compact(svg):
    """Whitespace collapsed, ids and xmlns dropped, numbers cut to one decimal."""
    svg = re.sub(r'\s+', ' ', svg)
    svg = re.sub(r' id="[^"]*"', '', svg)
    svg = re.sub(r'(\d+\.\d)\d+', r'\1', svg)
    svg = re.sub(r' xmlns="[^"]*"', '', svg).replace('> <', '><')
    return svg.strip()


def clean(raw):
    """compact(), after removing the XML declaration and the root's size, namespace and style attributes."""
    raw = re.sub(r'<\?xml[^>]*>', '', raw)
    root = re.match(r'\s*<svg\b[^>]*>', raw).group(0)
    root2 = re.sub(r'\s(width|height|xmlns(:\w+)?|version|style|enable-background|xml:space)="[^"]*"', '', root)
    return compact(root2 + raw[len(root) :])


def sanitise(svg):
    """Remove everything that could name what the SVG depicts, keeping how it looks.

    Comments, <title>, <desc>, <metadata> and <text> go; ids and class names are renamed to i0, c0 ... with
    every url(#...), href and CSS selector updated, because generated SVGs name their parts in them
    (`<!-- Bottle cap -->`, `id="bellGrad"`).
    """
    s = re.sub(r'<\?xml.*?\?>|<!DOCTYPE[^>]*>|<!--.*?-->', '', svg, flags=re.DOTALL)
    s = re.sub(r'<(title|desc|metadata|text|sodipodi:namedview)\b.*?</\1>', '', s, flags=re.DOTALL)
    s = re.sub(r'<(title|desc|metadata|text|tspan|sodipodi:namedview)\b[^>]*/>', '', s)
    s = re.sub(
        r'\s(?:inkscape|sodipodi|xml|xmlns:\w+|data-[\w-]+|aria-[\w-]+|version|enable-background|xml:space)(?::[\w-]+)?="[^"]*"',
        '',
        s,
    )
    ids = list(dict.fromkeys(re.findall(r'\bid="([^"]+)"', s)))
    classes = list(dict.fromkeys(c for v in re.findall(r'\bclass="([^"]+)"', s) for c in v.split()))
    styles = ' '.join(re.findall(r'<style[^>]*>(.*?)</style>', s, re.DOTALL))
    classes += [c for c in re.findall(r'\.([A-Za-z_][\w-]*)\s*[{,]', styles) if c not in classes]
    for i in sorted(ids, key=len, reverse=True):  # longest first, so one id that prefixes another is not hit early
        e, new = re.escape(i), f'i{ids.index(i)}'
        s = re.sub(rf'\bid="{e}"', f'id="{new}"', s)
        s = re.sub(rf'#{e}(?![\w-])', f'#{new}', s)
    for c in sorted(classes, key=len, reverse=True):
        e, new = re.escape(c), f'c{classes.index(c)}'
        s = re.sub(
            r'\bclass="([^"]*)"',
            lambda m, c=c, new=new: 'class="' + ' '.join(new if t == c else t for t in m.group(1).split()) + '"',
            s,
        )
        s = re.sub(rf'\.{e}(?![\w-])', f'.{new}', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'(\d+\.\d)\d+', r'\1', s)
    return s.replace('> <', '><').strip()


def label_words(label):
    return [w for w in re.findall(r'[a-z]{4,}', label.lower()) if w not in STOP]


def leaks(svg, label):
    """Words of the label, four letters or longer, that appear anywhere in the SVG text."""
    low = svg.lower()
    return [w for w in label_words(label) if w in low]


def greyed(svg):
    """Every fill and stroke colour set to one grey; black and none are kept, so line art is unchanged."""

    def rep(m):
        v = m.group(2).lower()
        return m.group(0) if v in ('none', '#000', '#000000', 'black') else f'{m.group(1)}="{GREY}"'

    svg = re.sub(r'(fill|stroke|stop-color)="([^"]*)"', rep, svg)
    return re.sub(
        r'(fill|stroke):\s*(#[0-9a-fA-F]{3,6})',
        lambda m: m.group(0) if m.group(2).lower() in ('#000', '#000000') else f'{m.group(1)}:{GREY}',
        svg,
    )


def _hex6(h):
    h = h.lstrip('#')
    return ''.join(c * 2 for c in h) if len(h) == 3 else h[:6]


def luma(svg):
    """Every hex colour replaced by the grey of the same lightness (Rec. 601 weights)."""

    def g(h):
        r, gg, b = (int(_hex6(h)[i : i + 2], 16) for i in (0, 2, 4))
        y = round(0.299 * r + 0.587 * gg + 0.114 * b)
        return f'#{y:02x}{y:02x}{y:02x}'

    svg = re.sub(r'(fill|stroke|stop-color)="(#[0-9a-fA-F]{3,6})"', lambda m: f'{m.group(1)}="{g(m.group(2))}"', svg)
    return re.sub(r'(fill|stroke|stop-color):\s*(#[0-9a-fA-F]{3,6})', lambda m: f'{m.group(1)}:{g(m.group(2))}', svg)


def _rgb(c):
    c = c.strip().lstrip('#')
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4)) if re.fullmatch(r'[0-9a-fA-F]{6}', c) else None


def flatten_gradients(svg):
    """Each gradient fill or stroke replaced by the mean of its stop colours, following href inheritance.

    picosvg rejects some gradients, and shading differs more between emoji sets than shape does.
    """
    grads = {}
    for m in re.finditer(r'<(linearGradient|radialGradient)\b([^>]*?)(/>|>(.*?)</\1>)', svg, re.DOTALL):
        attrs, inner = m.group(2), m.group(4) or ''
        gid = re.search(r'\bid="([^"]+)"', attrs)
        if not gid:
            continue
        stops = [_rgb(c) for c in re.findall(r'stop-color(?:="|:\s*)(#[0-9a-fA-F]{3,6})', inner)]
        href = re.search(r'href="#([^"]+)"', attrs)
        grads[gid.group(1)] = ([s for s in stops if s], href.group(1) if href else None)

    def colour(gid, depth=0):
        stops, href = grads.get(gid, ([], None))
        if not stops and href and depth < 5:
            return colour(href, depth + 1)
        if not stops:
            return None
        return '#' + ''.join(f'{round(sum(s[i] for s in stops) / len(stops)):02x}' for i in range(3))

    svg = re.sub(r'(fill|stroke)="url\(#([^)"]+)\)"', lambda m: f'{m.group(1)}="{colour(m.group(2)) or "#808080"}"', svg)
    svg = re.sub(r'(fill|stroke):\s*url\(#([^)"]+)\)', lambda m: f'{m.group(1)}:{colour(m.group(2)) or "#808080"}', svg)
    return re.sub(r'<(linearGradient|radialGradient)\b[^>]*?(/>|>.*?</\1>)', '', svg, flags=re.DOTALL)


def _num(m):
    v = round(float(m.group(0)), 1)
    return str(int(v)) if v == int(v) else str(v)


def normalise(svg):
    """One form for every emoji set: gradients flattened, then picosvg (shapes to paths, transforms, groups,
    strokes and clips resolved, absolute commands), then a 0-100 canvas at one decimal with only fill and
    opacity kept.

    Paint order is kept: later paths are drawn on top, so sorting them changes the picture.
    """
    from picosvg.svg import SVG
    from picosvg.svg_transform import Affine2D

    src = svg if 'xmlns=' in svg[:300] else svg.replace('<svg', NS, 1)
    pico = SVG.fromstring(flatten_gradients(src)).topicosvg()
    x, y, w, h = pico.view_box()
    k = 100 / max(w, h)
    t = Affine2D(k, 0, 0, k, -x * k + (100 - w * k) / 2, -y * k + (100 - h * k) / 2)
    out = []
    for p in pico.shapes():
        p = p.apply_transform(t)
        d = re.sub(r'-?\d*\.?\d+(?:e-?\d+)?', _num, p.d)
        d = re.sub(r'\s*([MLCQAZmlcqaz])\s*', r'\1', d).replace(',', ' ')
        attrs = f' fill="{p.fill}"' if p.fill else ''
        for a in ('opacity', 'fill_opacity'):
            v = getattr(p, a, 1)
            if v not in (1, 1.0, None):
                attrs += f' {a.replace("_", "-")}="{round(float(v), 2)}"'
        out.append(f'<path d="{d}"{attrs}/>')
    return sanitise(f'<svg viewBox="0 0 100 100">{"".join(out)}</svg>')


def normalise_or_raw(svg):
    """(normalised, True), or (svg, False) where picosvg cannot read it."""
    try:
        return normalise(svg), True
    except Exception:  # noqa: BLE001 - picosvg raises many unrelated types for inputs it cannot flatten
        return svg, False


def rapidata(raw):
    """A Rapidata SVG as asked: sanitised, cut to start at its <svg> element."""
    s = sanitise(raw)
    if not s.startswith('<svg'):
        s = s[s.find('<svg') :] if '<svg' in s else ''
    return s


RECIPES = {
    'compact': compact,  # OpenMoji emoji in the original 280
    'clean': clean,  # Twemoji, Fluent, and the other OpenMoji emoji
    'noto': lambda raw: clean(sanitise(raw)),  # Noto files start with comments and a doctype
    'rapidata': rapidata,
}


def prepare(raw, recipe, version='colour'):
    """The exact text Jev was shown for one item in one version: colour, grey, luma or normalised."""
    base = RECIPES[recipe](raw)
    if version == 'colour':
        return base
    if version == 'grey':
        return greyed(base)
    if version == 'luma':
        return luma(base)
    if version == 'normalised':
        return normalise_or_raw(base)[0]
    raise ValueError(f'unknown version {version!r}')
