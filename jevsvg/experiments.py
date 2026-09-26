"""Every result in the README, computed from data/ without calling Jev.

The classifier throughout is a logistic regression on standardised log-odds of Jev's answers, C=0.1.
Splits are stored in the manifests, so each number is reproducible exactly.
"""

import collections
import random

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D


def model(max_iter=3000):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=max_iter))


def fit_predict(Xtr, ytr, Xte):
    return model().fit(Xtr, ytr).predict(Xte)


def emoji_items():
    """The emoji entries the experiments train and test on: the OpenMoji training 200, the validation set, and Noto."""
    return [r for r in D.manifest('emoji') if r['e_heldout'] != '']


def subgroup(items):
    return np.array([D.SUBGROUPS.index(i['subgroup']) for i in items])


def acc(ok):
    return round(float(np.mean(ok)), 4)


def normalisation():
    """Raw vs normalised SVG text, fixed 100 questions: each style tested unseen, and all styles together.
    Test emoji are identities never seen in training in any style."""
    ans, qs = D.answers('emoji'), D.ids('fixed100')
    items = [i for i in emoji_items() if D.has(ans, i, 'normalised', qs)]
    held = [i['e_heldout'] == '1' for i in items]
    out = {'items': len(items), 'unseen_style': {}}
    for s in D.STYLES:
        tr = [i for i, h in zip(items, held) if i['set'] != s and not h]
        te = [i for i, h in zip(items, held) if i['set'] == s and h]
        row = {'n': len(te)}
        for v in ('colour', 'normalised'):
            ok = fit_predict(D.features(ans, tr, v, qs), subgroup(tr), D.features(ans, te, v, qs)) == subgroup(te)
            row['raw' if v == 'colour' else 'normalised'] = acc(ok)
            row['_' + v] = ok
        row['only_normalised'] = int((row['_normalised'] & ~row['_colour']).sum())
        row['only_raw'] = int((row['_colour'] & ~row['_normalised']).sum())
        del row['_colour'], row['_normalised']
        out['unseen_style'][s] = row
    out['unseen_style_mean'] = {
        k: round(float(np.mean([r[k] for r in out['unseen_style'].values()])), 4) for k in ('raw', 'normalised')
    }
    tr = [i for i, h in zip(items, held) if not h]
    te = [i for i, h in zip(items, held) if h]
    out['all_styles'] = {'n': len(te)}
    for v, k in (('colour', 'raw'), ('normalised', 'normalised')):
        out['all_styles'][k] = acc(
            fit_predict(D.features(ans, tr, v, qs), subgroup(tr), D.features(ans, te, v, qs)) == subgroup(te)
        )
    return out


def colour():
    """Train on the 200 OpenMoji training emoji, test on unseen emoji in each set, in colour, one grey, and lightness."""
    ans, qs = D.answers('emoji'), D.ids('fixed100')
    rows = D.manifest('emoji')
    tr = [r for r in rows if r['recipe'] == 'compact' and r['openmoji_split'] == 'train']
    val = [r for r in rows if r['recipe'] == 'clean']
    out = {}
    for s in ('openmoji', 'twemoji', 'fluent'):
        te = [r for r in val if r['set'] == s]
        out[s] = {'n': len(te)}
        for v in ('colour', 'grey', 'luma'):
            out[s][v] = acc(fit_predict(D.features(ans, tr, v, qs), subgroup(tr), D.features(ans, te, v, qs)) == subgroup(te))
    return out


def fourth_style():
    """Noto added as a fourth style: unseen, then in training (fixed 100, colour)."""
    ans, qs = D.answers('emoji'), D.ids('fixed100')
    items = emoji_items()
    others = [i for i in items if i['set'] != 'noto']
    noto = [i for i in items if i['set'] == 'noto']
    train_ids = {r['hexcode'] for r in D.manifest('emoji') if r['openmoji_split'] == 'train'}
    te = [i for i in noto if i['hexcode'] not in train_ids]
    F = lambda its: D.features(ans, its, 'colour', qs)
    out = {
        'noto_unseen_style_shared_identities': {
            'n': len(te),
            'accuracy': acc(fit_predict(F(others), subgroup(others), F(te)) == subgroup(te)),
        }
    }
    held = lambda i: i['noto_heldout'] == '1'
    te_all = [i for i in items if held(i)]
    te_noto = [i for i in noto if held(i)]
    for name, tr in (('3_styles', [i for i in others if not held(i)]), ('4_styles', [i for i in items if not held(i)])):
        m = model().fit(F(tr), subgroup(tr))
        out[name] = {
            'train': len(tr),
            'all_heldout': acc(m.predict(F(te_all)) == subgroup(te_all)),
            'n_all': len(te_all),
            'noto_heldout': acc(m.predict(F(te_noto)) == subgroup(te_noto)),
            'n_noto': len(te_noto),
        }
    return out


def corpus_pilot():
    """The 200-emoji pilot: the full 1,102-question corpus against the fixed 100, 5-fold x3 cross-validation.
    An emoji counts as right when it is right in most of the three repeats."""
    ans = D.answers('emoji')
    items = sorted([i for i in emoji_items() if i['pilot_rank'] != ''], key=lambda i: int(i['pilot_rank']))
    y = subgroup(items)
    corpus = D.questions('corpus')
    k = min(5, min(collections.Counter(y).values()))

    def cv(qs):
        X = D.features(ans, items, 'colour', qs, 'fixed100' if qs == fixed_qs else 'corpus')
        oks = [
            cross_val_predict(model(1000), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=s), n_jobs=-1) == y
            for s in range(3)
        ]
        return np.mean(oks, 0) > 0.5

    fixed_qs = D.ids('fixed100')
    fixed, full = cv(fixed_qs), cv([q['id'] for q in corpus])
    out = {
        'n': len(items),
        'fixed100': acc(fixed),
        'corpus_all': acc(full),
        'corpus_questions': len(corpus),
        'only_corpus': int((full & ~fixed).sum()),
        'only_fixed': int((fixed & ~full).sum()),
        'by_source': {},
    }
    for src in dict.fromkeys(q['source'] for q in corpus):
        qs = [q['id'] for q in corpus if q['source'] == src]
        out['by_source'][src] = {'questions': len(qs), 'accuracy': acc(cv(qs))}
    rng = random.Random(0)
    out['random_100'] = [acc(cv([q['id'] for q in rng.sample(corpus, 100)])) for _ in range(3)]
    return out


def rapidata():
    """23 single-object prompts drawn by LLMs: classify SVGs from models, or whole model families, never seen in training."""
    ans, qs = D.answers('rapidata'), D.ids('corpus100')
    items = D.manifest('rapidata')
    labels = sorted({i['label'] for i in items})
    y = lambda its: np.array([labels.index(i['label']) for i in its])
    F = lambda its: D.features(ans, its, 'colour', qs, 'corpus')
    tr = [i for i in items if i['split'] == 'train']
    te = [i for i in items if i['split'] == 'test']
    ok = fit_predict(F(tr), y(tr), F(te)) == y(te)
    rec = lambda i: ans[(i['item_id'], 'colour')]
    zs = np.array([max(rec(i)['zero_shot'], key=rec(i)['zero_shot'].get) == i['label'] for i in te])
    families = sorted({i['family'] for i in items})
    fam_ok, per_family = [], {}
    for f in families:
        tr_f = [i for i in items if i['family'] != f]
        te_f = [i for i in items if i['family'] == f]
        o = fit_predict(F(tr_f), y(tr_f), F(te_f)) == y(te_f)
        fam_ok += list(o)
        per_family[f] = {'n': len(te_f), 'accuracy': acc(o)}
    rho, p = spearmanr([rec(i)['shows_label'] for i in items], [float(i['alignment']) for i in items])
    return {
        'labels': len(labels),
        'train': len(tr),
        'test': len(te),
        'test_models': len({i['model'] for i in te}),
        'zero_shot': acc(zs),
        'model_split': acc(ok),
        'family_split': acc(fam_ok),
        'per_family': per_family,
        'shows_label_vs_human_alignment': {
            'spearman': round(float(rho), 3),
            'p': float(p),
            'n': len(items),
            'jev_yes_rate': acc([rec(i)['shows_label'] > 0.5 for i in items]),
        },
    }


def svg_path(item):
    """Where scripts/fetch.py puts an item's source SVG."""
    import hashlib

    if item['item_id'].startswith('rapidata:'):
        return D.DATA / 'svg' / 'rapidata' / f'{hashlib.sha1(item["item_id"].encode()).hexdigest()[:16]}.svg'
    return D.DATA / 'svg' / item['set'] / f'{item["hexcode"]}.svg'


def pixels(svg_text, size=16):
    """The SVG rendered on white and shrunk to size x size RGB, scaled 0-1."""
    import io
    import re

    import resvg_py
    from PIL import Image

    from .svg import NS

    s = svg_text if 'xmlns=' in svg_text[:300] else svg_text.replace('<svg', NS, 1)
    s = re.sub(r'\s(width|height)="[^"]*"', '', s[: s.index('>')]) + s[s.index('>') :]
    s = s.replace('<svg', '<svg width="32" height="32"', 1)
    try:
        im = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=s)))).convert('RGBA')
    except Exception:  # noqa: BLE001 - an SVG resvg cannot render counts as a blank grey image
        return np.full(size * size * 3, 0.5)
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    return np.asarray(bg.convert('RGB').resize((size, size)), float).ravel() / 255


def _rapidata_splits(X, fit=fit_predict):
    """Model-split accuracy and family-split accuracy for features X: {item_id: vector}, or a function of a list of items."""
    items = D.manifest('rapidata')
    labels = sorted({i['label'] for i in items})
    y = lambda its: np.array([labels.index(i['label']) for i in its])
    F = X if callable(X) else (lambda its: np.array([X[i['item_id']] for i in its]))
    tr = [i for i in items if i['split'] == 'train']
    te = [i for i in items if i['split'] == 'test']
    fam = []
    for f in sorted({i['family'] for i in items}):
        a = [i for i in items if i['family'] != f]
        b = [i for i in items if i['family'] == f]
        fam += list(fit(F(a), y(a), F(b)) == y(b))
    return {'model_split': acc(fit(F(tr), y(tr), F(te)) == y(te)), 'family_split': acc(fam)}


def _rapidata_texts():
    from .svg import prepare

    items = D.manifest('rapidata')
    if not all(svg_path(i).exists() for i in items):
        return None
    return {i['item_id']: prepare(svg_path(i).read_text(), 'rapidata') for i in items}


# element and path-command counts, and the share of colours nearest each of these, for the structure baseline
STRUCTURE_TAGS = (
    'path',
    'circle',
    'ellipse',
    'rect',
    'line',
    'polyline',
    'polygon',
    'g',
    'linearGradient',
    'radialGradient',
    'use',
    'filter',
)
STRUCTURE_COMMANDS = ('[Mm]', '[Ll]', '[HhVv]', '[CcSs]', '[Qq]', '[Aa]', '[Zz]')


def structure(svg_text):
    import re

    from .grids import COLOURS

    f = [svg_text.count('<' + t) for t in STRUCTURE_TAGS]
    d = ' '.join(re.findall(r' d="([^"]*)"', svg_text))
    f += [len(re.findall(c, d)) for c in STRUCTURE_COMMANDS]
    f += [len(svg_text) / 1000, len(re.findall(r'-?\d+\.?\d*', svg_text)) / 100]
    hist = np.zeros(len(COLOURS))
    for h in re.findall(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b', svg_text):
        h = ''.join(c * 2 for c in h) if len(h) == 3 else h
        hist[((COLOURS - np.array([int(h[k : k + 2], 16) for k in (0, 2, 4)])) ** 2).sum(1).argmin()] += 1
    return np.concatenate([np.log1p(f), hist / max(hist.sum(), 1)])


def rapidata_baselines():
    """Classifiers that see no Jev answer: rendered pixels, character n-grams of the SVG code, and element and colour
    counts. Needs `scripts/fetch.py rapidata`."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = _rapidata_texts()
    if texts is None:
        return {'skipped': 'run scripts/fetch.py rapidata first'}

    def tfidf(train_texts, ytr, test_texts):
        m = make_pipeline(
            TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=50000),
            LogisticRegression(C=10, max_iter=3000),
        )
        return m.fit(train_texts, ytr).predict(test_texts)

    return {
        'pixels_16x16': _rapidata_splits({k: pixels(t) for k, t in texts.items()}),
        'code_ngrams': _rapidata_splits(lambda its: [texts[i['item_id']] for i in its], fit=tfidf),
        'code_structure': _rapidata_splits({k: structure(t) for k, t in texts.items()}),
    }


def rapidata_grids():
    """The SVG rendered and written as a character grid, asked the same corpus100 questions; against a classifier
    on the grid cells themselves. The grid-cell rows need `scripts/fetch.py rapidata`."""
    from . import grids as G

    ans, qs = D.answers('rapidata'), D.ids('corpus100')
    items = D.manifest('rapidata')
    te = [i for i in items if i['split'] == 'test']
    texts = _rapidata_texts()
    out = {}
    for v, cells in (
        ('colour_grid', lambda t: np.eye(len(G.KEYS))[G.colour_cells(t).ravel()].ravel()),
        ('brightness_grid', lambda t: G.brightness_cells(t).ravel().astype(float)),
    ):
        F = lambda its, v=v: D.features(ans, its, v, qs, 'corpus')
        both = lambda its, v=v: np.hstack([D.features(ans, its, 'colour', qs, 'corpus'), D.features(ans, its, v, qs, 'corpus')])
        out[v] = {
            'zero_shot': acc([ans[(i['item_id'], v)]['zero_shot_choice'] == i['label'] for i in te]),
            'jev_answers': _rapidata_splits(F),
            'jev_answers_with_svg_answers': _rapidata_splits(both),
        }
        if texts is not None:
            out[v]['grid_cells_no_jev'] = _rapidata_splits({k: cells(t) for k, t in texts.items()})
    return out


def rapidata_clip():
    """CLIP ViT-B/32 (LAION-2B weights) on 224 x 224 renderings: zero-shot against "an icon of a <label>", and a
    classifier on its image embeddings. Needs the clip extra and `scripts/fetch.py rapidata`."""
    try:
        import open_clip
        import torch
    except ImportError:
        return {'skipped': 'install the clip extra'}
    from PIL import Image

    from .grids import render

    texts = _rapidata_texts()
    if texts is None:
        return {'skipped': 'run scripts/fetch.py rapidata first'}
    items = D.manifest('rapidata')
    labels = sorted({i['label'] for i in items})
    model, _, prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tok = open_clip.get_tokenizer('ViT-B-32')
    ims = [Image.fromarray(render(texts[i['item_id']], 224).astype('uint8')) for i in items]
    with torch.no_grad():
        Z = torch.cat([model.encode_image(torch.stack([prep(im) for im in ims[k : k + 32]])) for k in range(0, len(ims), 32)])
        T = model.encode_text(tok([f'an icon of a {label}' for label in labels]))
    Z = (Z / Z.norm(dim=-1, keepdim=True)).numpy()
    T = (T / T.norm(dim=-1, keepdim=True)).numpy()
    emb = {i['item_id']: z for i, z in zip(items, Z)}
    zs = {i['item_id']: labels[int(k)] for i, k in zip(items, (Z @ T.T).argmax(1))}
    te = [i for i in items if i['split'] == 'test']
    return {'zero_shot': acc([zs[i['item_id']] == i['label'] for i in te]), 'embeddings': _rapidata_splits(emb)}


ALL = {
    'normalisation': normalisation,
    'colour': colour,
    'fourth_style': fourth_style,
    'corpus_pilot': corpus_pilot,
    'rapidata': rapidata,
    'rapidata_baselines': rapidata_baselines,
    'rapidata_grids': rapidata_grids,
    'rapidata_clip': rapidata_clip,
}
