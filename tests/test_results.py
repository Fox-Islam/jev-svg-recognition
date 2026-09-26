"""The committed results are what the committed data gives, and the README quotes only those results."""

import json
import re
import warnings
from pathlib import Path

import pytest

from jevsvg import experiments

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / 'results' / 'results.json').read_text())
warnings.filterwarnings('ignore')


@pytest.mark.parametrize('name', ['normalisation', 'colour', 'fourth_style', 'rapidata'])
def test_experiment_reproduces_the_committed_result(name):
    assert experiments.ALL[name]() == RESULTS[name]


@pytest.mark.slow
def test_corpus_pilot_reproduces_the_committed_result():
    assert experiments.corpus_pilot() == RESULTS['corpus_pilot']


@pytest.mark.slow
@pytest.mark.parametrize('name', ['rapidata_baselines', 'rapidata_grids', 'rapidata_clip'])
def test_rapidata_comparison_reproduces_the_committed_result(name):
    got = experiments.ALL[name]()
    if 'skipped' in got:
        pytest.skip(got['skipped'])
    assert got == RESULTS[name]


def numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


@pytest.mark.parametrize('page', ['README.md', 'docs/results.md'])
def test_every_percentage_quoted_is_a_committed_result(page):
    known = {f'{v * 100:.1f}' for v in numbers(RESULTS) if 0 <= v <= 1}
    quoted = re.findall(r'(\d+\.\d)%', (ROOT / page).read_text())
    assert quoted, f'{page} quotes no percentages, so this test checks nothing'
    assert [q for q in quoted if q not in known] == []
