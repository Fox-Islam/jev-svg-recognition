"""Manifests, questions and recorded Jev answers in data/."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'
SUBGROUPS = [
    'animal-mammal',
    'animal-bird',
    'food-fruit',
    'food-prepared',
    'drink',
    'transport-ground',
    'sky-weather',
    'place-building',
    'clothing',
    'tool',
    'household',
    'sport',
    'time',
    'geometric',
]
STYLES = ('openmoji', 'twemoji', 'fluent', 'noto')


def manifest(name):
    with open(DATA / 'manifests' / f'{name}.csv', newline='') as f:
        return list(csv.DictReader(f))


def questions(name):
    """fixed100: [{id, question}]; corpus: [{id, source, question}]; corpus100: a list of corpus ids.

    Answers are keyed by id: four corpus questions share their text with another corpus question from a
    different source, and each was asked separately.
    """
    return json.loads((DATA / 'questions' / f'{name}.json').read_text())


def ids(name):
    q = questions(name)
    return q if isinstance(q[0], str) else [x['id'] for x in q]


def answers(name):
    """{(item_id, version): record}. `record['answers'][question_set]` maps question id to Jev's probability of yes.

    Answers are kept per question set (fixed100, corpus) because a question's answer shifts slightly with the
    other questions asked in the same call: 9,200 questions asked in both sets differ by 0.016 on average and
    by up to 0.16.
    """
    with gzip.open(DATA / 'answers' / f'{name}.jsonl.gz', 'rt') as f:
        return {(r['item'], r['version']): r for r in map(json.loads, f)}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def features(records, items, version, qs, question_set='fixed100', missing=0.5):
    """Log-odds of each answer: one row per item, one column per question. An unasked question counts as `missing`."""
    rows = []
    for it in items:
        d = records.get((it['item_id'], version), {}).get('answers', {}).get(question_set, {})
        rows.append([d.get(q, missing) for q in qs])
    return logit(rows)


def has(records, item, version, qs, question_set='fixed100'):
    d = records.get((item['item_id'], version), {}).get('answers', {}).get(question_set, {})
    return all(q in d for q in qs)
