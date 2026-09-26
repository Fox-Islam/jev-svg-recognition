"""Ask Jev a question set for every item that lacks those answers, and add them to data/answers.

    uv run python scripts/ask.py emoji colour fixed100
    uv run python scripts/ask.py rapidata colour corpus100
    uv run python scripts/ask.py rapidata colour_grid corpus100

Needs OPENROUTER_API_KEY or TYPESAFE_API_KEY, and the SVGs from scripts/fetch.py. Questions go 25 to a call,
as in every recorded run: a question's answer shifts slightly with the questions beside it, so a different
grouping gives slightly different numbers.
"""

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevsvg import data as D
from jevsvg.client import jev
from jevsvg.experiments import svg_path
from jevsvg.grids import state

PER_CALL = 25


def question_set(name):
    """(set name as stored, [(id, text)]): corpus100 answers are stored with the corpus they are drawn from."""
    if name == 'corpus100':
        text = {q['id']: q['question'] for q in D.questions('corpus')}
        return 'corpus', [(i, text[i]) for i in D.questions('corpus100')]
    return name, [(q['id'], q['question']) for q in D.questions(name)]


def main(manifest, version, qset):
    stored_as, qs = question_set(qset)
    records = D.answers(manifest)
    jobs = []
    for it in D.manifest(manifest):
        if not svg_path(it).exists():
            continue
        have = records.get((it['item_id'], version), {}).get('answers', {}).get(stored_as, {})
        need = [q for q in qs if q[0] not in have]
        jobs += [(it, need[i : i + PER_CALL]) for i in range(0, len(need), PER_CALL)]
    print(f'{len(jobs)} calls to make')

    def run(job):
        it, batch = job
        st = state(svg_path(it).read_text(), it.get('recipe', 'rapidata'), version)
        r = jev(st, {f'q{k:02d}': {'type': 'noul', 'instructions': text} for k, (_, text) in enumerate(batch)})
        return it, batch, r

    with ThreadPoolExecutor(8) as ex:
        for it, batch, r in ex.map(run, jobs):
            rec = records.setdefault((it['item_id'], version), {'item': it['item_id'], 'version': version, 'answers': {}})
            got = rec['answers'].setdefault(stored_as, {})
            for k, (qid, _) in enumerate(batch):
                got[qid] = r['answers'][f'q{k:02d}']['noul']
    with gzip.open(D.DATA / 'answers' / f'{manifest}.jsonl.gz', 'wt') as f:
        for rec in records.values():
            f.write(json.dumps(rec, sort_keys=True) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
