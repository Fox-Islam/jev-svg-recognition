"""Download the source SVGs into data/svg/ and check that preparing them reproduces the text Jev was shown.

    uv run python scripts/fetch.py emoji      # 1,441 small files from jsDelivr and GitHub, at pinned versions
    uv run python scripts/fetch.py rapidata   # 438 cells of Rapidata/svg-benchmark's parquet files, at a pinned revision

The SVGs are not in the repository: each set has its own licence (docs/data.md). Every prepared string is
compared with the sha1 in its manifest, and a mismatch means the answers in data/answers do not apply to it.
"""

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevsvg import data as D
from jevsvg.experiments import svg_path
from jevsvg.grids import state
from jevsvg.svg import prepare

# the Hub's parquet conversion of Rapidata/svg-benchmark that the manifest's file, row group and row refer to
RAPIDATA_REVISION = '98607dd4c79ac86b8d1c7b9b3dfbe01ce490102d'


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def fetch_emoji():
    def one(r):
        path = svg_path(r)
        if not path.exists():
            for _ in range(3):
                try:
                    body = urllib.request.urlopen(r['source_url'], timeout=60).read()
                except OSError:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                break
            else:
                return r['item_id'], 'download failed'
        raw = path.read_text()
        for v in ('colour', 'grey', 'luma', 'normalised'):
            want = r[f'sha1_{v}']
            if want and sha(prepare(raw, r['recipe'], v)) != want:
                return r['item_id'], f'{v} differs'
        return r['item_id'], 'ok'

    with ThreadPoolExecutor(16) as ex:
        report(list(ex.map(one, D.manifest('emoji'))))


def fetch_rapidata():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rows = D.manifest('rapidata')
    groups = {}
    for r in rows:
        groups.setdefault((r['parquet_file'], int(r['row_group']), r['column']), []).append(r)
    fs = HfFileSystem()
    results = []
    for (name, group, column), items in sorted(groups.items()):
        path = f'datasets/Rapidata/svg-benchmark@{RAPIDATA_REVISION}/default/train/{name}'
        with fs.open(path, 'rb', block_size=2**20) as f:  # one column of one row group: the PNG renders are most of the 44 GB
            cells = pq.ParquetFile(f).read_row_group(group, columns=[column]).column(0).to_pylist()
        for r in items:
            raw = cells[int(r['row'])]
            out = svg_path(r)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(raw)
            results.append((r['item_id'], check_rapidata(raw, r)))
        print(f'{len(results)}/{len(rows)}', flush=True)
    report(results)


def check_rapidata(raw, r):
    """The SVG text, and the two grids made from its rendering, against the manifest. A grid differs when the
    renderer does, and then the grid answers in data/answers do not apply."""
    if sha(prepare(raw, 'rapidata')) != r['sha1_colour']:
        return 'colour differs'
    for v in ('colour_grid', 'brightness_grid'):
        if hashlib.sha1(json.dumps(state(raw, 'rapidata', v), sort_keys=True).encode()).hexdigest() != r[f'sha1_{v}']:
            return f'{v} differs'
    return 'ok'


def report(results):
    bad = [(i, s) for i, s in results if s != 'ok']
    print(f'{len(results) - len(bad)}/{len(results)} reproduce the prepared text exactly')
    for i, s in bad[:20]:
        print(f'  {i}: {s}')


if __name__ == '__main__':
    {'emoji': fetch_emoji, 'rapidata': fetch_rapidata}[sys.argv[1]]()
