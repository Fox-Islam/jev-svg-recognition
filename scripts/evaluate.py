"""Recompute every result from data/ and write results/results.json. No Jev calls.

uv run python scripts/evaluate.py            # all experiments
uv run python scripts/evaluate.py rapidata   # one
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevsvg import experiments

warnings.filterwarnings('ignore')
OUT = Path(__file__).resolve().parent.parent / 'results' / 'results.json'

if __name__ == '__main__':
    names = sys.argv[1:] or list(experiments.ALL)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.parent.mkdir(exist_ok=True)
    for name in names:  # written after each experiment, so a long run keeps what it has finished
        results[name] = experiments.ALL[name]()
        print(name, json.dumps(results[name], indent=1), flush=True)
        OUT.write_text(json.dumps(results, indent=1, sort_keys=True) + '\n')
