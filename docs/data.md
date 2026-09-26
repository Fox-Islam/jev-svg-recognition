# Data

## What is in the repository

| File | Contents |
|---|---|
| `manifests/emoji.csv` | one row per emoji entry: set, Unicode code point, label, subgroup, recipe, source URL, the sha1 of each prepared version, and the splits the experiments use |
| `manifests/rapidata.csv` | one row per Rapidata SVG: label, prompt, model, model family, split, human alignment score, the sha1 of the SVG text and of each character grid, and where it is in the dataset: parquet file, row group, row within the group, and column (`svg1` or `svg2`) |
| `questions/*.json` | the question sets, by id (docs/method.md) |
| `answers/emoji.jsonl.gz`, `answers/rapidata.jsonl.gz` | one record per item and version; `answers[question_set][question_id]` is Jev's probability of yes |

A Rapidata `colour` record also holds `zero_shot`, Jev's probability for each of the 23 labels in one Choice,
and `shows_label`, its answer to "Does this SVG show <label>?". The `colour_grid` and `brightness_grid` records
hold the answers for the rendering written as a grid (`jevsvg/grids.py`), and `zero_shot_choice`, the label Jev
picked from that grid.

Manifest split columns:

| Column | Meaning |
|---|---|
| `openmoji_split` | `train` or `test` for the 280 OpenMoji emoji of the first colour test |
| `e_heldout` | 1 for emoji identities held out of training in the normalisation test; blank for entries outside the experiments |
| `noto_heldout` | 1 for identities held out in the fourth-style test |
| `pilot_rank` | order of the 200 corpus-pilot emoji |
| `split` (Rapidata) | `train` or `test`, by the model that drew the SVG |

## What is not, and why

The SVGs. `scripts/fetch.py` downloads them from the pinned sources below into `data/svg/` and checks every
prepared version, and each Rapidata character grid, against the sha1 in its manifest. Every one of the 1,441 emoji
entries and 438 Rapidata SVGs reproduces exactly. The grids depend on the renderer, pinned by uv.lock.

## Sources and licences

| Source | Version | Licence |
|---|---|---|
| [OpenMoji](https://openmoji.org) | npm `openmoji@17.0.0` | CC BY-SA 4.0 |
| [Twemoji](https://github.com/jdecked/twemoji) | `jdecked/twemoji@17.0.3` | graphics CC BY 4.0 |
| [Fluent Emoji](https://github.com/microsoft/fluentui-emoji), flat style | npm `@lobehub/fluent-emoji-flat@1.1.0` | MIT |
| [Noto Emoji](https://github.com/googlefonts/noto-emoji) | commit `e20cbc2bbec1`, `2D/svg` | Apache 2.0 |
| [Rapidata/svg-benchmark](https://huggingface.co/datasets/Rapidata/svg-benchmark) | the Hub's parquet conversion at revision `98607dd4c79a` | prompts CC BY 4.0 (Yupp AI for the non-seed prompts); the SVGs fall under each generating model provider's terms |

The emoji labels are Unicode CLDR short names and subgroups, as given in OpenMoji's `openmoji.json`. The
Rapidata prompts in `manifests/rapidata.csv` are reproduced under CC BY 4.0.

## Filters

- **Emoji**: the 14 subgroups of the tests, without skin-tone variants. 57 emoji are missing from the Fluent
  package and 31 from Noto's `2D/svg`. Noto SVGs over 20,000 characters are left out: white cane (57,722
  characters, which Jev did not answer) and gorilla (30,320).
- **Rapidata**: 32 single-object prompts were chosen by hand from the 57 of eight words or fewer, and 24 had SVGs
  from at least 15 models after these filters:
  - a human alignment score of 0.5 or more (627 of the chosen prompts' SVGs fell below);
  - at most 6,000 characters after sanitising;
  - no label word left after sanitising.

  The exit sign prompt is left out as well: its SVGs are black rectangles once the word EXIT, removed as
  `<text>`, is gone. That leaves 23 prompts and 438 SVGs from 41 models; 16 models are held out for the test.
