# jev-svg-recognition

Research task on weighting Jev's responses to classify SVGs. [View a demo here](http://jev-svg.lexic.cloud/).

I'm so sorry about this but the thought has been in my mind for ages, and was the reason I built [JevGraph](https://jevgraph.lexic.cloud/) in the first place. Basically, my thought was that regular image classifiers take in various forms of an image, do some maths on them, and then output some value that relates to what that image is. Doing maths on a number is kind of like asking a question to Jev, sort of??

Only issue was that Jev can't take image input directly. I tried representing the pixels in various ways but early tests had poor results, so I limited the rest of my testing to SVGs which Jev performs better-than-random chance at recognition on when given a `choice` over all of the valid options.

So, a few stumbles along the way but the idea I ended up with was this: Throw the SVG into context and then ask a whole bunch of questions about it, all sorts of things that might be related to what it is or isn't, "is this red?" "Is this a physical object?" "Is this an animal?" and so on, and then do some training on that output. The sense being that Jev might be able to glean some information out of the SVGs that feature classification alone can't.

To my surprise: It could! SVG -> 100 broad Jev questions -> classifier performed a fair bit better in these tests than SVG -> feature classifier, and in one case only slightly worse than CLIP reading the rendered pixels.

This is, of course, not a meaningful project on its own, but I think there's _something_ here to learn from. I'm not entirely sure what that is yet. Anyway, I had fun(?).

## Results

### Which of 23 objects an SVG shows, drawn by LLMs it has not seen

[Rapidata/svg-benchmark](https://huggingface.co/datasets/Rapidata/svg-benchmark) has 500 prompts, each drawn as
an SVG by up to 42 LLMs. From it: 23 prompts that each ask for one object (a lion, a fridge, a jellyfish),
drawings human raters scored 0.5 or more for matching their prompt, 438 SVGs in all. Training uses SVGs from some models and
testing uses SVGs from models never seen in training.

| Method | Test on 179 SVGs from 16 held-out models | Test on each model family held out in turn |
|---|---|---|
| Jev zero-shot: one Choice over the 23 labels, no training | 39.7% | - |
| Rendered 16×16 pixels → logistic regression | 73.7% | 78.1% |
| Character n-grams of the SVG code → logistic regression | 77.6% | 66.7% |
| Element and colour counts of the SVG code → logistic regression | 76.5% | 74.0% |
| **100 Jev yes/no answers about the SVG code → logistic regression** | **93.8%** | **92.0%** |
| 100 Jev answers about the rendering written as a 24×24 colour grid → logistic regression | 74.3% | 72.8% |
| [CLIP](https://github.com/mlfoundations/open_clip) ViT-B/32 zero-shot on the rendering, no training | 99.4% | - |
| CLIP image embeddings → logistic regression | 100.0% | 99.1% |

Guessing gets 1 in 23. Every row except the two zero-shot ones trains the same logistic regression on the same
259 SVGs; what differs is what it is given. Jev's answers about the SVG code beat every classifier that reads the
code or the pixels directly. The n-gram classifier learns each model family's coding habits and falls to 66.7%
when a family is held out, while Jev's answers hold at 92.0%. Given a character grid of the rendering instead of
the code, Jev does no better than a classifier on the grid itself, and zero-shot is basically chance.

CLIP, which sees the rendered image and was trained on image-caption pairs, names the object with no training at
all.

With whole families held out (every Claude, every GPT, ...), Claude drawings are
classified at 98.3% by a classifier that has seen no Claude drawing; the lowest family is quiver, an SVG
specialist, at 57.1%.

Asked directly whether an SVG shows its label, Jev agrees with the human alignment ratings not at all
(Spearman 0.004 over 438 SVGs) and says yes 73.7% of the time. It identifies an object from a closed set well,
and does not judge whether a drawing is good.

### Emoji categories, across drawing styles

Fourteen Unicode emoji subgroups (mammal, bird, fruit, drink, building, tool, household, ...) drawn by four
emoji sets: OpenMoji, Twemoji, Fluent and Noto. A subgroup mixes many objects, so this is harder than naming one
object. Every test emoji is one never seen in training, in any style.

| Trained on | Tested on | Accuracy |
|---|---|---|
| OpenMoji, Twemoji, Fluent | Noto, never seen in training | 36.0% |
| all four styles | Noto | 52.0% |
| OpenMoji, Twemoji, Fluent | all four styles | 49.8% |
| all four styles | all four styles | 56.0% |

Adding Noto to training raises Noto from 36.0% to 52.0%; that row also trains on more emoji. Guessing gets 1 in 14.

The full tables, including colour against greyscale and normalised against raw SVG text, are in
[docs/results.md](docs/results.md).

## How it works

1. **Prepare the SVG text** (`jevsvg/svg.py`). Everything that could name the object is removed: comments,
   `<title>`, `<desc>`, `<text>`, and id and class names, renamed to `i0`, `c0` with every reference updated.
   Generated SVGs name their parts (`<!-- Bottle cap -->`, `id="bellGrad"`), so without this the task becomes
   reading words.
2. **Ask Jev** a fixed question set about that text, 25 questions to a call (`jevsvg/client.py`,
   `scripts/ask.py`).
3. **Classify**: a logistic regression on the standardised log-odds of the answers (`jevsvg/experiments.py`).

Diagrams of training and of classifying one SVG, and why these choices, are in [docs/method.md](docs/method.md).

## Reproducing

```sh
uv sync
uv run python scripts/evaluate.py    # every result, from data/, no Jev calls
uv run pytest                        # add -m slow for the corpus pilot
```

`data/` holds the manifests, the question sets and every recorded answer. The SVGs themselves are not in the
repository, because each source has its own licence ([docs/data.md](docs/data.md)). To fetch them and check that
preparing them reproduces the exact text Jev was shown:

```sh
uv sync --extra fetch --extra clip
uv run python scripts/fetch.py emoji      # about a minute
uv run python scripts/fetch.py rapidata   # about a minute: 438 cells of 20 parquet files, at a pinned revision
```

With the SVGs fetched, `evaluate.py` also computes the pixel, code, grid and CLIP rows of the Rapidata table,
which take tens of minutes on a CPU.

To ask new questions, or answer new items, set `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY` and run
`scripts/ask.py`.

## Layout

| Path | Contents |
|---|---|
| `jevsvg/` | SVG preparation, the Jev client, data loading, the experiments |
| `scripts/` | `evaluate.py`, `fetch.py`, `ask.py` |
| `data/manifests/` | one row per item: label, source, split, and the sha1 of the text Jev was shown |
| `data/questions/` | the question sets, by id |
| `data/answers/` | Jev's answers, per item, version and question set |
| `results/results.json` | what `evaluate.py` writes |
| `docs/` | method, data and licences, full results, and the exploration that led here |

## Limits

- The label set is closed: the classifier picks one of the labels it was trained on. Zero-shot, with no
  training, Jev reaches 39.7% on the Rapidata objects.
- The Rapidata SVGs are all drawn by LLMs, which may draw a lion more alike than human illustrators do; the
  result is untested on human-drawn SVGs.
- Test sets are small: 179 SVGs for Rapidata, 150 to 464 emoji per emoji result.
- An answer shifts slightly with the other questions asked in the same call (docs/method.md), so a different
  grouping of the same questions gives slightly different numbers.

## Licence

The code is MIT. The data sources keep their own licences: [docs/data.md](docs/data.md).
