# Results

Everything here is in `results/results.json`, written by `scripts/evaluate.py` from `data/`; a test fails if a
percentage on this page is not one of those results. Accuracy is the share of test items given the right label.

## Rapidata: 23 objects drawn by LLMs

438 SVGs of 23 single-object prompts. The model split trains on 259 SVGs from 25 models and tests on 179 from
16 others. The family split holds out one model family at a time and trains on the rest.

| Method | Model split | Family split |
|---|---|---|
| Jev zero-shot Choice over the 23 labels | 39.7% | - |
| corpus100 answers about the SVG code → logistic regression | 93.8% | 92.0% |

Classifiers that see no Jev answer, on the same splits (`rapidata_baselines`):

| Input | Model split | Family split |
|---|---|---|
| 16×16 rendered pixels | 73.7% | 78.1% |
| character 2- to 5-grams of the SVG code, TF-IDF | 77.6% | 66.7% |
| element counts, path-command counts and a colour histogram of the SVG code | 76.5% | 74.0% |

The rendering written as a character grid and asked the same corpus100 questions (`rapidata_grids`,
`jevsvg/grids.py`), against a classifier on the grid cells themselves:

| Grid | Jev zero-shot | Jev answers → classifier | Grid cells → classifier, no Jev | Jev grid answers with the SVG-code answers |
|---|---|---|---|---|
| 24×24, 12 named colours | 4.5% | 74.3% / 72.8% | 71.0% / 72.4% | 93.3% / 91.8% |
| 32×32, 9 brightness levels | 7.8% | 62.0% / 59.1% | 66.5% / 70.1% | 91.6% / 90.6% |

Pairs are model split / family split. On a grid Jev adds nothing a classifier on the grid lacks, and adding the
grid answers to the SVG-code answers lowers accuracy slightly.

CLIP ViT-B/32 with LAION-2B weights on 224×224 renderings (`rapidata_clip`): zero-shot against "an icon of a
<label>" 99.4%; a logistic regression on its image embeddings 100.0% on the model split and 99.1% on the family
split.

Family split, by the family held out:

| Family | SVGs | Accuracy |
|---|---|---|
| claude | 180 | 98.3% |
| gpt | 79 | 88.6% |
| quiver | 28 | 57.1% |
| kimi | 22 | 90.9% |
| qwen3 | 18 | 100.0% |
| gemini | 17 | 70.6% |
| sakana | 17 | 100.0% |
| glm | 15 | 93.3% |
| deepseek | 14 | 100.0% |
| grok | 13 | 76.9% |
| minimax | 10 | 100.0% |
| mimo, muse | 9 each | 100.0% |
| nvidia, hunyuan, mistral | 4, 2, 1 | 100.0% |

"Does this SVG show <label>?" against the human alignment score of the same SVG: Spearman 0.004 (p = 0.93,
438 SVGs). Jev answers yes for 73.7%. Every SVG here was rated 0.5 or more for alignment, which narrows the range
a correlation can use, but a working check would show one within it.

## Emoji: 14 subgroups in four styles

The emoji are OpenMoji, Twemoji, Fluent and Noto drawings of the same Unicode emoji, labelled by Unicode
subgroup. Guessing gets 1 in 14. All use the fixed100 questions.

### A fourth style

Test emoji are identities never seen in training in any style (`noto_heldout` in the manifest).

| Trained on | Training emoji | All held-out (464) | Noto held-out (150) |
|---|---|---|---|
| OpenMoji, Twemoji, Fluent | 603 | 49.8% | 36.0% |
| all four | 897 | 56.0% | 52.0% |

The second row also has more training emoji. With the same emoji identities allowed in training in other
styles, a classifier trained on the other three sets reaches 53.1% on 256 Noto emoji, so part of an unseen-style
score comes from recognising the same emoji drawn differently.

### Colour, one grey, and lightness

Trained on the 200 OpenMoji training emoji, tested on emoji not in training, in the version named. One grey sets
every fill and stroke colour except black to #9b9b9b; lightness replaces each colour with the grey of the same
lightness.

| Test set | Colour | One grey | Lightness |
|---|---|---|---|
| OpenMoji (197) | 57.4% | 56.9% | 58.4% |
| Twemoji (277) | 46.2% | 41.9% | 38.6% |
| Fluent (243) | 38.7% | 30.4% | 36.2% |

On OpenMoji's own style colour makes no difference. Twemoji and Fluent draw shapes only with fills, with no
black outlines, and there colour helps.

### Normalised SVG text

`normalise()` rewrites every SVG into one form (docs/method.md). Trained on three styles, tested on the fourth,
with test identities never seen in training (`e_heldout`):

| Tested on | Emoji | Raw | Normalised | Right only after / only before |
|---|---|---|---|---|
| OpenMoji | 136 | 36.8% | 40.4% | 16 / 11 |
| Twemoji | 83 | 43.4% | 54.2% | 16 / 7 |
| Fluent | 74 | 40.5% | 40.5% | 7 / 7 |
| Noto | 150 | 46.0% | 36.7% | 8 / 22 |
| mean of the four | | 41.7% | 43.0% | |
| all four styles in training (443 tested) | | 50.8% | 50.3% | |

Normalising does not help overall: it gains on Twemoji and loses on Noto, whose shading it flattens.

### Question sets on 200 emoji

The 200 emoji of the corpus pilot (`pilot_rank`), 14 or 15 per subgroup, 5-fold cross-validation repeated three
times; an emoji counts as right when it is right in most of the three repeats.

| Questions | Count | Accuracy |
|---|---|---|
| fixed100 | 100 | 40.5% |
| the whole corpus | 1,102 | 46.5% |
| 100 drawn at random from the corpus, three draws | 100 | 47.5%, 47.0%, 45.0% |
| icon tags only | 450 | 45.0% |
| ImageNet names only | 372 | 47.5% |
| jevlint texts only | 173 | 46.5% |
| visual only | 60 | 44.0% |
| colour and material only | 24 | 44.0% |
| kinds of thing only | 23 | 42.5% |

The corpus gets 20 emoji right that the fixed100 misses, and misses 8 it gets. A random 100 from the corpus does
as well as all 1,102: a mix of question kinds matters, the count does not, which is why the Rapidata runs use
corpus100.
