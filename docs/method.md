# Method

## Why questions and a classifier

Jev answers one question about a state in one step well, and a question that needs an intermediate fact badly.
"Does this icon show a lion?" is one step. "What is the first letter of the thing this icon shows?" and "Is the
thing this icon shows alive?" are two: identify the thing, then judge it. Asked those, Jev's answers barely vary
with the icon ([exploration.md](exploration.md)). So every question here is one step about the SVG, and a
trained classifier does the combining that Jev does not.

## The pipeline

### Training: how the questions are weighted

The weights are fitted by the logistic regression from labelled examples; nobody sets them. The question set is
fixed before training (see Questions below).

```
 labelled SVGs (the Rapidata run: 259 SVGs from the training models, each tagged lion / fridge / rose ...)
        │
        ▼
 ┌──────────────────────┐
 │ prepare (code)       │  sanitise: comments, <title>, <text> removed, ids and classes renamed
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   100 yes/no questions, 25 per call: 4 calls per SVG
 │ Jev                  │   "Does this icon show an animal?"  "Is the drawing mostly orange?" ...
 └──────────┬───────────┘
            ▼
   answer table: one row per SVG, one column per question, each cell P(yes)
              q1     q2     q3   ...  q100   label
   svg 1     0.88   0.93   0.81  ...  0.10   lion
   svg 2     0.05   0.02   0.07  ...  0.64   fridge
            │
            ▼
 ┌──────────────────────┐
 │ log-odds (code)      │  x = log(p / (1 - p)): 0.5 → 0, a confident yes → large +, a confident no → large −
 │ standardise (code)   │  each column minus its training mean, divided by its training spread
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   one weight per label and question, and a bias per label
 │ logistic regression  │   C = 0.1 keeps the weights small, so no single question dominates
 └──────────┬───────────┘
            ▼
   weight table: labels × questions
              q1 orange  q2 animal  q3 face  ...  bias
   lion        +0.45      +0.40     +0.35    ...   0.00
   fridge      −0.30      −0.45     −0.20    ...  +0.30
   rose        −0.10      −0.35     −0.25    ...  +0.20
```

### Classifying one SVG

With three questions in place of 100. The numbers are made up to show the arithmetic, not taken from a run.

```
 an SVG from a model the classifier has not seen
        │
        ▼  prepare (code)
 state = {"svg": "<svg viewBox=...><path fill=\"#f08c1e\" d=\"M...\"/>...</svg>"}
        │
        ▼  one Jev call: three independent yes/no questions
 ┌────────────────────────────────────────────────────────────┐
 │ q1 "Is the drawing mostly orange?"            → P(yes) 0.88 │
 │ q2 "Does this icon show an animal?"            → P(yes) 0.93 │
 │ q3 "Does the drawing have a face with eyes?"   → P(yes) 0.81 │
 └────────────────────────────────────────────────────────────┘
        │   everything below is code; Jev is not asked again
        ▼  log-odds
   q1 = +1.99   q2 = +2.59   q3 = +1.45
        ▼  standardise with the training means and spreads
   q1 = +2.33   q2 = +2.06   q3 = +2.23
        ▼  score per label = Σ weight × value + bias
   lion   = 0.45×2.33 + 0.40×2.06 + 0.35×2.23 + 0.00 = +2.65
   fridge = −0.30×2.33 − 0.45×2.06 − 0.20×2.23 + 0.30 = −1.77
   rose   = −0.10×2.33 − 0.35×2.06 − 0.25×2.23 + 0.20 = −1.31
        ▼  softmax: scores to probabilities that sum to 1
   lion 0.970    rose 0.018    fridge 0.012
        ▼
 output: label "lion" with probability 0.97, and a probability for every label trained on
```

Jev answers only the yes/no questions; turning the answers into a label is arithmetic. In a composable-jev graph
the questions are `ask` nodes reading the context and the arithmetic is a `rule` node reading them.

## Preparing the SVG text

`jevsvg/svg.py`, `prepare(raw, recipe, version)`. The recipe depends on the source and is recorded per item in
the manifest:

| Recipe | Used for | Steps |
|---|---|---|
| `compact` | the 280 OpenMoji emoji of the first colour test | whitespace collapsed, ids and xmlns dropped, one decimal |
| `clean` | Twemoji, Fluent, the other OpenMoji emoji | the XML declaration and the root's size and style attributes removed, then `compact` |
| `noto` | Noto | `sanitise`, then `clean`: Noto files start with comments and a doctype |
| `rapidata` | Rapidata | `sanitise`, cut to start at `<svg` |

`sanitise` removes comments, `<title>`, `<desc>`, `<metadata>` and `<text>`, and renames ids and class names to
`i0`, `c0` with every `url(#...)`, `href` and CSS selector updated. A Rapidata SVG in which a word of its label,
four letters or longer, remains is left out of the data; one was. In Noto the only such words are SVG
element names, such as `circle` in the geometric emoji, which are kept.

The versions of an item:

- **colour**: the prepared text.
- **grey**: every colour except black set to #9b9b9b.
- **luma**: every colour set to the grey of the same lightness (0.299 R + 0.587 G + 0.114 B).
- **normalised**: gradients replaced by the mean of their stop colours; then
  [picosvg](https://github.com/googlefonts/picosvg) turns shapes into paths and resolves transforms, groups,
  strokes and clip paths; then the drawing is scaled into a 0-100 canvas with one decimal and only fill and
  opacity kept. Paths stay in their original order, because SVG paints later paths over earlier ones. On 80
  emoji from each set, a rendering of the normalised text differs from the original by a median of 0.000 to
  0.001 in mean pixel value (0-1 scale); the largest differences are Noto emoji whose shading was flattened.

## Questions

All questions are yes/no ("noul") with no criteria text. Every set is in `data/questions/`, by id.

- **fixed100**: 100 words drawn at random from the tags of the 200 OpenMoji training emoji, after Jev rated each
  tag word for "Is it a concrete, visible thing or symbol that could be drawn as a small picture?" and words at
  0.7 or more were kept. Asked as "Does this icon show a {word}?".
- **corpus**: 1,102 questions from six sources:
  - icon tags from OpenMoji emoji not used in any test, and from Lucide;
  - common nouns from the texts in the jevlint repository's cached datasets (decision-v7, SQuAD 2.0, FLORES-200);
  - ImageNet-1k class names shipped with open_clip, kept where Jev rated "Would most people recognise it in a
    simple drawing?" at 0.7 or more;
  - kinds of thing, from the DBpedia-14 and TREC categories plus broad kinds;
  - visual questions about the drawing's shape and parts;
  - colour and material questions.

  Words were filtered by the same concreteness question. Four questions have the same text as a question from
  another source ("Does this icon show an animal?" as an icon tag and as a kind of thing); each was asked
  separately, which is why answers are keyed by id.
- **corpus100**: the first 100 of the corpus in a fixed stratified order, where every prefix holds each source in
  proportion.

## Asking

`scripts/ask.py`: the state is `{"svg": <prepared text>}`, and questions go 25 to a call. Answers are stored per
question set, because an answer depends slightly on the other questions in its call. 9,200 answers were recorded
twice, for the same emoji and question in different calls: 30% were identical, the mean difference was 0.016
and the largest 0.16.

## Classifying

Each answer p becomes log(p / (1 - p)) with p clipped to [0.001, 0.999]; the columns are standardised; a
logistic regression with C = 0.1 predicts the label. Splits are stored in the manifests, so every result
reproduces exactly.
