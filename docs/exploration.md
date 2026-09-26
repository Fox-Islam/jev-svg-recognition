# Exploration

What was tried before settling on questions and a classifier. These numbers come from exploratory runs whose
code is not in this repository, on small hand-made sets, so they show direction, not precision.

## Pixels as text

| Encoding | Task | Result |
|---|---|---|
| 16×16 grid, `#` for ink and `.` for empty | 8 simple shapes | 32/32 |
| 32×32 grid, 5% of cells flipped | 8 simple shapes | 4/16 |
| 16×16 and 32×32 grids | hexagon, pentagon, octagon, star, heart, crescent … | 8/30, mostly "diamond" |
| brightness ASCII, edge maps, colour tiles, image statistics | 13 photographs | 2 to 4 of 13, near chance |

Jev reads the outline of a clean grid and nothing finer; noise and photographs defeat every encoding.

## SVG read directly

| Input | Result |
|---|---|
| 10 shapes as a bare `<polygon>` | 21/30; many-pointed outlines of curves read as "star" |
| 10 hand-written icons, Choice over their names | 9/10; 8 to 9/10 with colour, tag names or element order removed |
| 15 detailed hand-made icons, Choice over their names | 7/15 |

## Describing the icon first

- Jev naming each element's shape, one element per call: 36 of 36 right.
- Jev's own pairwise relations, compass direction and distance, then a Choice over names: 3/10.
- The same shapes with geometry written in code (size, position, touching, rings of repeated parts): 8/10.
- A profile of the depicted thing ("is it alive?", "what is it made of?") collapsed to the same answers for every
  icon, with every icon "metal" and "made by people", because the question needs the thing identified first.

## Spelling the name

| Method | Result |
|---|---|
| a 28-way Choice of the next letter | "house" spelled "hosue" with the word given |
| per step, 27 yes/no questions "does the name begin with exactly <prefix + letter>?", beam search | 6/6 exact with the word given |
| the same from the icon's SVG | word-shaped output, none of it the name; first letter 0/10 |
| a critic asking "is this English?" and "does this describe the image?", with retries | at most 5 plausible words in 10 (bus for a car, star for the sun); no retry improved on the first attempt |

Spelling works on a word already in the state, and not on a word Jev has to find first.

## Choosing questions

A search over question text kept the questions a classifier weighted most and mutated them. On 80 held-out
Lucide icons its questions beat the same number of random ones by 5 to 9 points, which is no more than the
spread between three random draws (8.7 points). On emoji it did not beat random questions at all.
