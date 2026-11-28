# Task-selection rule (written down before any pair-task results exist)

Committed 2026-11-27, before running the pair-task screen.

Goal: find a task where the attention mechanism matters, so that comparing
attention modules (quantum or classical) is meaningful.

Screen: `classic` (dot-product attention) vs `uniform` (fixed 1/T weights, no Q/K),
seeds 0, 1, 2, 60 epochs, lr 1e-3, on:
- `pair_same`   (are the two digits the same number?)
- `pair_larger` (is the left digit larger than the right?)

Each task: 10,000 train pairs, 2,000 val pairs, 10,000 test pairs built only from
the official MNIST test images. 14 tokens (7 per digit).

**A task qualifies if:**
1. mean test accuracy of `classic` exceeds `uniform` by **at least 3 points**, AND
2. **every** `classic` seed beats **every** `uniform` seed (no overlap).

If both qualify, use the one with the larger gap. If neither qualifies, try
the scale-up fallback (larger backbone / more data) under the same rule.

The chosen task is then used, unchanged, for the tuning and final runs of every
module (5 seeds each).

## Outcome (pair-task screen, run 2026-11-27, read 2026-11-28)
| Task | Classic (3 seeds) | Uniform (3 seeds) | Gap | Every classic seed > every uniform seed? | Qualifies? |
|---|---|---|---|---|---|
| pair_same | 85.5 ± 0.3 | 84.3 ± 0.6 | +1.2 | yes | **no** (gap < 3) |
| pair_larger | 89.5 ± 0.4 | 89.4 ± 0.3 | +0.1 | no | **no** |

Neither task meets the rule. `pair_same` is the first setting where attention beats
uniform consistently (all three classic seeds above all three uniform seeds), but the
gap is +1.2 points, below the 3-point threshold. As written above, the next step is
the scale-up fallback, under the same rule. Raw: `results/pair_screen/results.csv`.
