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
