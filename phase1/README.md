# Phase 1: bench check with Unlu et al.'s official code

These scripts run [EyupBunlu/QViT_HEP_ML4Sci](https://github.com/EyupBunlu/QViT_HEP_ML4Sci) (Unlu et al., 2024) on MNIST, to check that a published hybrid quantum ViT runs end to end on Kaggle.

- `unlu_validation.py`: trains the classical ViT and the hybrid (`hybrid2`) ViT at matched size (~3k params) and reports test accuracy from the best-validation checkpoint.
- `batch_timing.py`: times individual training steps of the hybrid model. Used to discover the one-time JAX compile cost.

## Kaggle cell (CPU session)
```
!pip install -q tensorcircuit-ng jax pennylane
!git clone -q https://github.com/EyupBunlu/QViT_HEP_ML4Sci.git /kaggle/working/QViT_HEP_ML4Sci
!sed -i -E "s/['\"]cuda['\"]/'cpu'/g" /kaggle/working/QViT_HEP_ML4Sci/QViT/*.py
!git clone -q https://github.com/manas-d-28/quantum-vit-bench.git /kaggle/working/bench
!cd /kaggle/working && python -u bench/phase1/unlu_validation.py
```

## Fixes needed on current Kaggle images (Python 3.13, NumPy 2)
1. `tensorcircuit` → `tensorcircuit-ng` (+ `jax`). The old package crashes on `np.ComplexWarning`.
2. Install `pennylane`, because the repo's `visual.py` imports it.
3. Use float32: float64 triggers a Double-vs-Float error inside the repo's model.
4. Some layers are lazy. Run one forward pass before counting parameters or creating the optimizer.
5. For CPU runs, replace the hard-coded `'cuda'` strings in the repo.

## Results (seed 0, Kaggle CPU)
| Epochs | Model | Params | s/epoch | Best val | Test |
|---|---|---|---|---|---|
| 30 | Classical | 2,986 | 2.0 | 91.1% | 88.9% |
| 30 | Hybrid2 | 3,114 | 21.4 | 80.7% | 76.9% |
| 100 | Classical | 2,986 | 1.8 | 91.6% | 88.6% |
| 100 | Hybrid2 | 3,114 | 18.9 | 84.0% | 81.4% |

The hybrid levels off around 82–84% validation accuracy from about epoch 45. The classical model levels off around 90–91% by about epoch 20. The first training step of the hybrid takes 1–2 minutes while JAX compiles the circuits. Deeper models compile much more slowly: at 4 layers with Embed 32, the compile took more than 4.5 minutes.
