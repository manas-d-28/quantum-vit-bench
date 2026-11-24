# quantum-vit-bench

A small, fair benchmark of the **quantum self-attention** ideas proposed in quantum Vision Transformer papers. Each paper's quantum part is a swappable module inside **one shared tiny ViT** and is trained on **MNIST** with simulated quantum circuits.

This is the companion code for a B.Tech survey on quantum mechanics in Transformers and Vision Transformers.

## Why a shared backbone?
Each paper reports results on its own data, with its own model size and tricks, so their numbers can't be compared directly. Here the classical parts are fixed: patching, layer norm, classifier head, data split, training budget and parameter count. Only the attention mechanism changes, and each quantum module brings its own data encoding.

## Papers covered
| Module | Paper | Where the quantum part sits |
|---|---|---|
| QSANN | Li, Zhao & Wang (2022), arXiv:2205.05625 | Q/K/V from parameterised circuits; Gaussian-projected attention |
| Unlu | Unlu et al. (2024), arXiv:2402.00776 | Q/K/V from circuits; scores from squared query–key differences |
| Comajoan Cara | Comajoan Cara et al. (2024), arXiv:2405.10284 | Circuits replace the Q/K/V/output projections and the MLP |
| HQViT | HQViT (2025), arXiv:2504.02730 | Amplitude-encoded image; swap test gives the attention scores |
| Cherrat | Cherrat et al., *Quantum* 8, 1265 (2024), arXiv:2209.08167 | Orthogonal (RBS) layers; orthogonal and compound transformers |
| QSAN | Zhao, Shi & Zhang (2022), arXiv:2207.07563 | Logic-gate similarity in place of the dot product |

Classical "mimics" of each scoring trick are included, so we can tell whether a gain comes from the qubits or from the attention formula.

## Fairness rules
- Matched parameter count (~3k)
- Same data split and training budget
- 3 seeds
- Report test accuracy, parameters, qubits, circuit depth, circuits per image and wall time

## Status
- **Phase 1 (done): bench check with Unlu et al.'s official code** ([repo](https://github.com/EyupBunlu/QViT_HEP_ML4Sci)), MNIST 14×14, 4000/1000/1000 split, Kaggle CPU, seed 0:

  | Model | Params | s/epoch | Test acc (100 epochs) |
  |---|---|---|---|
  | Classical ViT | 2,986 | 1.8 | 88.6% |
  | Hybrid (quantum Q/K/V) | 3,114 | 18.9 | 81.4% |

  See [`phase1/`](phase1/) for the scripts and the fixes needed on current Kaggle images.
- **Phase 2 (next): shared backbone + classical baseline**, then the quantum modules one at a time.

## Running on Kaggle
Notebooks clone this repo and run a script. Quantum circuits are simulated with TensorCircuit-NG on JAX. These run on CPU, so **CPU sessions are recommended**: no GPU queue, and no GPU quota used.
