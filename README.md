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
- **Phase 2 (done for the first three modules): shared backbone** in [`qvb/`](qvb/). The first three attention modules:
  - `classic`: standard dot-product attention
  - `gaussian`: classical mimic of the QSANN/Unlu scoring rule
  - `unlu`: Unlu et al.'s quantum Q/K/V circuits, checked gate by gate against the official repo with [`tools/check_unlu_circuits.py`](tools/check_unlu_circuits.py)

## Phase 2 results (shared backbone, MNIST, 60 epochs, seeds 0/1/2, Kaggle CPU)
| Attention | Params | Qubits / circuit | Circuits / image | s/epoch | Val acc | **Test acc** |
|---|---|---|---|---|---|---|
| `classic` (dot product) | 3,978 | — | — | 1.5 | 94.3 ± 0.3 | **92.3 ± 0.5** |
| `gaussian` (classical mimic) | 3,786 | — | — | 1.5 | 94.2 ± 0.8 | **93.3 ± 1.2** |
| `unlu` (quantum Q/K/V) | 3,898 | 4 | 168 | 15.4 | 95.9 ± 0.6 | **93.9 ± 0.2** |

Mean ± sample std over 3 seeds. Test accuracy is taken at the best-validation epoch, on 1000 test images, so one image is 0.1 points.
- **In the same backbone, the quantum model is no longer behind.** It scored about 1.6 points above dot-product attention, and every quantum seed beat every classic seed. In Phase 1 it was 7 points behind, but there the two models were also wired differently.
- **The classical mimic gets most of the gain.** It uses the same Gaussian scoring rule with ordinary weights, and it lands within ~0.6 points of the quantum model, which is inside the seed spread. So far, the scoring rule explains the improvement better than the qubits do.
- **The quantum model costs ~10× more training time** in simulation.

## The shared backbone (`qvb/`)
- `data.py`: MNIST resized to 14×14 and cut into 7 row-patches (7 tokens × 28 features). Split 4000/1000/1000 with a fixed seed, so every model sees the same images.
- `backbone.py`: tiny pre-norm ViT, identical for every model:
  - token embedding + learned positional embedding
  - 2 blocks of `x + W_O·Attn(LN x)` and `x + FFN(LN x)`
  - mean pooling, then a linear classifier
- `attention.py`: the swappable attention modules. Every module splits the 16-dim embedding into 4 heads of 4 features, so each quantum circuit uses 4 qubits.
- `quantum.py`: circuits (TensorCircuit-NG on JAX) wrapped as PyTorch layers. Each circuit type compiles once and is shared by every layer.
- `train.py`: trains one model and appends a row to `results/results.csv`. The row records params, qubits per circuit, circuits per image, best val, test acc (best-val checkpoint) and time per epoch.

```
python -m qvb.train --model unlu --seed 0 --epochs 60
python -m qvb.train --model classic --quick        # 2-epoch smoke test on 640 images
```

**Differences from Unlu et al.'s own code.** These are deliberate, so that only the attention differs between models:
1. Their classical and hybrid blocks are wired differently. The hybrid has no separate attention residual or second LayerNorm. Here both use the same block.
2. Their positional embedding adds the same vector to every token. Here it's learned per token.
3. Their `head_dimension=8` with Embed 16 means 8 heads of 2 features (2-qubit circuits). Here it's 4 heads × 4 qubits, the setting of their paper's main result.

## Running on Kaggle
Notebooks clone this repo and run a script. Quantum circuits are simulated with TensorCircuit-NG on JAX. These run on CPU, so **CPU sessions are recommended**: no GPU queue, and no GPU quota used. Use a session with internet on:
```
!pip install -q tensorcircuit-ng jax
!git clone -q https://github.com/manas-d-28/quantum-vit-bench.git /kaggle/working/bench
!cd /kaggle/working/bench && python -m qvb.train --model classic --seed 0 --out /kaggle/working/results
```
