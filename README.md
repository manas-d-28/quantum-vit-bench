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
  - `qsann`: QSANN (Li et al. 2022), built from the paper (no official code). A 4-feature head is encoded on 2 qubits (d = n(D_enc+2) with D_enc = 0). U_q/U_k/U_v have depth 1. Q and K read ⟨Z₁⟩; V reads Z₁, Z₂, X₁, X₂. The Gaussian score is unscaled.
  - `comajoan`: Comajoan Cara et al. (2024), following their official code (ML4SCI/QMLHEP). Quantum layers replace the Q/K/V and output projections and sit inside the FFN (Dense → quantum → GELU → Dense). Each layer acts on the full embedding, one qubit per feature: 16 qubits, and 32 in the FFN. Each layer is Rx angle embedding, then one BasicEntangler layer (trainable Rx + CNOT ring), then ⟨Z⟩ on every qubit. Attention stays dot-product.
  - `hqvit`: HQViT (arXiv:2504.02730), built from the paper (no official code). All tokens are amplitude-encoded together, normalised by the whole image's norm (3 index qubits + 4 patch qubits). Block-diagonal PQCs U_Q/U_K/U_V act on the 4-qubit patch register. Scores are the swap-test overlaps |⟨q_i|k_j⟩|², softmaxed with no scaling. V is measured, giving probabilities |(U_V x_i)_l|². One full-width head. *Assumption:* the paper gives no gate list, so the PQC here is 2 layers of [Ry, Rz on every qubit, CNOT ring], then a final Ry layer. The 16×16 unitary is built and applied exactly in PyTorch (noiseless, infinite shots).
  - `overlap`: classical mimic of HQViT. Same formula and normalisation, but with free real matrices instead of unitaries.
  - `overlap_small`: the same mimic with rank-2 W_Q/W_K/W_V. The whole model has 3,978 params, the same as `classic` and within 7% of `hqvit`.
- Every circuit is checked against an independent NumPy statevector simulation (and HQViT's unitary against TensorCircuit): `python -m tools.test_circuits`

## Statistics and tuning (`tools/`)
- `qvb/train.py` saves per-example test predictions (`preds_*.npz`) for every run.
- `python -m tools.stats RESULTS_DIR` prints mean ± std and a **95% CI**. The CI comes from a two-level bootstrap: resample seeds, then test examples.
- `python -m tools.stats RESULTS_DIR --task T --compare A B` runs a **paired** comparison on identical test examples. It prints the 95% CI of the difference and a bootstrap p-value, and runs an **equivalence test**: a 90% CI inside ±margin points (TOST).
- `python -m tools.stats RESULTS_DIR --select-lr` picks each model's learning rate by mean **validation** accuracy. The test set is never used for tuning.
- `sh tools/run_grid.sh TASK "MODELS" "LRS" "SEEDS" EPOCHS OUT` runs the learning-rate search and the final seeds.

**Finding: Comajoan Cara's quantum layer is exactly classical.** Before the CNOTs the state is a product state with ⟨Z_j⟩ = cos(x_j + w_j). CNOTs only turn Z-strings into Z-strings, so every output is exactly a product of cosines, ⟨Z_i⟩ = ∏_{j∈S_i} cos(x_j + w_j). The sets S_i come from pushing Z_i back through the CNOT ring ([`qvb/comajoan.py`](qvb/comajoan.py)). The bench uses this formula, so their 16- and 32-qubit layers run at classical speed. It matches a full statevector simulation to ~1e-16. With a 16-qubit ring, most outputs multiply 2–16 cosines, so they tend to shrink toward 0.

## Results (shared backbone, MNIST, 60 epochs, seeds 0/1/2, Kaggle CPU)
| Attention | Params | Qubits / circuit | Circuits / image | s/epoch | Val acc | **Test acc** |
|---|---|---|---|---|---|---|
| `classic` (dot product) | 3,978 | — | — | 1.5 | 94.3 ± 0.3 | **92.3 ± 0.5** |
| `gaussian` (classical mimic) | 3,786 | — | — | 1.5 | 94.2 ± 0.8 | **93.3 ± 1.2** |
| `unlu` (quantum Q/K/V) | 3,898 | 4 | 168 | 15.4 | 95.9 ± 0.6 | **93.9 ± 0.2** |
| `qsann` (quantum, QSANN) | 3,738 | 2 | 168 | 11.9 | 93.9 ± 0.3 | **92.6 ± 1.0** |
| `comajoan` (quantum Q/K/V/O + FFN) | 3,242 | 16 / 32 | 70 | 2.2* | 92.5 ± 0.7 | **91.5 ± 0.7** |
| `hqvit` (quantum, swap-test attention) | 3,714 | 7+7+1 | 4 | 5.5† | 95.3 ± 0.2 | **93.8 ± 0.7** |
| `overlap` (classical mimic of HQViT) | 5,130 | — | — | 0.8 | 95.4 ± 0.8 | **93.5 ± 1.2** |
| `uniform` (**control**: fixed 1/T weights, no Q/K) | 3,722 | — | — | 0.9 | 94.9 ± 0.3 | **93.6 ± 0.9** |

Mean ± sample std over 3 seeds. Test accuracy is taken at the best-validation epoch, on 1000 test images, so one image is 0.1 points.
- **In the same backbone, the quantum model is no longer behind.** It scored about 1.6 points above dot-product attention, and every quantum seed beat every classic seed. In Phase 1 it was 7 points behind, but there the two models were also wired differently.
- **The classical mimic gets most of the gain.** It uses the same Gaussian scoring rule with ordinary weights, and it lands within ~0.6 points of the quantum model, which is inside the seed spread. So far, the scoring rule explains the improvement better than the qubits do.
- **The quantum model costs ~10× more training time** in simulation.
- **QSANN (2-qubit circuits) lands at 92.6 ± 1.0,** between classic and the mimic, and within the seed spread of both. Its seeds overlap with every other model's, so it shows no measurable gain from its qubits. Per-seed numbers: `results/phase2/` … `results/phase5/`.
- **Comajoan Cara (91.5 ± 0.7) is the lowest of the five, about 0.8 below dot-product attention, and its seeds overlap classic's.** It also has ~18% fewer parameters (3,242), because each quantum layer has one angle per qubit instead of a weight matrix. *Its 2.2 s/epoch uses the exact cosine-product formula, not circuit simulation. Its "quantum" layers are classically computable, so this model shows no quantum advantage by construction.
- **HQViT (93.8 ± 0.7) ties its classical mimic (93.5 ± 1.2)** and is level with Unlu. †Its 5.5 s/epoch is an exact PyTorch simulation of the 16×16 block unitaries, with infinite-shot swap tests. A real device would need many shots per score. **Caveat on what its attention does:** whole-image amplitude encoding gives each of the 7 (LayerNormed, roughly equal-norm) tokens about 1/7 of the state's weight, so |⟨q_i|k_j⟩|² ≲ ‖x_i‖²‖x_j‖²/‖X‖⁴ ≈ 1/49 ≈ 0.02. A softmax over numbers this small is close to uniform, so HQViT's attention should behave almost like averaging over tokens. Its good score therefore suggests that on this task the attention pattern matters little.
- **The uniform-attention control confirms it: 93.6 ± 0.9 with no queries or keys at all.** Every token just receives the average of all tokens' values. That ties Unlu, HQViT and both mimics, and beats dot-product attention. **So at this scale (7 tokens, 14×14 MNIST, ~4k params), the task cannot tell attention mechanisms apart.** The ties above are therefore not evidence that quantum attention matches classical attention in general. They only show that nothing here depends on the attention pattern. Next step: a harder setting where real attention clearly beats uniform, then re-run the modules there.

## Does the task need attention at all? (classic vs uniform, 3 seeds, 60 epochs)
| Setting | Tokens | Classic test | Uniform test | Classic − uniform |
|---|---|---|---|---|
| MNIST 14×14, 2×14 strips (main bench) | 7 | 92.3 ± 0.5 | 93.6 ± 0.9 | −1.3 |
| MNIST 16×16, 4×4 patches | 16 | 91.7 ± 0.3 | 91.8 ± 1.3 | −0.1 |
| Fashion-MNIST 14×14, 2×14 strips | 7 | 82.8 ± 0.5 | 84.3 ± 0.4 | −1.5 |
| Fashion-MNIST 16×16, 4×4 patches | 16 | 81.5 ± 0.9 | 83.0 ± 0.7 | −1.4 |

**In none of the four settings does real attention beat uniform averaging.** On Fashion-MNIST uniform is clearly better: in both settings every uniform seed beats every classic seed. At this scale (~4k params, 4k training images), the useful work is done per token by the embedding and FFN, followed by averaging. Learned attention only adds parameters that are hard to fit. So **classification accuracy at this scale cannot rank attention mechanisms**, quantum or classical. Raw: `results/control_harder/`.

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
python -m qvb.train --model classic --task pair_same --seed 0     # two digits: same number?
python -m qvb.train --model classic --task pair_larger --seed 0   # two digits: is left > right?
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
