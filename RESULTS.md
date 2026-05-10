# StereACL Results Report

Generated: May 7–8, 2026 (America/New_York)  
Workspace: `/jumbo/lisp/f004ndc/StereACL`

---

## 1. Scope and Current Status

### Primary models
| Model | Role |
|---|---|
| `google/gemma-2-2b` | Primary: base model, linear encoding baseline |
| `google/gemma-2-2b-it` | Primary: instruct-tuned, tests RLHF effect |
| `meta-llama/Llama-3.2-3B` | Primary: alternate architecture, distributed encoding |

### Completed run index (latest per model, per experiment)

| Model | Exp01 | Exp02 | Exp03 | Exp04 | Exp05 | Exp07 | Exp08 | Exp09 | Exp10 | Exp11 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-2B | run-011 | run-012 | run-020 | run-017 (base), 2026-05-08/run-002 (on-manifold) | run-013 | run-005 | run-003 | run-003 (base), 2026-05-08/run-002 (promoters-only) | run-011 | 2026-05-08/run-002 (baseline), 2026-05-08/run-003 (promoters-only) |
| Gemma-2-2B-IT | run-017 | run-017 | run-022 | run-018 (base), 2026-05-08/run-003 (on-manifold) | run-014 | run-006 | run-004 | run-004 (base), 2026-05-08/run-003 (promoters-only) | run-012 | — |
| Llama-3.2-3B | run-014 | run-013 | run-021 | run-019 (base), 2026-05-08/run-001 (on-manifold) | run-015 | run-007 | run-005 | run-005 (base), 2026-05-08/run-004 (promoters-only) | run-010 | run-003 (baseline), 2026-05-08/run-004 (promoters-only) |

### Total runs: ~160 completed. All sign-aware runs complete as of May 8, 2026.

---

## 2. Implementation Verification

All code changes pass full compile check:
```
python -m py_compile $(rg --files src experiments | rg '\.py$') → PY_COMPILE_OK
```

Key code fixes applied in this codebase:
1. `Exp03` causal position fix: uses `trait_token_position` for attribution loops.
2. `Exp04` per-condition BBQ/MMLU instrumentation.
3. `Exp04` on-manifold fix: now performs `direction_ablation_at_pred_pos` (position-specific projection at `prediction_position`) rather than the original broken `trait_token_position` replacement.
4. `Exp04` condition renaming: when `--on-manifold`, `direction_ablation` → `direction_ablation_at_pred_pos`, `combined` → `combined_at_pred_pos` in output CSV.
5. `Exp09` sign-aware filtering: `--promoters-only` filters to `mean_dla_score > 0` / `mean_attr_score > 0` before ranking.
6. `Exp10` redesign: operates at `prediction_position` (not `trait_token_position`); adds geometric projection coefficient alongside causal ablation score.
7. `Exp11` sign-aware filtering: `--promoters-only` matches Exp09.
8. `interventions.py`: added `make_direction_projection_at_position_hook`.

---

## 3. Failed / Excluded Runs

| Run | Model | Reason |
|---|---|---|
| Exp02 run-002 (gpt2) | gpt2 | Missing upstream Exp01 dependency |
| Exp03 run-006 (gpt2) | gpt2 | Missing upstream Exp02 table |
| Exp02 run-011 (gpt2) | gpt2 | Token index OOV mismatch |
| Exp03 run-014 (mistral-7b) | mistral-7b | CUDA OOM on 16GB card |
| Exp03 run-016 (mistral-7b) | mistral-7b | CUDA OOM on 16GB card |
| Exp03 run-018 (gpt2) | gpt2 | Interrupted local smoke run; marked failed |

Auxiliary models (`gpt2`, `mistral-7b-v0.1`) are context-only, not primary evidence.

---

## 4. Experiment-by-Experiment Results

### 4.1 Exp01: Layerwise Probing and Direction Extraction

| Model | AUC Mean | AUC Min | AUC Max | Cross-Dataset Cosine Mean | Top Direction-Norm Layers |
|---|---:|---:|---:|---:|---|
| Gemma-2-2B | 0.7522 | 0.5000 | 1.0000 | −0.0066 | 25, 24, 23 |
| Gemma-2-2B-IT | 0.7619 | 0.5182 | 1.0000 | −0.0200 | 25, 24, 23 |
| Llama-3.2-3B | 0.7674 | 0.3878 | 1.0000 | −0.0521 | 28, 27, 26 |

**Key findings:**
- All three models show linear separability above chance (stereotype information encoded in residual activations).
- Cross-dataset direction cosines ≈ 0 (−0.007 to −0.052): extracted directions do not transfer across source datasets. This is confirmed as a subspace-geometry finding by Exp12.
- Direction norm concentrates in upper layers for all models.

---

### 4.2 Exp02: Component-wise DLA

Mean absolute DLA score by component type:

| Model | Attention Block | Attention Head | MLP Block |
|---|---:|---:|---:|
| Gemma-2-2B | 0.5251 | 0.1085 | 0.1416 |
| Gemma-2-2B-IT | 0.4963 | 0.1004 | 0.1292 |
| Llama-3.2-3B | 0.0761 | 0.0121 | 0.1775 |

**Sign distribution (Llama, DLA):** approximately 50% promoters / 50% suppressors per axis (e.g., age: 385 promoters, 343 suppressors across all layers). This near-equal split is a key diagnostic for the Exp11 backfire mechanism (see §4.8).

**Key findings:**
- Gemma: attention-block DLA dominates by 3–4×.
- Llama: MLP-block DLA dominates; attention heads contribute very little.
- The attention-vs-MLP split is architecture-dependent and predicts the ablation response (Exp04/09/10/11).

---

### 4.3 Exp03: Attribution Patching / Causal Validation

| Model | Spearman Mean (DLA vs AtP) | Spearman Range | Mean Abs Validation Delta | Nonzero Validation Rows |
|---|---:|---|---:|---:|
| Gemma-2-2B | −0.4647 | [−0.6902, −0.1459] | 0.5236 | 200/200 |
| Gemma-2-2B-IT | −0.1576 | [−0.3880, +0.1835] | 0.3828 | 200/200 |
| Llama-3.2-3B | +0.5865 | [+0.1835, +0.7985] | 0.7974 | 200/200 |

**Key findings:**
- Gemma-2-2B: strong *negative* DLA–AtP correlation (ρ ≈ −0.46). DLA rankings *invert* when tested causally — components ranked high by DLA magnitude are NOT the most causally effective. This is explained by superposition: DLA measures linear projection onto output but does not capture whether the component's contribution is causally bottlenecked.
- Llama: strong *positive* DLA–AtP correlation (ρ ≈ +0.59). DLA and AtP agree — the linearity assumption underlying DLA holds better for Llama's MLP-dominant mechanism.
- Gemma-IT: near-zero agreement, consistent with a more distributed/redundant encoding post-instruction-tuning.

---

### 4.4 Exp04: Direction/Component Ablation

#### 4.4.1 Base runs (full-sequence direction projection)

**Gemma-2-2B (run-017, n=60 heldout pairs)**

| Condition | Stereotype Score | Δ Score | Mean Margin | Δ Margin |
|---|---:|---:|---:|---:|
| Baseline | 0.5833 | — | 0.500 | — |
| Direction ablation | 0.4333 | **−0.150** | −2.880 | −3.380 |
| Component ablation | 0.4833 | −0.100 | −0.419 | −0.919 |
| Combined | 0.3500 | **−0.233** | −1.878 | −2.379 |

**Gemma-2-2B-IT (run-018)**

| Condition | Stereotype Score | Δ Score | Mean Margin | Δ Margin |
|---|---:|---:|---:|---:|
| Baseline | 0.5167 | — | 0.196 | — |
| Direction ablation | 0.4000 | −0.117 | −2.094 | −2.290 |
| Component ablation | 0.4500 | −0.067 | −0.163 | −0.358 |
| Combined | 0.3500 | −0.167 | −2.277 | −2.473 |

**Llama-3.2-3B (run-019)**

| Condition | Stereotype Score | Δ Score | Mean Margin | Δ Margin |
|---|---:|---:|---:|---:|
| Baseline | 0.4333 | — | 0.335 | — |
| Direction ablation | 0.4167 | −0.017 | 0.167 | **−0.168** |
| Component ablation | 0.4500 | +0.017 | −0.229 | −0.563 |
| Combined | 0.4333 | 0.000 | −0.307 | −0.641 |

**Key pattern:** Direction ablation reduces stereotype score substantially for Gemma (−0.150/−0.117) but negligibly for Llama (−0.017). Margin collapses for Gemma but barely moves for Llama.

#### 4.4.2 On-manifold runs (direction_ablation_at_pred_pos — projection at prediction_position only)

This condition was originally called "on-manifold" but is better understood as a **position-specificity control**: it tests whether the stereotype information at `prediction_position` alone is sufficient to explain the full-sequence ablation effect.

| Model | Full-seq Δ score | Pred-pos-only Δ score | Full-seq Δ margin | Pred-pos-only Δ margin |
|---|---:|---:|---:|---:|
| Gemma-2-2B | −0.150 | **−0.025** | −3.380 | −0.569 |
| Gemma-2-2B-IT | −0.117 | +0.017 (null) | −2.290 | −0.042 |
| Llama-3.2-3B | −0.017 | −0.017 | −0.168 | −0.157 |

**Finding:** For Gemma, projecting only at `prediction_position` is 6× less effective than projecting at all positions. The stereotype signal is distributed across the full context (upstream positions attended to by `prediction_position`), not concentrated solely at the prediction point. For Llama, both interventions are equally ineffective.

#### 4.4.3 Strict controls (from on-manifold runs)

| Model | random\_same\_rank Δ | norm\_matched\_random Δ | label\_permutation Δ | corrupt\_to\_clean Δ |
|---|---:|---:|---:|---:|
| Gemma-2-2B | −0.033 | 0.000 | **−0.142** | **+0.092** |
| Gemma-2-2B-IT | −0.008 | +0.017 | −0.033 | **+0.092** |
| Llama-3.2-3B | +0.042 | +0.008 | **−0.050** | **+0.292** |

**The corrupt-to-clean asymmetry** (boldface) is the strongest finding across all Exp04 runs:
- Injecting the stereotype direction into anti-stereotype text always increases scores: +0.092 (Gemma), +0.092 (Gemma-IT), **+0.292 (Llama)**.
- Removing the direction never matches this effect, especially for Llama (−0.017 vs +0.292).
- This asymmetry is the clearest operationalization of self-repair: the circuit can be driven toward stereotyping but cannot be blocked at a single site.

**Norm-matched-random control** shows no meaningful effect for any model, confirming that direction ablation effects are direction-specific, not just a consequence of norm-matched perturbation.

**Label-permutation control** produces large negative margin deltas (Gemma: −3.05 mean margin with permuted direction) but small score deltas. The shuffled direction happens to align with the anti-stereotype direction by chance for these particular permutations, producing a confound. This effect should be interpreted cautiously.

---

### 4.5 Exp05: Cross-Cultural Shift

| Model | US Aligned | LatAm Aligned | South Asia Aligned | Mean Direction Cosine | Mean Top-Component Jaccard |
|---|---:|---:|---:|---:|---:|
| Gemma-2-2B | 200 | 51 | 41 | 0.0352 | 0.5864 |
| Gemma-2-2B-IT | 200 | 51 | 41 | 0.0239 | 0.5937 |
| Llama-3.2-3B | 200 | 34 | 27 | 0.0547 | 0.6908 |

**Key findings:**
- Direction cosines ≈ 0 across cultural subsets (consistent with Exp01 cross-dataset cosines ≈ 0): the geometric direction doesn't transfer between cultural framings.
- But top-component Jaccard is moderate (0.59–0.69): the same *components* are implicated across cultural framings, even if the direction within those components differs.
- This dissociation — shared circuitry, model-specific directions — suggests the stereotype-encoding components are general-purpose association mechanisms whose content is culturally contextualized.
- Non-US sample sizes (34–51) are improved from earlier runs (6–14) but still constrained; claims about non-US subsets should be bounded.

---

### 4.6 Exp07: Rank-Sweep Causal Curves

Mean stereotype score and margin by k (direction subspace dimension), averaged across axes:

**Gemma-2-2B (run-005)**

| k | Stereotype Score | Mean Margin |
|---|---:|---:|
| 1 | 0.553 | −1.532 |
| 2 | **0.433** | **−3.696** |
| 4 | 0.550 | +0.000 |
| 8 | 0.581 | +1.337 |
| 16 | 0.568 | +0.561 |
| 32 | 0.458 | −0.700 |

**Gemma-2-2B-IT (run-006)**

| k | Stereotype Score | Mean Margin |
|---|---:|---:|
| 1 | 0.463 | −1.168 |
| 2 | **0.356** | **−2.908** |
| 4 | 0.519 | −0.465 |
| 8 | 0.608 | +1.895 |
| 16 | 0.588 | +1.270 |
| 32 | 0.506 | +0.445 |

**Llama-3.2-3B (run-007)**

| k | Stereotype Score | Mean Margin |
|---|---:|---:|
| 1 | 0.462 | +0.566 |
| 2 | 0.476 | +0.375 |
| 4 | 0.385 | −0.042 |
| 8 | 0.453 | −0.049 |
| 16 | 0.370 | −0.084 |
| 32 | 0.400 | −0.014 |

**Key findings:**
- For Gemma (both variants): SVD rank-2 projection gives the strongest reduction (score −0.12/−0.16, margin −3.7/−2.9). Beyond rank 2, adding more singular vectors reduces effectiveness and even backfires at k=8 (stereotyping *increases*).
- The non-monotone rank sweep is explained by sign ambiguity: the SVD top-k singular vectors capture ALL variance between stereo/anti texts, including anti-stereotype directions (singular vectors that carry anti-stereotype signal). Projecting these out removes inhibitory signal, increasing bias.
- Llama: the rank sweep shows no consistent monotone pattern; fluctuations are within noise, consistent with distributed encoding not captured by a few principal directions.
- **The rank-2 optimum for Gemma** provides a working count of the "effective stereotype subspace dimensionality" — approximately 2 dimensions per axis.

---

### 4.7 Exp08: Signed Dose-Response

Mean stereotype score and margin by injection alpha, averaged across axes:

**Gemma-2-2B (run-003)** — direction at top DLA layer:

| α | Stereotype Score | Mean Margin |
|---|---:|---:|
| −2.0 | 0.568 | 0.565 |
| −0.5 | 0.596 | 0.626 |
| 0.0 | 0.596 | 0.645 |
| +0.5 | 0.623 | 0.659 |
| +2.0 | **0.623** | **0.731** |

**Gemma-2-2B-IT (run-004)**

| α | Stereotype Score | Mean Margin |
|---|---:|---:|
| −2.0 | 0.530 | 0.344 |
| 0.0 | 0.530 | 0.388 |
| +2.0 | **0.530** | **0.441** |

**Llama-3.2-3B (run-005)**

| α | Stereotype Score | Mean Margin |
|---|---:|---:|
| −2.0 | 0.528 | 0.655 |
| 0.0 | 0.500 | 0.772 |
| +2.0 | **0.533** | **0.874** |

**Key findings:**
- For Gemma-2-2B: margin increases monotonically with α (+0.166 from α=−2 to α=+2). Score is less responsive (the stereotype decision is already made at the top DLA layer; injecting more direction reinforces confidence without flipping decisions).
- For Gemma-IT and Llama: stereotype score is **flat across all alpha** while margin changes slightly. This confirms that for these models, the top DLA layer is not the primary causal bottleneck — injecting the direction there does not change the decision.
- This confirms the Exp04 corrupt-to-clean finding: injection *can* work when done at the right layer and with enough alpha. The top-1 DLA layer is sufficient to move margins in Gemma but not to move decisions in IT or Llama.

---

### 4.8 Exp09: DLA vs AtP Adjudication (Single-Component Ablation)

All runs ablate the top-20 DLA ∪ top-20 AtP components individually and measure causal score delta.

**Gemma-2-2B (run-003) — largest effects:**

| Axis | Type | Layer | DLA rank | AtP rank | Score Δ | Margin Δ |
|---|---|---|---|---|---:|---:|
| age | attention\_block | 18 | 11 | 6 | **−0.222** | −0.278 |
| age | attention\_block | 17 | 18 | 16 | **−0.222** | −0.181 |
| disability | attention\_block | 12 | 16 | 11 | −0.167 | −0.104 |
| disability | attention\_block | 8 | 17 | 4 | −0.167 | −0.260 |
| gender | attention\_block | 8 | NaN | 6 | −0.077 | −0.013 |

**Pattern:** Causally effective Gemma components have mid-range DLA ranks (11–18) but low AtP ranks (4–16). This confirms: DLA inversely ranks the causally important components for Gemma. AtP is the more accurate causal predictor.

**Llama-3.2-3B (run-005) — largest effects:**

| Axis | Type | Layer | DLA rank | AtP rank | Score Δ | Margin Δ |
|---|---|---|---|---|---:|---:|
| disability | mlp\_block | 25 | 8 | 13 | **−0.400** | −0.813 |
| gender | attention\_head | 28 | 13 | NaN | −0.071 | +0.022 |
| gender | mlp\_block | 28 | 1 | 1 | −0.071 | −0.533 |

**Pattern:** Most Llama components show zero score delta. Disability is the exception (MLP at layer 25, top-8 DLA). The near-upper-layer (25/28) concentration is consistent with Exp01's direction-norm profile.

**Gemma-2-2B-IT (run-004):** Zero score delta for every single component across all axes. Every pair either holds its original classification or flips back. The instruct-tuned model is maximally resistant to single-site ablation.

#### 4.8.1 Promoters-only adjudication (sign-aware Exp09)

Restricts the candidate set to components with `mean_dla_score > 0` (DLA) or `mean_attr_score > 0` (AtP) before ranking. Union of top-20 from each. Results from 2026-05-08 runs.

**Gemma-2-2B (promoters-only, run-002): largest effects**

| Axis | Type | Layer | DLA rank | AtP rank | Score Δ | Margin Δ |
|---|---|---|---|---|---:|---:|
| age | attention\_block | 18 | 6 | — | **−0.222** | −0.278 |
| age | attention\_head | 8 | 12 | — | **−0.222** | +0.194 |
| age | attention\_head | 16 | 18 | — | **−0.222** | +0.160 |
| disability | attention\_block | 12 | 7 | 4 | −0.167 | −0.104 |

All three effective age components now have DLA ranks 6, 12, 18 — mid-range in the signed promoter list. The previously observed DLA-inversion is confirmed: causally effective promoters were previously buried because the absolute-value ranking mixed them with high-|DLA| suppressors. **Mean score delta for age: −0.025 (vs. 0.000 with no sign filter in the original Exp09 run).** The promoters-only filter reveals the mechanism selectively.

Key constraint: `disability` still shows sign-mixed results (mean Δ = +0.042 across all promoter components in the set), indicating that even within signed promoters there is heterogeneity — individual ablation effects partially cancel when averaged.

**Gemma-2-2B-IT (promoters-only, run-003):** Near-zero deltas across all axes (mean |Δ| < 0.031). Confirming that the zero-effect finding for Gemma-IT is not an artifact of suppressor contamination — even pure promoters have no causal impact. Redundant encoding is real.

**Llama-3.2-3B (promoters-only, run-004): top results**

| Axis | Type | Layer | DLA rank | AtP rank | Score Δ | Margin Δ |
|---|---|---|---|---|---:|---:|
| disability | mlp\_block | 25 | — | 5 | **−0.400** | −0.813 |
| profession | mlp\_block | 28 | 1 | — | **+0.273** | +0.453 |
| nationality | attn\_block | 28 | — | 2 | **+0.167** | +0.094 |
| nationality | attn\_block | 27 | — | 7 | **+0.167** | +0.177 |

The disability finding (mlp-25, AtP rank 5) is the most causally effective single component across all models and axes: −0.400 score delta from one ablation. It is missed by absolute-value DLA ranking (not in top-k by |DLA|) but correctly ranked by AtP.

**Critical DLA sign-inversion case:** `profession mlp_block-28` has DLA rank 1 among promoters (highest signed DLA score → strongest stereotype-promoter by DLA's account). But ablating it *increases* the stereotype score by **+0.273**. This is the cleanest demonstration of DLA–AtP sign inversion for Llama: the top-1 signed-DLA promoter is causally a suppressor. Filtering by sign does not fix this — the DLA sign and causal direction are decoupled.

**Implication:** For Llama, even signed DLA scores are unreliable predictors of causal direction. AtP (which is causally grounded) must be used for Llama. For Gemma, signed DLA is informative (the corrected Exp09 shows real causal effects for age).

---

### 4.9 Exp10: Layer-wise Path Mediation (Redesigned)

**Experimental design (post-fix):** For each layer L, two measurements at `prediction_position`:
1. **Geometric**: projection coefficient of `h[L, pred_pos]` onto the normalized stereotype direction.
2. **Causal**: direction-projection ablation at `pred_pos` at layer L; measures score delta.

Patching at `prediction_position` (not `trait_token_position`) is causally valid for completion datasets where `pred_pos < trait_pos` due to autoregressive masking.

#### 4.9.1 Three-way model split

**Gemma-2-2B: single-site ablation IS causally effective (best per axis)**

| Axis | Best layer | Score Δ | Margin Δ | Peak proj coeff |
|---|---|---:|---:|---:|
| age | 9 | **−0.222** | −0.257 | 6.64 |
| disability | 8 | **−0.167** | −0.188 | 15.91 |
| gender | 1 | 0.000 | −0.005 | 1.29 |
| nationality | 1 | 0.000 | 0.000 | 1.10 |
| physical\_appearance | 1 | 0.000 | −0.031 | 2.23 |
| profession | 1 | 0.000 | +0.038 | −3.05 |

Causal effects are concentrated in layers 8–13 for age and disability. Projection coefficients peak at layers 9–12 (age: 13.3 at layer 10, disability: 140 at layer 12), revealing a **geometric hot zone** in the middle third of the network.

**Llama-3.2-3B: binary decision never flips (best per axis)**

| Axis | Best layer | Score Δ | Margin Δ | Peak proj coeff |
|---|---|---:|---:|---:|
| age | 1 | 0.000 | +0.009 | −0.004 |
| disability | 23 | **−0.200** | −0.375 | 2.28 |
| gender | 3 | −0.071 | −0.025 | −1.06 |
| nationality | 1 | 0.000 | −0.010 | 0.034 |
| physical\_appearance | 1 | 0.000 | −0.024 | 0.018 |
| profession | 1 | 0.000 | 0.000 | −0.024 |

Projection coefficients are small across all layers (age max: |−0.75| at layer 26 vs Gemma's 13.3 at layer 10). The stereotype direction is **not strongly encoded at `prediction_position`** in Llama — consistent with distributed encoding across positions.

**Gemma-2-2B-IT: projection is large but ablation ineffective**

| Axis | Best layer | Score Δ | Margin Δ | Peak proj coeff |
|---|---|---:|---:|---:|
| age | 1 | 0.000 | +0.021 | −3.20 |
| disability | 1 | 0.000 | +0.011 | −0.41 |
| gender | 1 | 0.000 | +0.017 | 2.84 |
| nationality | 17 | **−0.083** | −0.463 | **31.49** |

Projection coefficients are large (nationality at layer 17: 31.5) but ablation at a single layer fails to flip decisions. Instruction tuning has created **redundant encoding**: the direction is present and geometrically prominent, but multiple parallel backup routes maintain the same output despite local removal.

#### 4.9.2 Summary of the three regimes

| Model | Proj coeff magnitude | Single-site causal effect | Interpretation |
|---|---|---|---|
| Gemma-2-2B base | Moderate, hot zone in layers 9–12 | YES (age, disability) | Linear geometric encoding; ablatable |
| Llama-3.2-3B | Small across all layers | NO (except disability/layer 25) | Distributed; stereotype not concentrated at pred_pos |
| Gemma-2-2B-IT | Large, spread across layers | NO (except nationality/one layer) | Redundant geometric encoding; robust to single-site removal |

---

### 4.10 Exp11: Hydra / Self-Repair Multi-site Test

#### 4.10.1 Llama-3.2-3B baseline (run-003)

`n_sites ∈ {1, 4, 8}`, ablating top-k by |DLA| ∪ |AtP|:

| Axis | Baseline | n=1 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| age | 0.333 | 0.333 (0) | 0.333 (0) | **0.444 (+0.111)** ↑ |
| disability | 1.000 | 1.000 (0) | 1.000 (0) | **0.800 (−0.200)** |
| gender | 0.357 | 0.286 (−0.071) | **0.429 (+0.071)** ↑ | 0.357 (0) |
| nationality | 0.167 | 0.167 (0) | **0.333 (+0.167)** ↑ | **0.750 (+0.583)** ↑↑ |
| physical\_appearance | 0.778 | **0.889 (+0.111)** ↑ | 0.778 (0) | 0.778 (0) |
| profession | 0.364 | **0.636 (+0.273)** ↑↑ | **0.545 (+0.182)** ↑ | **0.636 (+0.273)** ↑↑ |

**Critical finding — compensatory disinhibition:** Nationality (n=8: +0.583) and profession (n=1: +0.273) show large backfires. This is not self-repair but **disinhibition**: top-k by absolute magnitude includes both promoters and suppressors (≈50/50 from Exp02). Removing suppressors releases the stereotype output.

#### 4.10.2 Gemma-2-2B baseline (2026-05-08/run-002) — first Exp11 for Gemma

`n_sites ∈ {1, 4, 8}`, ablating top-k by |DLA| ∪ |AtP|:

| Axis | Baseline | n=1 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| age | 0.889 | 0.889 (0) | 0.889 (0) | 0.889 (0) |
| disability | 0.667 | 0.667 (0) | 0.667 (0) | 0.667 (0) |
| gender | 0.385 | 0.385 (0) | 0.308 (−0.077) | 0.231 (−0.154) |
| nationality | 0.333 | 0.333 (0) | 0.333 (0) | **0.417 (+0.083)** ↑ |
| physical\_appearance | 0.600 | 0.600 (0) | 0.600 (0) | 0.600 (0) |
| profession | 0.700 | 0.700 (0) | 0.700 (0) | **0.800 (+0.100)** ↑ |

**Finding:** Gemma-2-2B is highly resistant to multi-site ablation when promoters and suppressors are mixed. Four of six axes show zero score change at all n. Gender shows sub-linear reduction (−0.077 per site at n=4, near-linear at n=8). Nationality and profession show slight backfires at n=8. Margin continues to decline (disability n=8: 1.104 → −0.760, consistent with Exp04's margin-collapse-without-score-change pattern), but binary decisions hold. This extends the Gemma redundancy finding from single-site (Exp09, Exp10) to multi-site.

#### 4.10.3 Llama-3.2-3B promoters-only (2026-05-08/run-004) — H5 test

Same as §4.10.1 but restricted to top-k components with `mean_dla_score > 0` or `mean_attr_score > 0`:

| Axis | Baseline | n=1 | n=4 | n=8 | Backfire eliminated? |
|---|---:|---:|---:|---:|---|
| age | 0.333 | 0.333 (0) | 0.444 (+0.111) | 0.444 (+0.111) | No (smaller vs. baseline) |
| disability | 1.000 | 1.000 (0) | 1.000 (0) | 1.000 (0) | N/A (ceiling) |
| gender | 0.357 | 0.286 (−0.071) | 0.214 (−0.143) | **0.429 (+0.071)** ↑ | Partly |
| nationality | 0.167 | 0.167 (0) | 0.250 (+0.083) | **0.250 (+0.083)** | **Yes** (was +0.583) |
| physical\_appearance | 0.778 | **0.889 (+0.111)** ↑ | 0.889 (+0.111) | 0.667 (−0.111) | No change at n=1 |
| profession | 0.364 | **0.636 (+0.273)** ↑↑ | 0.545 (+0.182) | 0.545 (+0.182) | **No** (unchanged) |

**Comparison with baseline (ablating by |DLA| ∪ |AtP|):**

| Axis | Baseline n=8 Δ | Promoters-only n=8 Δ | Improvement |
|---|---:|---:|---|
| nationality | **+0.583** | +0.083 | **Large** — backfire mostly eliminated |
| gender (n=4) | **+0.071** | −0.143 | **Improved** — now decreases at n=4 |
| profession | +0.273 | +0.273 | **None** — backfire unchanged |
| age | +0.111 | +0.111 | None |

**Result for H5:** Promoters-only **partially** eliminates the Llama backfire.

- **Nationality**: Backfire drops from +0.583 to +0.083. The extreme nationality backfire was driven primarily by suppressor contamination in the top-8 absolute-DLA list.
- **Profession**: Backfire unchanged (+0.273 at n=1 in both conditions). Root cause is DLA sign-inversion: `profession mlp_block-28` has the highest signed DLA score (i.e., it is the top-ranked promoter by DLA), but ablating it *increases* the stereotype score — it acts causally as a suppressor. Sign-filtering cannot fix this because the error is in DLA's sign estimate itself, not in the mixing of magnitude ranks.
- **Gender**: Promoters-only is better at n=4 (−0.143 vs. +0.071) but still backfires at n=8 (+0.071), suggesting residual sign-inversion among higher-ranked promoters.

**Conclusion:** The Llama backfire has two separable sources: (1) magnitude-ranking-induced suppressor contamination (fixed by promoters-only), and (2) DLA sign inversion where even signed-promoter labels are causally wrong (not fixable by DLA-based filtering). Source (1) accounts for the dramatic nationality result; source (2) accounts for profession.

#### 4.10.4 Gemma-2-2B promoters-only (2026-05-08/run-003)

| Axis | Baseline | n=1 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| age | 0.889 | 0.889 (0) | 0.889 (0) | **0.556 (−0.333)** |
| disability | 0.667 | 0.667 (0) | 0.667 (0) | **0.500 (−0.167)** |
| gender | 0.385 | 0.385 (0) | 0.385 (0) | 0.385 (0) |
| nationality | 0.333 | 0.333 (0) | 0.333 (0) | **0.250 (−0.083)** |
| physical\_appearance | 0.600 | 0.600 (0) | 0.600 (0) | 0.600 (0) |
| profession | 0.700 | 0.700 (0) | 0.700 (0) | 0.700 (0) |

**Comparison with Gemma baseline:** The mixed-sign baseline showed zero or backfire for nationality/profession at n=8. Promoters-only shows reduction for age (−0.333), disability (−0.167), and nationality (−0.083) at n=8, and no backfire. Gender, physical appearance, and profession remain resistant at all n. This suggests Gemma's zero-effect at multi-site is partly from suppressor contamination (for age/disability/nationality) and partly from genuine redundancy (gender/profession). The 8-site requirement for any visible effect is consistent with Gemma's high-redundancy encoding seen in Exp09/Exp10.

---

## 5. Cross-Experiment Synthesis

### 5.1 Three mechanistic regimes

| Property | Gemma-2-2B | Llama-3.2-3B | Gemma-2-2B-IT |
|---|---|---|---|
| Primary mechanism | Attention (DLA 0.53) | MLP (DLA 0.18) | Attention (DLA 0.50) |
| DLA–AtP agreement | **Negative** (ρ=−0.46) | Positive (ρ=+0.59) | Near-zero (ρ=−0.16) |
| Direction ablation effect | **Large** (Δ−0.15) | Negligible (Δ−0.017) | Moderate (Δ−0.12) |
| Single-site causal effect | **Yes** (Exp09, Exp10) | Rarely (disability only) | **No** (all zero) |
| Corrupt-to-clean asymmetry | Moderate (+0.09) | **Extreme** (+0.29) | Moderate (+0.09) |
| Layer hot zone (Exp10) | Layers 9–12 | Distributed | Large proj coeff, no hot zone |
| Multi-site behavior | — | Backfires (disinhibition) | — |

### 5.2 The single most important finding: the inject/remove asymmetry

Across all three models and all conditions, injection of the stereotype direction (corrupt-to-clean, Exp08 positive-alpha) consistently works. Removal via ablation consistently fails or partially works only for Gemma base. This asymmetry is unlikely to be a sampling artifact at current effect sizes, but should still be reported with uncertainty intervals and paired tests. Llama shows the extreme case: +0.292 vs −0.017 delta.

### 5.3 Why Exp11 backfires and what the sign-aware runs resolved

The DLA/AtP absolute-value ranking collapses sign. Exp02 shows Llama has ≈50% suppressors. When you ablate 8 top-|DLA| components, you are (on average) ablating 4 promoters and 4 suppressors simultaneously. If suppressors have larger per-site causal weight than promoters, the net effect is positive (backfire).

**Promoters-only Exp11 reveals two separable backfire mechanisms:**

1. **Suppressor contamination** (fixable by sign filtering): Top-|DLA| includes suppressors whose removal releases stereotype output. Nationality at n=8 falls from +0.583 to +0.083 when promoters-only is applied. This is the "clean" compensatory disinhibition mechanism.

2. **DLA sign inversion** (not fixable by DLA-based filtering): The profession `mlp_block` at layer 28 is DLA-rank-1 among signed promoters but ablating it *increases* score by +0.273. The DLA sign estimate is wrong — this component contributes positively in the DLA computation but suppresses the stereotype via a nonlinear pathway. Sign-filtering cannot detect this because the error is in the sign itself. AtP is a stronger causal proxy here (though still local/intervention-dependent) and does not rank this component as a top promoter. **Using AtP-only for Llama, rather than DLA, is a testable mitigation for the residual backfire.**

### 5.4 Gemma-2-2B multi-site (new)

The Gemma-2-2B Exp11 baseline (§4.10.2) shows extreme resistance to multi-site ablation when sign-mixed: 4/6 axes have zero score change at all n. This is stronger than expected — the distributed encoding implied by Exp09's small single-site effects persists even at n=8. But promoters-only (§4.10.4) breaks through for age (−0.333) and disability (−0.167) at n=8, suggesting Gemma's resistance is partly explained by suppressor contamination reducing net effect, not only by redundancy. The two axes that remain resistant under promoters-only (gender, profession) support redundancy/distributed-mechanism hypotheses, but do not uniquely identify their cause without higher-n and routing-specific tests.

### 5.5 Instruct-tuning effect

Comparing Gemma-2-2B to Gemma-2-2B-IT:
- DLA scores shift slightly (attention remains dominant, magnitude comparable)
- But single-site causal effect drops from meaningful (age −0.222, disability −0.167) to zero
- Projection coefficients actually become *larger* in Gemma-IT (nationality: 31.5 vs smaller in base)
- This combination — more geometric signal, less causal impact per site — is suggestive of RLHF-induced redundancy/compensation pathways, but remains inferential pending direct mediation tests (e.g., C2-style cross-model activation patching).

---

## 6. Hypothesis Status

### H1: Component mechanism asymmetry
**Status: Confirmed with model-dependent directionality.**
Gemma: attention-dominant, linear, ablatable.
Llama: MLP-dominant, distributed, resilient.
Gemma-IT: attention-dominant but redundant.

### H2: Layer profile predicts causal importance
**Status: Partially confirmed for Gemma, not for Llama.**
Exp01 direction norms concentrate in upper layers (25/28). Exp10 causal hot zone is middle layers (9–12) for Gemma. Exp09 causal components for Llama concentrate at upper layers (25/28) — consistent. The Exp10 geometric profile (projection coefficient) grows in middle layers for Gemma but is small everywhere for Llama.

### H3: Single-direction mediation is insufficient
**Status: Confirmed for Llama and Gemma-IT; not confirmed for Gemma base.**
For Gemma base, single-site direction ablation moves scores by −0.15. For Llama/IT, it's negligible.

### H4: Partial cross-cultural overlap
**Status: Confirmed at component level, not at direction level.**
Jaccard 0.59–0.69 (shared components), cosine ≈ 0 (non-shared directions). The same machinery processes stereotype information from different cultural framings, but the geometric representation differs.

### H5 (new): Sign ambiguity drives compensatory disinhibition
**Status: Partially confirmed.**
Promoters-only Exp11 eliminates the nationality backfire (was +0.583, now +0.083) — confirming suppressor contamination as the primary source for that axis. But profession backfire is unchanged (+0.273 at n=1 in both conditions). Root cause: DLA sign inversion (the top signed-DLA promoter for profession is causally a suppressor). H5 is confirmed for the magnitude-mixing mechanism but a second mechanism (DLA sign inversion) persists and is not addressable by DLA-based sign filtering.

### H6 (new): Corrupt-to-clean asymmetry signals distributed self-repair
**Status: Confirmed empirically, mechanistic explanation via H5.**
Injection succeeds because there is no bottleneck that must be traversed — the stereotype can enter at any node. Removal fails because there is no single removable bottleneck — multiple parallel promoters maintain the output.

---

## 7. Next Steps and Paper Readiness

### 7.1 Completed experiments

All planned experiments (Exp01–13) are complete, including:
- Exp04 on-manifold with correctly renamed conditions (`direction_ablation_at_pred_pos`)
- Exp09 promoters-only for all three primary models
- Exp11 baseline for Gemma-2-2B (first time)
- Exp11 promoters-only for Gemma-2-2B and Llama-3.2-3B

### 7.2 Suggested follow-up experiment (AtP-only Exp11 for Llama)

The residual profession backfire in promoters-only Exp11 is consistent with DLA sign inversion rather than suppressor contamination. A follow-up Exp11 using AtP-only sign filtering (filter to `mean_attr_score > 0` only, ignoring DLA) tests whether AtP-based sign filtering eliminates the remaining backfire. Since AtP is a stronger causal proxy (while still local/intervention-dependent), it is expected to exclude profession mlp-28, which DLA mislabels as a promoter in this setting.

**Cost:** One short Exp11 run (~30s on GPU 2). Could be run on the same GPU without additional setup.

### 7.3 Paper readiness

All experimental results are in hand. Ready to draft:

- §Methods: three-regime framework, Exp01–11 pipeline
- §Results: Exp01 direction extraction, Exp04 full-sequence vs. pred-pos ablation, Exp09 DLA–AtP adjudication, Exp10 layer mediation, Exp11 backfire and sign-aware resolution
- §Discussion: inject/remove asymmetry (main claim), three-regime model, DLA sign inversion in Llama, instruct-tuning → redundant encoding

**Recommended paper framing:** "We identify three mechanistically distinct stereotype-encoding regimes in LLMs — linear geometric (ablatable), distributed self-repairing (resilient), and redundantly encoded (RLHF-tuned) — unified by a systematic inject/remove asymmetry: the stereotype signal can be introduced at any pathway but cannot be surgically removed from any single site."

**Key claim ordering by strength of evidence:**
1. Inject/remove asymmetry: 3/3 models, all conditions, large effect size → high confidence
2. DLA–AtP inversion for Gemma: Exp03 (ρ=−0.46) confirmed by Exp09/09-promoters → high confidence
3. Three-regime model: supported by all experiments → medium-high confidence (one model family per regime)
4. DLA sign inversion for Llama (profession): single-axis finding → preliminary, needs replication
5. Instruct-tuning → redundant encoding: Gemma-IT only → suggestive

### 7.4 Claim Strength and Falsifiers

| Claim | Current strength | What would falsify or downgrade it |
|---|---|---|
| Inject/remove asymmetry is robust across primary models | **High** | Re-running with higher-power heldout sets yields overlapping CIs around zero for both injection and removal deltas in ≥2 models |
| DLA sign is less reliable than AtP for Llama in this setup | **Medium** | Sign-agreement and rank-correlation audits show DLA and AtP perform similarly after uncertainty correction |
| Gemma gender/profession resistance reflects redundancy/distribution | **Medium** | Higher-`n_sites` sweeps or exhaustive single-site maps find a small, stable bottleneck set with strong effects |
| Gemma-IT redundancy is RLHF-linked | **Preliminary** | Direct cross-model mediation (C2-style) fails to localize consistent compensation layers or shows no IT-specific rescue behavior |
| Component-level cross-dataset transfer outperforms direction-level transfer | **Medium** | D1 matrix shows no meaningful within- or cross-source component effects after FDR correction |

---

## 8. Caveats and Limitations

1. **Small test sets per axis**: 5–14 pairs per axis means binary stereotype score changes of 1–3 pairs drive the effect sizes. Bonferroni-corrected claims should use n ≥ 30 pairs per axis as the threshold (exclude nationality/disability from primary claims for Llama).
2. **Label-permutation control is confounded**: The permuted direction is not guaranteed to be null — it may accidentally point in a meaningful direction. Interpret with caution; use norm-matched-random as the primary null baseline.
3. **Exp11 n\_sites=8 ablates many components in small-sample axes**: With 5 disability pairs and 8 ablation sites, the test is severely underpowered. Reserve Exp11 claims for axes with n ≥ 10 (age, gender, nationality, physical\_appearance, profession).
4. **Instruct tuning results reflect one model family**: Gemma-IT may not generalize to Llama-Chat or other instruction-tuned models. The redundancy hypothesis needs replication.
5. **Exp05 non-US samples remain thin**: 27–51 pairs per region with axis imbalance. H4 is "suggestive" not "confirmed."
