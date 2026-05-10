# StereACL Pre-Registration

Registered: 2026-05-07  
Authors: [redacted for blind review]  
Status: **FROZEN** — Do not update after first GPU run completes.

This document constitutes the analysis freeze for the extended experiment suite (Exp07–13 and Exp04-extended). It specifies hypotheses, planned tests, primary metrics, exclusion criteria, and multiple-comparison correction strategy. Any deviation from this specification in the final write-up must be flagged as post-hoc.

---

## 1. Primary Models

The following models are the primary evidence base:

1. `google/gemma-2-2b`
2. `google/gemma-2-2b-it`
3. `meta-llama/Llama-3.2-3B`

GPT-2 and Mistral-7B-v0.1 results are auxiliary context only and will not contribute to primary hypothesis verdicts.

---

## 2. Primary Metrics

In order of priority:

1. **`stereotype_score`** — fraction of pairs where the model assigns higher log-probability to the stereotypical completion. Range [0, 1]; 0.5 = chance.
2. **`mean_margin`** — mean log-probability difference (stereo − anti) over the held-out pair set. Signed; negative = anti-stereotypical lean.

Secondary metrics (reported but not primary evidence for H1–H9):
- `median_margin`
- `bbq_accuracy`, `mmlu_5shot_accuracy` (capability tradeoff)
- `spearman_rho` (for ranking agreement tests)
- Principal angle cosines (geometry)

---

## 3. Held-Out Pairs

- All evaluations use the **test split** defined by `train_test_split.json` from Exp01, limited to `--heldout-pairs` (typically 60–200 per model).
- The test/train split was determined before any model evaluation; it is fixed.
- Runs with **fewer than 30 valid evaluated pairs per condition** are excluded from primary hypothesis verdicts.

---

## 4. Hypotheses

### H1 — Component asymmetry (from RESULTS.md, now extended)
**Claim**: Attention blocks contribute more to stereotype DLA than MLP blocks in Gemma variants; MLP blocks dominate in Llama. This asymmetry is confirmed causally (not just correlationally) by single-component zero-ablation in Exp09.

**Test**: `09_dla_atp_adjudication.py` — single-site ablation `stereotype_score_delta` for attention_block vs mlp_block components. Confirmed if: Gemma mean |delta| for attention_block > mean |delta| for mlp_block; Llama reversal.

**New extension**: DLA and AtP rankings agree in Llama (Spearman ρ > 0) but disagree in Gemma (ρ < 0). Single-site causal ablation in Exp09 adjudicates which ranking is the better causal predictor.

**Verdict criterion**: For each model, compute Spearman ρ between (dla_rank, |stereotype_score_delta|) and (atp_rank, |stereotype_score_delta|). The method with higher ρ is the better causal predictor.

---

### H2 — Layer profile and late-layer emphasis
**Claim**: The stereotype causal effect concentrates in the upper 40% of layers (late layers), as indicated by Exp01 direction norms and Exp02 DLA scores.

**Test**: `10_path_mediation.py` — layer-wise residual path patching. Confirmed if: the largest `|mean_margin_delta|` values are concentrated in the top 40% of layer indices.

**Operationalization**: Sort layers by `|mean_margin_delta|` descending; confirmed if ≥ 60% of the top-5 most-effective layers fall in layers with index > 0.6 × n_layers.

---

### H3 — Single-direction mediation is insufficient
**Claim**: Projecting out the rank-1 stereotype direction does not saturate the causal effect. Higher ranks k > 1 produce progressively larger reductions in stereotype_score, indicating multi-dimensional encoding.

**Test**: `07_rank_sweep.py` — monotonic decrease of stereotype_score as k increases from 1 to 32. Confirmed if: mean(stereotype_score at k=16) < mean(stereotype_score at k=1) − 0.05, across primary models.

---

### H4 — Partial cross-cultural overlap
**Claim**: US and non-US (LatAm, South Asia) stereotype directions share partial subspace structure (mean principal angle cosine > 0.2) but are not identical (mean cosine < 0.9).

**Test**: `12_local_atlas.py` — cross-layer principal angle cosines. Confirmed if: 0.2 < mean_cos < 0.9 for cross-axis comparisons.

**Note**: As of the pre-registration, Exp05 cultural samples are 34–51 per region — sufficient for directional claims but acknowledged as limited.

---

### H5 — Signed dose-response monotonicity
**Claim**: Injecting the stereotype direction (alpha > 0) increases stereotype_score; removing it (alpha < 0, via projection) decreases it. The effect is monotone in alpha.

**Test**: `08_dose_response.py` — Spearman ρ between alpha and stereotype_score across the 9-point grid. Confirmed if: ρ > 0.8 for each primary model × axis combination, and the 0-alpha condition falls within ±0.05 of the unpatched baseline.

---

### H6 — Hydra / self-repair is present in Llama
**Claim**: Multi-site ablation in Llama produces super-linear per-site stereotype_score reduction (per_site_score_gain increases with n_sites), consistent with compensatory self-repair at single-site ablations.

**Test**: `11_hydra_multisite.py` — compare per_site_score_gain at n_sites=1 vs n_sites=4 vs n_sites=8. Confirmed if: mean per_site_score_gain at n_sites=4 > 1.5 × mean per_site_score_gain at n_sites=1 for Llama.

**Expected non-result for Gemma**: Gemma does not show this pattern (consistent with Exp04 showing strong direction ablation effect in Gemma without dissociation).

---

### H7 — DLA is not a valid causal predictor for Gemma
**Claim**: In Gemma, the DLA ranking (from Exp02) has lower causal predictive validity than the AtP ranking (from Exp03), as adjudicated by single-component ablation in Exp09.

**Test**: Spearman ρ between (mean_abs_dla_score, |stereotype_score_delta|) vs (mean_abs_attr_score, |stereotype_score_delta|). Confirmed if: ρ_AtP > ρ_DLA + 0.2 for Gemma-2-2B.

---

### H8 — On-manifold intervention is more effective than direction projection
**Claim**: Replacing the stereotype residual with the matched anti-text residual (on-manifold) produces a larger reduction in stereotype_score than projecting out the stereotype direction, because it stays within the natural manifold of the model's activations.

**Test**: Exp04 extended with `--on-manifold` flag. Confirmed if: on-manifold direction_ablation stereotype_score < direction-projection direction_ablation stereotype_score by ≥ 0.05, for at least 2 of 3 primary models.

---

### H9 — Strict controls confirm specificity
**Claim**: The four strict controls (random_same_rank, norm_matched_random, label_permutation, corrupt_to_clean) produce effects substantially smaller than the real direction/component ablation, confirming that the intervention is specific to stereotype encoding rather than a generic disruption.

**Test**: Exp04 extended with `--strict-controls`. Confirmed if: for each primary model, |stereotype_score_delta| for direction_ablation > 2 × max(|stereotype_score_delta| for random_same_rank, norm_matched_random, label_permutation).

---

### H10 — Directions do not transfer cross-model
**Claim**: Applying Gemma's stereotype direction to Llama (at matched layer fraction) produces a negligible causal effect on stereotype_score (|delta| < 0.05), confirming that the directions are model-specific.

**Test**: `13_cross_model_transfer.py` direction transfer experiment. Confirmed if: |mean stereotype_score_delta| < 0.05 across axes.

---

## 5. Multiple Comparison Correction

- For H1–H4, H6–H10: Bonferroni correction over the number of axes (typically 4–8) × the number of primary models (3). A corrected p < 0.05 is required for strong claims; effect size (Cohen's d or magnitude of delta) will be reported regardless.
- For H5 (monotonicity): Spearman ρ reported directly; no threshold correction.
- All exploratory analyses beyond the above hypotheses are labeled **post-hoc** in the paper.

---

## 6. Exclusion Criteria

The following run results are excluded from primary hypothesis evaluations:

1. Runs with `status != "completed"` in `manifest.json`.
2. Conditions with `n_pairs < 30` in any output CSV.
3. Auxiliary models (gpt2, mistral-7b-v0.1) in all primary comparisons.
4. Any Exp12 comparison where `n_principal_angles < 2` (too few directions to form a subspace).
5. Any Exp13 direction transfer result where the source direction was zero-padded by > 25% of its original dimension (reports a note, not excluded from the table, but excluded from H10 verdict).

---

## 7. Capability Tradeoff Audit (Exp04 extended)

For every condition where `stereotype_score_delta < −0.05` (meaningful de-biasing), the corresponding `bbq_accuracy` and `mmlu_5shot_accuracy` values must be reported alongside the stereotype metric. A condition is "acceptable" only if both capability metrics remain within 5 percentage points of the baseline condition.

---

## 8. Claim Matrix

The following table maps each paper claim to the required evidence:

| Claim | Required Exp | Required Result |
|---|---|---|
| Attention > MLP in Gemma (causal) | Exp09 | |attn_delta| > |mlp_delta| for Gemma |
| MLP > Attn in Llama (causal) | Exp09 | |mlp_delta| > |attn_delta| for Llama |
| AtP beats DLA as causal predictor in Gemma | Exp09 | ρ_AtP > ρ_DLA + 0.2 |
| Late-layer concentration of causal effect | Exp10 | ≥3 of top-5 causal layers in upper 40% |
| Multi-dim stereotype encoding (rank > 1) | Exp07 | score(k=16) < score(k=1) − 0.05 |
| Dose-response monotone | Exp08 | Spearman ρ > 0.8 per model |
| Hydra self-repair in Llama | Exp11 | per_site_gain(4) > 1.5× per_site_gain(1) |
| Controls confirm specificity | Exp04-strict | direction_ablation delta > 2× all controls |
| On-manifold superior to projection | Exp04-on-manifold | on_manifold_score < projection_score − 0.05 |
| Directions model-specific | Exp13 | |transfer_delta| < 0.05 |
| Partial cross-cultural subspace overlap | Exp12 | 0.2 < mean_cos < 0.9 |

---

*This document is sealed. Any departures from this specification in the final paper must be explicitly noted as post-hoc in the "Limitations" section.*
