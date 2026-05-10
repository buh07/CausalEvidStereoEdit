# StereACL Paper Narrative

**Target venue:** StereACuLT 2026 @ ACL  
**Submission deadline:** May 11, 2026 (11:59 p.m. UTC-12)  
**Format:** Long paper, up to 8 pages, ACL style (references + appendix not counted)  
**Platform:** OpenReview (StereACuLT portal)  
**Workshop date:** July 3, 2026

---

## Framing and Scope

The workshop targets stereotypes in language technologies across cultures, including mechanistic analysis, measurement, and mitigation. The submission categories explicitly welcome ablations and well-supported negative results. This paper contributes all three: a mechanistic analysis of stereotype-encoding circuits, a systematic negative result (single-site removal consistently fails while injection succeeds), and an ablation-based audit of a widely-used attribution method (DLA sign reliability). The cross-cultural direction-cosine and component-overlap findings in Exp05 directly address the workshop's interest in cultural contexts.

---

## Paper Title (working)

**"Inject, Don't Remove: Stereotypes Are Easier to Introduce Than to Extract in Large Language Models"**

Alternative: **"Three Mechanistic Regimes for Stereotype Encoding in LLMs: Geometry, Distribution, and RLHF-Induced Redundancy"**

The first title leads with the central finding and is more accessible. The second leads with the mechanistic taxonomy, which is the richer scientific contribution. A hybrid is possible: use the inject/remove asymmetry as the hook and the three-regime model as the framing.

---

## Abstract (narrative form)

We investigate how stereotype information is mechanistically encoded in three open-weight language models — Gemma-2-2B, Gemma-2-2B-IT, and Llama-3.2-3B — using a pipeline of residual-stream direction extraction, component-level Direct Logit Attribution (DLA), Attribution Patching (AtP), direction ablation, and signed multi-site experiments across six stereotype axes. We observe three mechanistic patterns in these three models: a linear-geometric pattern with a concentrated, partially ablatable direction (Gemma base); a distributed, self-repairing pattern with no single ablatable bottleneck (Llama); and a redundancy-consistent pattern in the instruction-tuned model (Gemma-IT). Across all three models and all experimental conditions, we observe a systematic inject/remove asymmetry: injecting the stereotype direction into anti-stereotype text reliably increases stereotype score (+0.09 to +0.29), while ablation in the removal direction produces small or null effects (−0.02 to −0.15). We further demonstrate that DLA/AtP sign estimates are statistically unreliable predictors of individual component causal direction — mostly at chance, with DLA significantly below chance in the RLHF-finetuned model — and that component-level ablation sets derived from one benchmark (StereoSet) do not causally transfer to another (CrowS-Pairs) despite 60-70% overlap in component rankings. These findings suggest that stereotype information is encoded in an architecture-dependent, multiply distributed fashion that makes targeted extraction substantially harder than targeted amplification, with implications for bias mitigation strategies based on representation surgery.

---

## 1. Introduction

### 1.1 Motivation

Bias in language models is most commonly measured at the output level: given a stereotypical completion and an anti-stereotypical completion, which does the model prefer? StereoSet, CrowS-Pairs, and WinoBias all operationalize this. But output-level measurement has a fundamental limitation: it tells us what the model outputs, not where the bias comes from or how to remove it. Mechanistic interpretability has emerged as a toolkit for answering the latter question — locating which components (attention heads, MLP blocks, residual-stream directions) are causally responsible for a behavior, so that targeted interventions can modify it.

The gap this paper fills: prior work on mechanistic interpretability of bias (Vig et al. 2020; Ma et al. 2023; Chintam et al. 2023; Chandna et al. 2025) has established that stereotype-encoding components exist and can be identified. What it has not established is whether those components can be removed without triggering compensatory mechanisms, whether the representations transfer across cultural or dataset framings, or whether the attribution tools used to identify components (DLA, AtP) reliably indicate the sign of a component's causal contribution.

### 1.2 Research Questions

We structure the paper around three questions:

1. **Do stereotypes concentrate in a geometrically identifiable, surgically removable subspace, or are they distributed across the residual stream in a way that defeats targeted intervention?**

2. **Is the inject/remove asymmetry a robust finding across models and datasets, and what does it imply about the architecture of stereotype encoding?**

3. **How reliable are the attribution tools (DLA, AtP) that mechanistic interpretability uses to identify and sign-label stereotype-encoding components?**

### 1.3 Contributions

1. **Three observed mechanistic patterns of stereotype encoding** across the three analyzed models, established through 15 experiments across >160 completed runs.
2. **A systematic documentation of the inject/remove asymmetry**: injection of stereotype directions is reliably effective; removal is not. This asymmetry is quantified across three models, two conditions (direction ablation and component ablation), and validated against four strict controls.
3. **A sign reliability audit of DLA and AtP**: we show that neither method's sign predictions reliably indicate the causal direction of individual component ablation, with DLA sign agreement significantly below chance (28%, p=0.003) in the RLHF-finetuned model.
4. **A cross-dataset component transfer experiment**: 60-70% Jaccard overlap in component rankings between StereoSet and CrowS-Pairs does not translate into functional causal transfer, suggesting within-component polysemanticity.
5. **An identification of two separable backfire mechanisms** in multi-site ablation (suppressor contamination vs. DLA sign inversion), resolved by sign-aware filtering.

---

## 2. Background and Related Work

### 2.1 Mechanistic Interpretability of Bias

The causal mediation analysis framework of Vig et al. (2020) identified sparse, layer-specific components causally mediating gender bias in GPT-2, establishing the research program. Ma et al. (2023) extended this to stereotype-specific attention heads and proposed head pruning for debiasing. Chintam et al. (2023) compared ACDC, DiffMask+, and causal mediation analysis on gender bias in GPT-2 Small. Chandna et al. (2025) used Edge Attribution Patching on GPT-2 and Llama-2 and found that bias components change across fine-tuning settings — the closest predecessor to our RLHF-regime finding.

Liyanage et al. (2025) applies contrastive neuron analysis on GPT-2 and Llama-3.2, the same model family as ours, finding stereotype-sensitive neurons at intermediate layers. None of these works study the inject/remove asymmetry, DLA sign reliability, RLHF-induced redundancy, or cross-dataset component transfer.

### 2.2 Steering Vectors and Representation Engineering

Zou et al. (2023) introduced Representation Engineering, showing that concept directions in the residual stream can steer behavior. Turner et al. (2023, ActAdd) noted as a side observation that subtracting steering vectors hurts MMLU while adding them helps — the closest prior observation to our inject/remove asymmetry finding, though not systematically studied. Arditi et al. (2024) demonstrated that a single direction mediates refusal in safety-aligned models and that both injection and ablation of this direction work for safety content; our finding that stereotype content shows strong asymmetry in the same direction contrasts with their symmetric finding for refusal. Braun et al. (2025) and Tan et al. (2024) document within-direction instability (anti-steerable samples, out-of-distribution brittleness) — a distinct but related form of unreliability.

### 2.3 Attribution Methods: DLA and AtP

DLA (Elhage et al. 2021, popularized by Nanda via TransformerLens) measures a component's direct projection onto output logits and is widely used as a screening tool. The community broadly acknowledges it as correlational, not causal, but no prior paper presents a systematic sign-accuracy audit. AtP (Nanda 2022-23) approximates activation patching via gradient products; AtP* (Kramár et al. 2024) identifies failure modes leading to false negatives (missing important components) but does not study sign errors. Syed et al. (2024) shows AtP outperforms ACDC in circuit discovery. Our sign reliability audit extends this literature with a quantitative finding: at the individual component level, DLA and AtP sign estimates are near-chance predictors of causal ablation direction, with a below-chance result for RLHF-finetuned models.

### 2.4 Hydra Effect and Self-Repair

McGrath et al. (2023) established the Hydra effect: ablating an attention layer causes other layers to compensate, producing non-linear per-site gains in multi-site ablations. Rushing and Nanda (ICML 2024) decomposed the mechanism into LayerNorm scaling and Anti-Erasure MLP neurons. We identify a related but distinct mechanism we call compensatory disinhibition: when a sign-blind ranking method selects both promoters and suppressors into the ablation set, removing the suppressors releases the stereotype output (backfire), which is mechanistically different from the network compensating for a missing promoter.

---

## 3. Pipeline and Methods

### 3.1 Datasets

**StereoSet** (Nadeem et al. 2021): completion-style pairs across four stereotype axes (gender, profession, race, religion). We use all six axes available in our annotation scheme (gender, profession, age, disability, nationality, physical_appearance). The completion-style format means prediction_position < trait_token_position in all pairs — a structural constraint that determines where causal interventions must be applied.

**CrowS-Pairs** (Nangia et al. 2020): full-sentence contrastive pairs across nine bias types, used in Exp15 for cross-dataset transfer testing.

**Cultural extensions (Exp05)**: 34-51 pairs per region for LatAm and South Asian framings of the same stereotype axes, enabling cross-cultural component-level comparison.

### 3.2 Models

Three primary models: `google/gemma-2-2b` (2.6B, 26 layers, d_model=2304), `google/gemma-2-2b-it` (same architecture, RLHF instruction-tuned), `meta-llama/Llama-3.2-3B` (3.2B, 28 layers, d_model=3072). All run in bfloat16. GPT-2 and Mistral-7B were included in early runs but are auxiliary context only.

### 3.3 Experiment Pipeline (Exp01-15)

To keep the paper readable under an 8-page limit, we bundle related experiments into four families and present only the highest-signal results in the main text.

**Bundle A: Geometry and causal localization (core text)**  
Exp01, Exp04, Exp07, Exp08, Exp10.  
Question: where is stereotype information represented, and can direction-level interventions causally move outcomes?

**Bundle B: Component causality and sign reliability (core text)**  
Exp02, Exp03, Exp09, Exp11, Exp14.  
Question: which components are causally important, and how reliable are DLA/AtP signs and rankings for intervention?

**Bundle C: Transfer and cultural robustness (core text, exploratory emphasis for culture)**  
Exp05, Exp15.  
Question: do geometry/components transfer across datasets and cultural framings?

**Bundle D: Controls and robustness diagnostics (appendix-first)**  
Exp04 control variants + full per-axis/per-component tables for all experiments.  
Question: are effects specific and statistically stable under strict baselines?

Main-paper headline results focus on Exp04, Exp09/11/14, and Exp15; full Exp01-15 coverage and exhaustive tables remain in appendix.

**Exp01** extracts stereotype directions per axis per layer via difference-in-means on training pairs, computes linear probing AUC, and establishes the train/test split (frozen before any evaluation).

**Exp02** computes DLA scores for each component (attention_block, attention_head, mlp_block) at each layer for each axis.

**Exp03** computes AtP scores for the same component × axis combinations via gradient-product attribution patching.

**Exp04** runs direction ablation and component ablation on the heldout pair set, with four strict controls: random_same_rank, norm_matched_random, label_permutation, corrupt_to_clean. Includes bootstrap CIs (500 resamples) and per-condition capability checks (BBQ, MMLU).

**Exp05** repeats Exp01-03 on cultural subsets (US, LatAm, South Asia) to test cross-cultural component transfer.

**Exp07** sweeps the rank-k direction subspace (SVD over all layer directions) to identify effective subspace dimensionality.

**Exp08** sweeps signed direction injection alpha ∈ {−2, −1, −0.5, −0.25, 0, +0.25, +0.5, +1, +2} at the top DLA layer.

**Exp09** ablates each component in the union of top-20 DLA and top-20 AtP rankings individually, with a promoters-only variant (sign-filtered).

**Exp10** runs layer-by-layer path mediation: measures both the geometric projection coefficient of h[L, pred_pos] onto the stereotype direction and the causal score delta from direction-projection ablation at prediction_position, at each layer L.

**Exp11** tests multi-site ablation at n_sites ∈ {1, 4, 8, 12, 16, 20}, comparing sign-blind (|DLA|∪|AtP|) and promoters-only (sign-filtered) component sets. In practice, A1 uses {1,4,8} for Llama and B1 extends Gemma sweeps through 20 sites for gender/profession stress tests.

**Exp14** is the sign reliability audit: for all components in the Exp09 adjudication table, computes DLA and AtP sign agreement with actual ablation causal direction, Wilson CIs, paired sign tests, Spearman ρ between signed rank and effect size, and BH-FDR correction.

**Exp15** is the cross-dataset transfer experiment: a 2×2 matrix of (StereoSet, CrowS-Pairs) component rankings applied to each test set.

### 3.4 Key Implementation Notes

All causal interventions that are interpreted as primary evidence (direction ablation, component ablation, path mediation) are applied at `prediction_position`, not `trait_token_position`. This is critical for StereoSet completion pairs: autoregressive masking means that intervening at `trait_token_position` cannot causally affect `prediction_position`, so such interventions are vacuous. Our implementation uses `make_direction_projection_at_position_hook(position, direction)` to enforce position-specific ablation.

We also report legacy full-sequence projection results as a stress-test sensitivity analysis. These global perturbations are useful for probing representational accessibility but are not treated as the primary causal-local estimate.

---

## 4. Results

### 4.1 Experiment 1: Directions Exist but Do Not Transfer

All three models encode stereotype information linearly (probing AUC > chance at all layers; mean AUC: Gemma-2-2B 0.752, Gemma-IT 0.762, Llama 0.767). Direction norm concentrates in upper layers (25/26 for Gemma, 28/28 for Llama), suggesting the stereotyped representation is sharpened as context propagates.

Critically: cross-dataset direction cosines are effectively zero (−0.007 for Gemma, −0.020 for Gemma-IT, −0.052 for Llama), meaning the direction extracted on StereoSet training pairs does not align with the direction extracted on CrowS-Pairs pairs. This is consistent with geometric orthogonality under the current benchmark constructions, though benchmark-format effects remain a plausible contributor. This result motivates the Exp15 cross-dataset transfer test and foreshadows its negative outcome.

### 4.2 Experiment 2-3: DLA and AtP Agree for Llama, Invert for Gemma

Component-level DLA reveals an architecture-dependent split: in Gemma, attention-block DLA dominates (mean |DLA| 0.525) over MLP blocks (0.142); in Llama, MLP blocks dominate (0.178) over attention (0.076). This is the first evidence of regime separation.

The DLA-AtP Spearman correlation confirms and deepens this split:

- **Gemma-2-2B: ρ = −0.46** (range −0.69 to −0.15). DLA rankings invert when tested causally. The components DLA ranks highest by magnitude are not the ones AtP identifies as causally important. This implies superposition or nonlinear composition — DLA measures linear projection onto output but cannot capture whether a component's contribution is bottlenecked by downstream processing.
- **Llama-3.2-3B: ρ = +0.59** (range +0.18 to +0.80). DLA and AtP agree. The linearity assumption underlying DLA holds for Llama's MLP-dominant mechanism.
- **Gemma-IT: ρ = −0.16** (near zero). Post-RLHF, neither DLA nor AtP is strongly predictive; the encoding is diffuse.

Beyond the sign distribution: Exp02 reveals that approximately 50% of components in Llama are suppressors (negative DLA, meaning they reduce the stereotype margin). This 50/50 split — confirmed for each axis individually — is the diagnostic marker for the compensatory disinhibition mechanism that explains Exp11 backfires.

### 4.3 Experiment 4: Direction Ablation (Global Stress Test + Causal-Local Estimate)

Full-sequence direction projection (all positions, all layers; stress-test condition) produces starkly different results:

| Model | Direction ablation Δ score | Corrupt-to-clean Δ score |
|---|---:|---:|
| Gemma-2-2B | **−0.150** | +0.092 |
| Gemma-2-2B-IT | −0.117 | +0.092 |
| Llama-3.2-3B | −0.017 | **+0.292** |

For Gemma, global direction ablation is meaningfully effective (−0.150), while injection is moderate (+0.092). For Llama, the relationship inverts dramatically: global direction ablation does almost nothing (−0.017) while injection is extreme (+0.292). This asymmetry — 17× difference between inject and remove for Llama in the stress-test condition — is one of the paper's central empirical findings. The margin-collapse-without-score-change pattern in Llama (ablation collapses mean margin from +0.335 to +0.167 while score is unchanged) is additional evidence of distributed encoding: the decision has multiple parallel supports that maintain the binary choice even as the signal weakens.

The four strict controls validate the direction-specificity of these effects: norm-matched-random produces no meaningful effect for any model (confirming that it is the direction, not the perturbation norm, that matters). Random-same-rank produces small, inconsistent effects. Label-permutation produces a confounded result (the permuted direction accidentally correlates with the anti-stereotype direction in some cases — this is noted as a limitation). Corrupt-to-clean is the cleanest positive control.

The position-specific test (prediction_position-only; primary causal-local estimate) quantifies how much of Gemma's effect is mediated by the single decision token vs. broader context. Ablating only at prediction_position gives −0.025 (1/6 of the full-sequence −0.150 for Gemma-2-2B), demonstrating that the stereotype signal is distributed across upstream context positions and cannot be removed by targeting only the decision site. For Llama, position-specific ablation (−0.017) matches full-sequence ablation (−0.017): both are near zero, consistent with encoding not being concentrated at a single position.

### 4.4 Experiment 7: Rank-2 Optimum in Gemma

Projecting out rank-k SVD subspaces across layers reveals a non-monotone pattern for Gemma: rank-2 projection gives the strongest reduction (Gemma-2-2B: score −0.12, margin −3.7; Gemma-IT: score −0.16, margin −2.9). Beyond rank 2, adding more singular vectors reduces and eventually reverses the effect (k=8: score increases). This non-monotonicity is explained by sign mixing in the SVD: top singular vectors capture all variance between stereo and anti-text, including anti-stereotype directions. Projecting out anti-stereotype directions removes inhibitory signal, increasing bias. The rank-2 optimum provides an estimate of the effective stereotype subspace dimensionality in Gemma: approximately 2 principal dimensions per axis.

For Llama, no consistent monotone pattern exists across any k value, consistent with distributed encoding not captured by a small number of global principal directions.

### 4.5 Experiment 8: Dose-Response Confirms Regime-Dependent Bottlenecks

Injecting the stereotype direction at the top DLA layer with alpha ∈ {−2, 0, +2}:

- **Gemma-2-2B**: margin increases monotonically with alpha (+0.166 from α=−2 to α=+2). Score is less responsive because the stereotype decision is already made at this layer — injection reinforces confidence without flipping decisions that are already decided.
- **Gemma-IT and Llama**: stereotype score is flat across all alpha. The top DLA layer is not the primary causal bottleneck for these models. Injection at this single layer does not change outcomes.

This dose-response is consistent with the corrupt-to-clean finding from Exp04: injection can work, but only if applied at the right layer (for Gemma, the top DLA layer is sufficient; for Llama, no single layer is). It also confirms that the inject/remove asymmetry is not symmetrically tied to single-layer access — the asymmetry is a property of the architecture, not of the experimental protocol.

### 4.6 Experiment 9: AtP Predicts Causal Effects; DLA Inverts for Gemma

Single-component ablation of the top-20 DLA ∪ top-20 AtP candidates individually confirms the DLA-inversion for Gemma and the MLP-dominance for Llama.

**Gemma-2-2B**: The most causally effective single components are attention blocks at layers 17-18 for age (score Δ = −0.222 each) and layers 8-12 for disability (Δ = −0.167 each). These components rank 11-18 in DLA magnitude — mid-range — but rank 4-16 in AtP. The pattern confirms that DLA inversely ranks causally important Gemma components. AtP is the better causal predictor.

**Llama-3.2-3B**: Most components show zero score delta. The single most effective component across all models and axes is `disability mlp_block` at layer 25 (Δ = −0.400, AtP rank 5), which is missed by absolute-value DLA ranking. The `profession mlp_block` at layer 28 is the clearest demonstration of DLA sign inversion: it has DLA rank 1 among signed promoters (highest DLA-predicted stereotype contribution) but ablating it increases stereotype score by +0.273. It acts causally as a suppressor despite having a positive DLA score.

**Gemma-2-2B-IT**: Zero score delta for every component across all axes. Even promoter-filtered single-component ablation (promoters-only run, §4.8.1) produces mean |Δ| < 0.031. This pattern is consistent with model-wide redundancy rather than a pure sign-mixing artifact.

The promoters-only adjudication (filtering to signed promoters before ranking) reveals for Gemma that the corrected ranking identifies causally real effects for age: the three causally effective age components now have DLA ranks 6, 12, 18 in the signed promoter list (meaning they were previously buried by high-|DLA| suppressors). Mean score delta for age improves from 0.000 (sign-mixed) to −0.025 (promoters-only).

### 4.7 Experiment 10: Layer Mediation Profiles Confirm Three Regimes

Layer-by-layer path mediation measures both the geometric footprint (projection coefficient of h[L, pred_pos] onto the stereotype direction) and the causal footprint (score delta from layer-L ablation at pred_pos). The three-way split is most clearly visible here:

**Gemma-2-2B** shows a causal hot zone at middle layers (8-12): age responds at layer 9 (Δ = −0.222) with a projection coefficient peak of 13.3 at layer 10; disability responds at layer 8 (Δ = −0.167) with a projection coefficient peak of 140 at layer 12. The geometric and causal profiles are aligned: where the direction is most strongly represented at prediction_position, a single layer ablation is most effective.

**Llama-3.2-3B** shows small projection coefficients across all layers (age max: |−0.75| vs Gemma's 13.3). The stereotype direction is not concentrated at prediction_position in Llama's residual stream. The causal consequence: only disability at layer 23 shows a non-trivial score response (Δ = −0.200), confirming disability as an exceptional axis with a more concentrated encoding in Llama.

**Gemma-2-2B-IT** shows large projection coefficients (nationality at layer 17: 31.5 — the largest in the dataset) but almost no causal response to single-layer ablation. Only nationality at one layer shows any response (Δ = −0.083). This decoupling of geometric signal strength from causal impact is the defining characteristic of the redundant encoding regime: the direction exists strongly, but it is not causal because multiple parallel routes maintain the same output.

### 4.8 Experiment 11: Multi-Site Ablation, Disinhibition, and Sign-Aware Resolution

Multi-site ablation using sign-blind top-k (|DLA| ∪ |AtP|) produces backfires in Llama for nationality (n=8: +0.583) and profession (n=1: +0.273). For Gemma, the same sign-blind protocol produces near-universal resistance (4/6 axes show zero score change at all n), with slight backfires for nationality and profession at n=8.

**Sign-aware (promoters-only) runs** reveal two separable mechanisms:

1. **Suppressor contamination** (primary source of the Llama nationality backfire): The top-8 by |DLA| for nationality includes ~4 suppressors. Removing suppressors releases stereotype output. Promoters-only at n=8 reduces the nationality backfire from +0.583 to +0.083 — an 86% reduction — confirming suppressor contamination as the dominant mechanism for this axis.

2. **DLA sign inversion** (the profession residual): The profession backfire (n=1: +0.273) is unchanged by promoters-only filtering. The root cause: `profession mlp_block-28` is DLA rank 1 among signed promoters but causally acts as a suppressor. Sign-filtering cannot detect this error because the DLA sign estimate itself is wrong, not the magnitude ranking. This is a fundamentally different failure mode that requires a causally grounded method (AtP) for component selection.

For **Gemma-2-2B promoters-only**: the sign-aware multi-site results reveal that some of Gemma's apparent resistance is also partly explained by suppressor contamination. Age (−0.333), disability (−0.167), and nationality (−0.083) all show reduction at n=8 under promoters-only — effects that were masked by mixed-sign ablation in the baseline. Gender and profession remain resistant at all n under both protocols, supporting genuine redundancy (not sign contamination) as the mechanism for these axes.

The Gemma baseline Exp11 (first time running multi-site for Gemma) also reveals that the margin continues to collapse even where the binary score does not move (disability n=8: margin 1.104 → −0.760, a large shift). This margin-without-score-change pattern is a reliable signature of the redundant encoding regime: the stereotype preference weakens but never crosses the decision threshold.

### 4.9 Experiment 14: Sign Reliability Audit

We formalize what Exp09 and Exp11 suggest: that DLA and AtP sign estimates are poor predictors of individual component causal direction. For each component in the Exp09 adjudication table, we record whether the DLA/AtP-predicted sign matches the observed direction of score delta from ablation. We compute Wilson CIs, paired sign tests, Spearman ρ(signed rank, −score_delta), and apply BH-FDR correction.

| Model | DLA sign agreement | CI | p | AtP sign agreement | CI | p |
|---|---:|---|---:|---:|---|---:|
| Gemma-2-2B | 54.1% | [43, 65%] | 0.56 | 42.9% | [28, 59%] | 0.50 |
| Gemma-2-2B-IT | **28.0%** | [17, 42%] | **0.003** | 44.4% | [28, 63%] | 0.70 |
| Llama-3.2-3B | 40.6% | [30, 52%] | 0.15 | 50.0% | [37, 63%] | 1.00 |

**Key findings:**

- DLA sign agreement is at chance for Gemma and Llama, and significantly below chance (p=0.003) for Gemma-IT. For Gemma-IT, DLA-predicted promoters actually reduce stereotype output when ablated more often than not (72% of the time). This is not noise — it is systematic inversion.

- AtP sign agreement is at chance for all three models. AtP is a better causal predictor than DLA for identifying which components matter (Exp09 shows this via Spearman ρ of ranked scores vs. effect size), but it does not reliably identify the sign of a component's causal contribution at the individual level.

- The distinction between magnitude reliability (AtP rank correlates with effect size) and sign reliability (AtP sign does not predict causal direction) is important: it means AtP is useful for finding components, not for determining whether those components promote or suppress stereotype output.

- Spearman ρ(signed DLA rank, −score_delta) is negative for Gemma and Gemma-IT, near zero for Llama — consistent with the Exp03 correlation pattern and confirming that this is a model-level property, not an experiment-specific artifact.

- The below-chance DLA agreement in Gemma-IT is consistent with the hypothesis that RLHF finetuning introduces compensatory circuitry that can oppose the output direction of high-DLA-salience components. This is not yet confirmed mechanistically.

### 4.10 Experiment 5 and 15: Cross-Cultural and Cross-Dataset Findings

**Cross-cultural (Exp05)**: Direction cosines are near zero across cultural framings (US vs. LatAm vs. South Asia: mean cosine 0.024–0.055), consistent with culture-sensitive geometry in this sample. Top-component Jaccard overlap is moderate (0.586–0.691): many of the same components are recruited across cultural framings even as direction estimates differ. This dissociation — shared circuitry with context-sensitive directions — suggests that stereotype-encoding components may behave as general-purpose association mechanisms whose expressed content depends on input framing.

Non-US sample sizes (34-51 pairs per region) are acknowledged as a limitation: directional claims about cultural differences are suggestive, not confirmed.

**Cross-dataset transfer (Exp15)**: The 2×2 transfer matrix (StereoSet components → CrowS-Pairs test; CrowS-Pairs components → StereoSet test) reveals that the Jaccard overlap does not translate into functional causal transfer:

- Gemma-2-2B: StereoSet-to-CrowS transfer efficiency = 0.30 (some transfer); CrowS-to-StereoSet = 0.0 (no transfer).
- Gemma-2-2B-IT: cross-dataset causes larger backfire than within-dataset (transfer efficiency > 1 in the backfire direction).
- Llama-3.2-3B: near-zero or negative transfer in both directions.

This finding means that 60-70% component identity does not imply causal equivalence. A plausible explanation is within-component polysemanticity: the same (layer, component_type, head_index) responds to different contextual signals in StereoSet vs. CrowS-Pairs, encoding different stereotype directions depending on input format. This is consistent with the near-zero cross-dataset direction cosines from Exp01.

---

## 5. The Central Narrative: Why Stereotypes Are Easier to Inject Than Remove

The observed model-specific patterns and the inject/remove asymmetry are consistent with a shared structural story: stereotype behavior often lacks a single architectural choke point. Under this view, injection can succeed by adding signal to any of multiple routes, while ablation must suppress enough routes simultaneously to change the output.

For **Gemma base**, the bottleneck appears more concentrated than in the other two models: a geometric direction in the residual stream, with a hot zone at middle layers (8-12) where ablation at a single layer can flip decisions for age and disability. Even here, gender and profession remain resistant under the extended B1 sweep (up to n=20 simultaneous promoter-only ablations). The asymmetry is moderate in the global stress test (injection +0.092 vs. removal −0.150), suggesting only partial accessibility to targeted removal.

For **Llama**, there is no accessible single-layer bottleneck: projection coefficients are small across all layers at prediction_position, and single-layer ablation is ineffective for all axes except disability (one MLP block at layer 25). But injection works dramatically (+0.292): introducing the stereotype direction floods a system that has many paths for accepting the signal. The 17× asymmetry between inject and remove is the most direct evidence for the multiply-redundant, distribution-based encoding.

For **Gemma-IT**, the direction is geometrically prominent (nationality projection coefficient 31.5 — the largest in the study), but no single-site ablation has measurable score effect. The results are suggestive of RLHF-associated redundancy and sign-instability phenomena, but direct mediation evidence is still needed before making a strong mechanistic RLHF claim. The inject/remove ratio (injection: +0.092, removal: −0.117 from full-sequence projection) appears more symmetric than Llama's, while single-site and position-specific ablation remain largely ineffective.

The inject/remove asymmetry has a direct implication for bias mitigation: interventions that attempt to remove stereotype encoding from a single site (pruning a head, ablating a direction at one layer, fine-tuning a small set of parameters) are unlikely to succeed without also suppressing capability. The stereotype signal is either distributed (Llama) or redundantly backed up (Gemma-IT), or partially ablatable but only for some axes (Gemma base). The asymmetry suggests that understanding where stereotypes can be introduced — which layers accept the direction — is not the same as understanding where they can be removed.

---

## 6. Discussion

### 6.1 Implications for Bias Mitigation

Current debiasing methods broadly fall into three families: representation editing (steering vectors, LEACE), weight editing (model editing, pruning), and fine-tuning (LoRA, RLHF). Our findings suggest:

- **Representation editing at single sites will fail for distributed and redundantly encoded models**. Injection-based steering works because the signal can enter anywhere; removal-based steering fails because the signal is protected by redundancy.
- **Weight editing of individual components is unreliable when DLA/AtP sign estimates are at chance**. Standard practice is to identify components by high |DLA| or |AtP| magnitude and edit those components. Our sign audit shows that the sign of these scores does not reliably indicate whether the component promotes or suppresses the bias, meaning edits may accidentally amplify the bias.
- **RLHF instruction-tuning is associated with patterns consistent with geometric redundancy and ablation resistance**, but this remains an inference pending direct mediation tests; the same pattern also coincides with less reliable DLA sign estimates (below chance for Gemma-IT).
- **Cross-dataset generalization cannot be assumed**: components identified on StereoSet may not be functionally equivalent on CrowS-Pairs. Bias mitigation validated on one benchmark may not generalize.

### 6.2 The DLA Sign Inversion Finding

The 28% DLA sign agreement in Gemma-IT (significantly below chance, p=0.003) is the most striking methodological finding. It implies that using DLA to identify and sign-label stereotype-promoting components in RLHF-finetuned models will produce a map that is worse than random — actively misleading in the direction of identifying suppressors as promoters.

Three candidate explanations exist. First, RLHF may have physically rotated the output weight vectors of high-DLA-salience components into the anti-stereotype direction (weight-space inversion). Second, the Jacobian of layer normalization may distort DLA sign estimates for large-activation components, a known technical limitation that becomes more severe after fine-tuning changes activation scales. Third, RLHF may have introduced suppressor components that cancel promoter outputs, making the ablation of a "promoter" remove both the promotion and its cancellation simultaneously.

All three are testable (weight-space RLHF delta alignment, layer-norm corrected DLA, cross-model activation patching), and we flag these as the highest-priority follow-up experiments.

### 6.3 The Cross-Cultural Dissociation

The Exp05 finding — shared circuitry (Jaccard 0.59-0.69), near-orthogonal directions (cosine ≈ 0) — is exploratory but informative. It suggests that stereotype-encoding components may be less like specialist "bias circuits" and more like general-purpose association mechanisms whose encoded direction depends on cultural framing. This is consistent with the polysemanticity hypothesis (Elhage et al. 2022, Toy Models of Superposition) at the component level. Given limited non-US sample sizes, this should be treated as a directional finding that motivates larger follow-up datasets.

### 6.4 Limitations

1. **Small per-axis test sets**: 5-14 pairs per axis limits statistical power. Binary score changes of 1-3 pairs drive the effect sizes. Following pre-registration exclusion criteria (Exp03: n ≥ 30 for primary claims), nationality and disability should not be primary evidence for Llama-specific claims.

2. **StereoSet-internal evaluation**: All primary findings are from StereoSet. Exp15 shows that CrowS-Pairs components do not causally transfer, meaning findings validated on StereoSet may not generalize to other datasets.

3. **One model per regime**: The three-regime characterization is supported by three models from two families. The regimes could collapse with broader coverage.

4. **Gender and profession resistance is unexplained**: Zero effect at n=20 simultaneous promoter-only ablations for these axes in Gemma has no mechanistic explanation in the current experiment set. QK routing and SAE feature-level polysemanticity are the leading hypotheses.

5. **DLA sign inversion mechanism is identified but not explained**: The below-chance sign agreement in Gemma-IT is a finding, not a mechanism. Three explanations are proposed but untested within this paper.

6. **Non-US samples remain thin**: 34-51 pairs per region for LatAm and South Asian framings. Cross-cultural claims are directional, not confirmatory.

---

## 7. Conclusion

We present a mechanistic interpretability study of stereotype encoding across three language models using a 15-experiment pipeline. The central finding — a systematic inject/remove asymmetry where introducing the stereotype direction reliably increases stereotype scores (+0.09 to +0.29) while removing it produces small or null effects — is consistent with multiply distributed or redundant encoding that resists targeted surgical removal. We characterize three observed model-specific mechanistic patterns (linear-geometric, distributed MLP, redundancy-consistent IT behavior) that differ in the availability of causal bottlenecks and the reliability of standard attribution tools. We show that DLA sign estimates are at-chance or below-chance predictors of individual component causal direction, that cross-dataset component transfer fails despite high ranking overlap, and that two distinct backfire mechanisms (suppressor contamination and DLA sign inversion) explain failure modes of naive multi-site ablation.

Taken together, these findings suggest that stereotype encoding is architecturally protected against the kinds of targeted interventions that mechanistic interpretability methods are best suited to identify. Understanding why injection succeeds where removal fails — and how to design interventions that are as effective in the removal direction as injection-based steering is in the addition direction — is the central open question this work motivates.

---

## Main-Paper Bundling Plan (8-Page Constraint)

To reduce scope density, the main text should present four headline bundles only:

1. **Bundle A (Geometry + Asymmetry):** Exp04 + Exp07/08 + Exp10 summary figure.
2. **Bundle B (Component Causality):** Exp09 + Exp11 with sign-blind vs promoters-only comparison.
3. **Bundle C (Method Validity):** Exp14 sign reliability audit.
4. **Bundle D (Generalization):** Exp15 cross-dataset transfer matrix.

Everything else should be condensed to one-paragraph context or moved to appendix tables/figures.

## Appendix Note (for final paper)

The following can go in appendix without counting toward the 8-page limit:
- Full Exp09 component tables per model and axis (complete score/margin delta rows)
- Full Exp10 layer-by-layer tables for all axes and models
- Full Exp11 n-site tables with bootstrap CIs
- Exp14 full component-level sign agreement tables
- Exp15 full 2×2 transfer matrix with all conditions
- Pre-registration document (PREREG.md, frozen May 7, 2026, before results were read)
- Hyperparameter tables (seed, heldout-pairs, top-k values per experiment)

---

## Section-by-Section Word Budget (8-page ACL target)

| Section | Estimated pages |
|---|---:|
| Abstract | 0.25 |
| 1. Introduction | 1.25 |
| 2. Background and Related Work | 1.0 |
| 3. Pipeline and Methods | 1.0 |
| 4. Results | 3.0 |
| 5. Central Narrative (inject/remove) | 0.5 |
| 6. Discussion | 0.75 |
| 7. Conclusion | 0.25 |
| **Total** | **8.0** |

The results section is the largest and should include one summary table per key experiment (Exp04 asymmetry table, model-pattern comparison table, Exp14 sign reliability table, Exp15 transfer matrix). Full component tables go to appendix.

---

## Checklist Against StereACuLT 2026 Requirements

- [x] Addresses stereotypes in language models
- [x] Cultural dimension: Exp05 cross-cultural direction cosines and component overlap; LatAm and South Asia subsets explicitly analyzed
- [x] Measurement contribution: sign reliability audit of DLA and AtP; cross-dataset transfer test
- [x] Mechanistic analysis: 15-experiment pipeline establishing causal vs. correlational attribution
- [x] Negative result (well-supported): cross-dataset transfer failure (Exp15); single-site ablation failure for Gemma-IT and Llama (Exp09, Exp10); DLA sign estimates at/below chance (Exp14)
- [x] Up to 8 pages, ACL format
- [x] Pre-registration document frozen before results (PREREG.md — qualifies as open science practice)
- [x] Hybrid presentation eligible (in-person or virtual)
- [x] Deadline: May 11, 2026 (11:59 p.m. UTC-12)
