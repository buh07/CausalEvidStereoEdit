# Stereotype Localization in LLMs via Residual-Stream Geometry: A 1-Week Workshop Plan

**Target venue:** StereACuLT 2026 (1st Workshop on Stereotypes Across Cultures in Language Technologies, co-located with ACL 2026, San Diego)

**Format:** Short paper (≤4 pages of content + unlimited references/appendix) — feasible in 1 week. Long paper (≤8 pages) is achievable only if Days 6–7 are stretched and writing efficiency is high.

**Author background:** ML/AI; no human-subjects component.

---

## 1. Research Question and Hypothesis

**Central question.** When a modern decoder-only LLM produces a stereotypical completion, which components — attention heads or MLPs — are most responsible for *writing* the stereotype-relevant signal into the residual stream, and at which layers?

**Reframing this geometrically.** Treating the residual stream as the central object (Elhage et al. 2021), every component at every layer makes an additive write into a shared vector space. We define a **stereotype direction** per bias axis (gender, race, nationality, religion) using difference-in-means between stereotypical and anti-stereotypical contrast pairs, then ask: how much of each component's per-layer write projects onto this direction, and how much does each component's write change the model's stereotypical-vs-anti-stereotypical logit difference?

**Working hypotheses.**
1. **H1 (component asymmetry):** MLP writes will dominate the stereotype-direction projection in mid-upper layers (consistent with DAMA's FFN finding on LLaMA-1), while attention heads will dominate at *information-routing* layers earlier in the network (consistent with Vig 2020, Ma 2023, Yang 2025, BiasGym 2025 attention-head findings).
2. **H2 (layer profile):** Linear probability of stereotype identity rises sharply in middle layers and plateaus, mirroring findings from Marks & Tegmark's truth-direction analysis and Dubey 2025's GPT-2 large bias-probing curve.
3. **H3 (single-direction mediation):** Stereotypes will *not* be mediated by a single direction in the manner of Arditi et al.'s refusal direction; instead, ablating along the difference-in-means direction will only partially reduce stereotypical completions, consistent with D'Souza's 2026 ablation paradox.
4. **H4 (cross-cultural shift, stretch):** The implicated layers/components for culture-specific stereotypes (e.g., LatAm stereotypes from EspañStereo, regional stereotypes from SeeGULL Multilingual) will *partially overlap* with those for US/English stereotypes, suggesting both shared and culture-specific mechanisms.

Hypotheses 1–3 are the core; H4 is a stretch contribution that strengthens the StereACuLT cultural framing.

---

## 2. Related Work

### 2.1 Bias / stereotype localization in LLM internals

The literature splits across attention-only, MLP-only, and neuron-only viewpoints, and no prior paper has produced a clean attention-vs-MLP write decomposition on a modern decoder LLM using residual-stream geometry.

- **Vig et al. (NeurIPS 2020).** Causal mediation analysis on GPT-2; identified gender bias as concentrated in a sparse subnetwork. Foundational but neuron-level on small encoders.
- **Ma, Scheible, Vosoughi et al. (EMNLP 2023).** "Deciphering Stereotypes" — attribution to attention heads in BERT/RoBERTa/T5; head pruning as debiasing. **Workshop organizer's prior work**; cite prominently. Limitation for our purposes: encoder-only, attention-only, no residual geometry.
- **Limisiewicz, Mareček, Musil — DAMA (ICLR 2024).** Localized gender bias to mid-upper FFN layers (~20–25 in LLaMA-7B) using causal analysis; debiased via orthogonal projection of FFN weights. **Closest direct precedent** for an MLP-localization claim. We differ by: (a) including attention heads in the same comparison, (b) using modern Llama-3 / Gemma-2 with available SAEs, (c) extending beyond gender, (d) explicit residual-stream geometry framing.
- **Yang et al. — Bias A-head? (TrustNLP 2025).** Per-head bias scores on BERT/GPT-2/OPT. Attention-head focused, no MLP comparison.
- **Chintam et al. (BlackboxNLP 2023).** Compared CMA, ACDC, and DiffMask+ on GPT-2 small for gender bias; useful methodological precedent showing the three methods agree on which components matter.
- **Liu et al. — "Devil in the Neurons" (ICLR 2024).** Integrated Gap Gradients (IG²) for neuron-level bias attribution on encoder LMs.
- **Islam et al. — BiasGym (arXiv 2508.08855, 2025).** Attention-head localization via single-token elicitation; tested on real and synthetic stereotypes including country-based.
- **Shan & Mueller (arXiv 2512.20796, Dec 2025).** SAE-based bias-feature ablation on Gemma-2-9B using Gemma Scope. Demonstrates that targeted feature ablation reduces profession stereotypes while preserving demographic recognition. Most modern SAE-centric bias work; we will cite and build on its Gemma Scope methodology.
- **D'Souza — "Can We Locate and Prevent Stereotypes in LLMs?" (arXiv 2604.19764, 2026).** GPT-2 small + Llama-3.2 with contrastive activations and head bias scores. Reports the **ablation paradox**: high probe accuracy in residual streams but small effect from localized ablations, suggesting stereotypes propagate as high-dimensional residual-stream directions through skip connections. This motivates our geometry framing.
- **Dubey (arXiv 2508.09019, 2025).** Layer-wise residual-stream linear probing for bias on GPT-2 large; activation steering for mitigation. Small-scale single-model preview of our direction; we differ by decomposing into MLP vs attention writes and using modern decoders.
- **Yu et al. — "Entangled in Representations" (arXiv 2508.08879, 2025).** Patchscopes-based probing of cultural knowledge across 23 regions; "cultural flattening" finding. **Most StereACuLT-relevant prior work**; complements our component-decomposition angle.

### 2.2 Residual-stream geometry methodology we will borrow

- **Elhage et al. — "A Mathematical Framework for Transformer Circuits" (Anthropic 2021).** Residual-stream-centric framing. Foundation of our analysis.
- **Belrose et al. — Tuned Lens (arXiv 2303.08112, 2023).** Used in place of logit lens for layer-wise readout (logit lens is unreliable on Llama / Gemma).
- **Wang et al. — "Interpretability in the Wild" (ICLR 2023).** Direct logit attribution (DLA) methodology.
- **Nanda (2023); Syed, Rager, Conmy (arXiv 2310.10348).** Attribution patching.
- **Kramár et al. — AtP* (arXiv 2403.00745, 2024).** Improved attribution-patching fidelity.
- **Marks & Tegmark — "Geometry of Truth" (arXiv 2310.06824).** Difference-in-means direction extraction; PCA visualization; orthogonal-projection ablation. **Methodological template we will most closely imitate.**
- **Arditi et al. — "Refusal in Language Models is Mediated by a Single Direction" (NeurIPS 2024).** Single-direction ablation framing — we test the analogous hypothesis for stereotypes.
- **Park, Choe, Veitch (ICML 2024).** Linear representation hypothesis with the causal inner product — formal grounding for "concept as direction."
- **Lieberum et al. — Gemma Scope (arXiv 2408.05147, 2024).** Open SAEs at every layer of Gemma-2-2B/9B. Enables optional Experiment 5.
- **Templeton et al. — Scaling Monosemanticity (Anthropic 2024).** Cataloged bias-related SAE features in Claude 3 Sonnet; analogue for our Gemma Scope feature search.
- **Conmy et al. — ACDC (NeurIPS 2023).** Automated circuit discovery; possible methodological cross-check.

### 2.3 Cross-cultural stereotype benchmarks

- **Jha et al. — SeeGULL (ACL 2023)** and **Bhutani et al. — SeeGULL Multilingual (ACL 2024).** Stereotype tuples for 178 countries and 20 languages.
- **Ma et al. — EspañStereo (EMNLP 2025).** Spanish-language regional stereotypes; another organizer-affiliated dataset to cite.
- **Nadeem et al. — StereoSet (ACL 2021)** and **Nangia et al. — CrowS-Pairs (EMNLP 2020),** with Blodgett et al.'s "Norwegian Salmon" (ACL 2021) caveat — we will use a filtered subset and report the filtering protocol in the appendix.
- **Parrish et al. — BBQ (Findings of ACL 2022).** Behavioral validation in QA format.

---

## 3. Datasets

We will use *existing* datasets only — no new annotation. All probes and contrasts are constructed automatically from public benchmarks.

### 3.1 Primary: contrast pairs for direction extraction and DLA

**StereoSet intrasentence subset (filtered).** ~2,100 paired completions across gender, profession, race, religion. We will apply the Blodgett et al. 2021 critique by automatically filtering items where the stereotype/anti-stereotype labels are inconsistent or where the third (unrelated) option doesn't act as a control (we filter using LLM-as-judge with GPT-4o as a sanity check, then spot-check 200 items manually). Expect to retain ~60–70%.

**CrowS-Pairs (Nangia et al. 2020).** 1,508 minimal pairs across 9 bias types. Used for cross-benchmark validation; same filtering pipeline applied.

**SeeGULL English subset (Jha et al. 2023).** ~7,750 stereotype tuples (group, attribute, country) with offensiveness annotations. Used to construct contrast pairs of the form *"People from {country} are {stereotypical attribute}"* vs *"…{anti-stereotypical attribute}."* This dataset is our primary cross-cultural anchor and aligns directly with workshop themes.

### 3.2 Behavioral validation

**BBQ (Parrish et al. 2022).** 58k QA items across 11 demographic axes, with ambiguous and disambiguated contexts. Used post-hoc to verify that components/directions our localization identifies, when ablated, change downstream QA bias scores. We will sample ~500 items per axis to keep evaluation tractable.

### 3.3 Cross-cultural extension (stretch — Experiment 5)

**SeeGULL Multilingual (Bhutani et al. 2024).** Selected subsets in Spanish and Hindi (resource availability) to test whether localization findings transfer across cultures and languages. Alternatively **EspañStereo (Ma et al. 2025)** Spanish-only — politically advantageous to cite given organizer affiliation.

### 3.4 Construction protocol (applies to all)

1. Extract paired (stereotypical, anti-stereotypical) sentences from each dataset.
2. Tokenize with the model's tokenizer; align contrast pairs at the *trait token* (the position where the stereotypical vs anti-stereotypical word differs).
3. Filter items where the contrast-pair tokens are not both single tokens (avoids confounds in DLA); retain pairs where they are. We expect ~70% retention.
4. Stratify by bias axis (gender / race / nationality / religion / profession) for per-axis analysis.
5. Reserve 20% as held-out test set for causal-validation experiments to avoid leakage.

**Final expected dataset size:** ~6,000–8,000 contrast pairs total across StereoSet + CrowS-Pairs + SeeGULL after filtering, roughly balanced across axes.

---

## 4. Models

We prioritize models with (a) open weights, (b) available SAEs and lenses, (c) tractable size for 1-week compute, and (d) coverage in prior bias-localization literature.

### 4.1 Primary model: Gemma-2-2B (base + IT)

- **Why:** Gemma Scope (Lieberum et al. 2024) provides JumpReLU SAEs at every layer for both base and instruction-tuned variants. 26 layers, 2.6B params, fits comfortably on a single 24GB GPU. Recent (2024) and unstudied for stereotype localization in this depth.
- **Comparison vs IT vs base:** Test whether instruction tuning re-routes bias to different components — a small but novel finding.

### 4.2 Cross-validation model: Llama-3.2-3B (base + Instruct)

- **Why:** Different architecture / training data than Gemma; checks whether component-localization findings are model-specific or universal. SAEs available via EleutherAI / Goodfire. Used by D'Souza 2026, enabling direct comparison.

### 4.3 Optional larger model (only if time permits): Gemma-2-9B

- Only if Days 6–7 have headroom. Useful for comparison with Shan & Mueller 2025's Gemma-2-9B SAE work.

### 4.4 Compute budget

- **Hardware assumed:** 1× A100 40GB or H100 80GB (rentable on Lambda / RunPod / Modal at ~$2/hr; ~$200 budget for the week is sufficient).
- **Estimated peak runs:**
  - Forward passes for ~7,000 contrast pairs × 2 models = ~14,000 forward passes per experiment. Trivial (<1 hour each).
  - Attribution patching across all layer-component pairs on Gemma-2-2B (26 layers × 8 heads × 2 components/layer ≈ 416 patches) for 1,000-item subset = ~3–5 hours per axis.
  - SAE feature inference (Gemma Scope) over 7,000 pairs × 26 layers ≈ ~6 hours of disk-bound inference.

---

## 5. Experimental Plan

We use **TransformerLens** (Nanda) for hooking and DLA, **nnsight** (Fiotto-Kaufman et al.) as a fallback for Gemma-2-9B if needed, **SAELens** (Bloom et al.) for Gemma Scope SAE access, and standard Hugging Face `transformers` for tokenization and dataset I/O.

### Experiment 1 — Layer-wise probing and stereotype direction extraction

**Goal:** Establish at which layers a stereotype is linearly probable in the residual stream, and extract the per-axis direction we use throughout subsequent experiments.

**Setup.** For each contrast pair *(s_stereo, s_anti)* from filtered StereoSet + CrowS-Pairs + SeeGULL English:
1. Run forward pass; cache residual-stream activations at every layer at the *trait-token-minus-1* position (the position immediately preceding where the trait word would be predicted).
2. Compute the **difference-in-means direction** *d_ℓ = mean(h_ℓ^stereo) − mean(h_ℓ^anti)* per layer ℓ and per axis.
3. Train a logistic-regression probe per (layer, axis) on residual streams (80/20 split). Report AUC.
4. Validate that *d_ℓ* generalizes by computing cosine similarity between *d_ℓ* extracted from StereoSet vs CrowS-Pairs vs SeeGULL on the same axis.

**Outputs.**
- Per-layer probe AUC curve (one per axis), expected to rise sharply through middle layers and plateau (~layers 12–18 of 26 for Gemma-2-2B) — a Marks-Tegmark-style figure.
- Per-axis direction vectors *d_ℓ* (saved to disk for use in Experiments 2–4).
- Cross-dataset cosine similarity matrix — sanity check for direction stability.
- PCA scatter plot of stereo vs anti-stereo activations at the best-probing layer (visual hook for the paper).

**Time:** Day 1 evening + Day 2 morning. Outputs: 4 figures.

### Experiment 2 — Component-wise direct logit attribution (DLA)

**Goal:** The novel core. Quantify per-component, per-layer how much each attention head and each MLP block writes to the stereotype direction.

**Setup.**
1. For each contrast pair, run a forward pass and decompose the final-layer residual stream into per-component contributions: **h_final = embed + Σ_ℓ Σ_h attn_{ℓ,h} + Σ_ℓ mlp_ℓ**.
2. **DLA score** for component *c* on pair *p*: project *c*'s residual write through the unembedding matrix and compute the logit difference between the stereotypical and anti-stereotypical trait tokens. Average over pairs per axis.
3. **Geometric write score** for component *c*: cosine similarity between *c*'s residual write and the layer-best stereotype direction *d_ℓ\** from Experiment 1.

**Outputs.**
- Two heatmaps per axis: (i) layer × attention head DLA contribution; (ii) layer × component (attn vs MLP) DLA contribution.
- Direct quantitative answer to H1: "MLPs at layer ℓ contribute X% of the total stereotype-axis logit difference; attention heads contribute Y%; layer-by-layer breakdown in Figure N."
- Top-K components per axis flagged for Experiment 3.

**Time:** Day 2 afternoon + Day 3. Outputs: 4–6 heatmaps, 1 summary table.

### Experiment 3 — Attribution patching for causal validation

**Goal:** Verify that the components flagged by DLA are *causally* responsible, not just correlated with the output.

**Setup.**
1. For each contrast pair, take *s_stereo* as "clean" and *s_anti* as "corrupt" (or vice versa).
2. Use **AtP*** (Kramár et al. 2024) to compute attribution scores for every (layer, head) and (layer, MLP) at the trait position in a single backward pass. This costs roughly 2× a forward pass per pair — well within budget.
3. Validate top-20 candidates per axis with **full activation patching** (more expensive but exact): replace the activation of that component with the corrupt-run activation and measure the change in stereotypical-token logit.
4. **Cross-method agreement:** Compute Spearman correlation between Experiment 2's DLA ranking and Experiment 3's AtP* ranking — replicate Chintam et al. 2023's methodological cross-check.

**Outputs.**
- AtP* attribution heatmaps per axis (analogous to Experiment 2).
- Activation-patching validation table for top-20 components.
- Spearman correlation between DLA and AtP* — methodological robustness claim.

**Time:** Day 3 evening + Day 4. Outputs: 2 heatmaps, 1 table, 1 correlation figure.

### Experiment 4 — Causal validation: direction ablation and component ablation

**Goal:** Test H3. Compare two intervention regimes — single-direction ablation (Arditi-style) and top-K component ablation (Ma 2023 / DAMA-style) — on the *same* held-out test set.

**Setup.**
1. **Direction ablation.** For each axis, ablate the stereotype direction *d_ℓ* by orthogonal projection at every layer in residual stream during forward pass, on held-out 20% of contrast pairs. Measure (i) StereoSet stereotype score, (ii) BBQ accuracy disparity by demographic group on a sampled 500-item subset, (iii) MMLU 5-shot accuracy as a capability-preservation control.
2. **Top-K component ablation.** Zero-ablate (i.e., replace with the mean across pairs) the top-K components from Experiment 3. Repeat the same evaluations.
3. **Combined.** Both interventions simultaneously. Test whether they are additive (suggests independent mechanisms) or redundant.

**Expected outcome and interpretation.**
- If direction ablation reduces stereotype score substantially → analog of Arditi finding for stereotypes. Big result.
- If only component ablation works → confirms DAMA / Ma 2023 view; scope of contribution is in *which* components.
- If neither works alone but combined does → ablation paradox confirmed; D'Souza 2026's hypothesis vindicated.

Any of these is publishable; reconciling against prior literature is the contribution.

**Time:** Day 5. Outputs: 1 results table, 1 capability-degradation comparison figure.

### Experiment 5 — Cross-cultural component shift (stretch)

**Goal:** Test H4. Apply the Experiment 2 pipeline separately to (a) US/English stereotypes, (b) LatAm/Spanish stereotypes from EspañStereo or SeeGULL Multilingual Spanish subset, (c) South Asian stereotypes from SeeGULL Multilingual Hindi subset.

**Setup.** Construct three contrast-pair subsets, ~500 pairs each. Re-extract per-culture stereotype directions; re-run DLA component decomposition; compute pairwise cosine similarity of culture-specific directions and pairwise overlap of top-20 implicated components.

**Outputs.**
- Cross-culture direction-similarity matrix.
- Component-overlap Venn / Jaccard heatmap.
- Qualitative analysis: which components are culture-shared, which are culture-specific.

**Time:** Day 6 (only if Days 1–5 stayed on schedule). Outputs: 2 figures.

### Experiment 6 — SAE feature corroboration (deep stretch — only if abundant time)

**Goal:** Cross-validate findings in feature space using Gemma Scope SAEs.

**Setup.** For top-flagged (layer, component) pairs from Experiment 2/3, decode their residual writes through the corresponding Gemma Scope SAE and identify which features they activate. Search Neuronpedia for stereotype-relevant features and verify they appear among the top-activating features at the implicated layers.

**Time:** Day 6–7 only if everything else completed early. Drop without remorse if not.

---

## 6. Day-by-Day Timeline

| Day | Morning | Afternoon / Evening |
|-----|---------|---------------------|
| **1** | Environment setup: TransformerLens, SAELens, datasets HF, GPU rental. Load Gemma-2-2B-base/IT and Llama-3.2-3B. Smoke-test forward passes, hook caching, single-pair DLA. | Dataset construction: load StereoSet + CrowS-Pairs + SeeGULL English; apply filtering pipeline (LLM-as-judge); save processed contrast pairs to parquet. Manual spot-check 200 items. |
| **2** | **Experiment 1**: extract per-layer per-axis stereotype directions; train layer-wise probes on Gemma-2-2B. Generate AUC curves and PCA plots. | Begin **Experiment 2**: implement DLA hooks; compute attention-head and MLP DLA scores on a small (500-pair) batch as a sanity check. |
| **3** | Complete **Experiment 2** at full scale on Gemma-2-2B (base and IT). Generate component heatmaps. Identify top-20 components per axis. | Begin **Experiment 3**: implement AtP* hooks; run on 1,000-pair sample. |
| **4** | Complete **Experiment 3**: full-scale AtP*; validate top-20 with exact activation patching. Compute DLA-vs-AtP* Spearman correlation. | Replicate Experiments 1–3 at reduced scale on **Llama-3.2-3B** for cross-model validation. |
| **5** | **Experiment 4**: implement orthogonal-projection direction ablation; run on held-out test set; collect StereoSet + BBQ + MMLU metrics for direction-ablation, component-ablation, and combined conditions. | Begin **Experiment 5** (stretch): construct cross-cultural contrast pairs; extract per-culture directions; run DLA on Spanish subset. |
| **6** | Complete **Experiment 5**: cross-culture component overlap analysis; generate Jaccard / Venn figures. **Experiment 6** (deep stretch): SAE feature lookup if time. | Start writing: Introduction, Related Work, Methods sections. Generate all final figures. |
| **7** | Finish writing: Results, Discussion, Limitations. Internal review of prose and figures. | Format check against ACL template; submit to OpenReview. |

**Critical scope-control rules.**
- If Day 3 ends without Experiment 2 fully complete on Gemma-2-2B, drop the Llama-3.2-3B replication on Day 4 and use that day to finish Experiment 2.
- If Day 5 ends without Experiment 4 complete, drop Experiment 5 and write the paper as a Gemma-2-only short paper.
- Experiment 6 is acceptable to skip entirely.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Difference-in-means directions don't generalize across StereoSet / CrowS-Pairs / SeeGULL. | Medium | Report cross-dataset cosine similarity; if low, restrict to within-dataset analysis and note this as a finding. |
| Ablation paradox: neither direction nor component ablation reduces stereotype score. | Medium-high | This *is* a finding. Frame as confirmation of D'Souza 2026 and discuss implications for debiasing literature. |
| StereoSet measurement issues weaken claims (Blodgett 2021). | High | Filter aggressively; lead with SeeGULL for the cross-cultural framing; report results with and without each dataset. |
| Attribution patching is noisy (RelP 2025 documents this). | Medium | Use AtP* (Kramár 2024) for improved fidelity; cross-check top components with full activation patching. |
| Gemma Scope SAE inference is slow / disk-bound. | Medium | Treat Experiment 6 as fully optional; pre-cache one layer of SAE features only if attempted. |
| Component-ablation results contradict prior literature (e.g., DAMA's MLP claim doesn't hold on Gemma-2). | Low-medium | Frame as a model-specific finding; discuss training-data and architectural differences as candidates. |
| Cross-cultural data is too small for statistical claims (Experiment 5). | Medium | Use bootstrap CIs; if 500 pairs per culture is too few for stable directions, drop Experiment 5. |
| Compute overrun. | Medium | Stick to Gemma-2-2B as primary; Gemma-2-9B and Llama-3.2-3B are nice-to-haves only. Modal / RunPod can be spun up on demand. |
| Hydra effect / self-repair (McGrath et al.) makes ablation effects misleading. | Low | Note the limitation; report ablation magnitude as effect-on-output, not as proof of causal sufficiency. |

---

## 8. Deliverables and Expected Contribution

**Concrete deliverables by end of Day 7:**
1. A 4-page short paper (or 8-page long paper if Experiments 5–6 succeed) ready for OpenReview submission, with:
   - 2–4 main figures (probe-AUC curves, component heatmaps, ablation comparison bars).
   - 1–2 main tables (ablation results with capability metrics, DLA-vs-AtP* correlation).
   - Appendix with filtering protocols, full per-axis heatmaps, cross-model replication.
2. A public GitHub repository with reproducible code (TransformerLens hooks, dataset processing, all experiment notebooks).
3. Cached intermediate artifacts (extracted directions, attribution scores) for reproducibility.

**Expected scientific contribution.**
- The first clean *attention-vs-MLP residual-stream write decomposition* of stereotypes on a modern decoder-only LLM, reconciling the partially-conflicting prior claims of DAMA (FFN-mediated) vs Ma 2023 / Yang 2025 / BiasGym 2025 (attention-mediated).
- A geometric framing — "stereotype direction" — bridging the bias-localization literature with the residual-stream-direction interpretability paradigm (Marks-Tegmark, Arditi, Park-Veitch).
- An empirical test of single-direction stereotype mediation analogous to Arditi's refusal result.
- (Stretch) A first cross-cultural component-localization comparison, directly addressing StereACuLT's central theme.

**Conservative success criterion** (Day 7 paper ready): Experiments 1–4 on Gemma-2-2B with a Llama-3.2-3B robustness check.

**Ambitious success criterion** (long paper): Experiments 1–5 on both models, with SAE corroboration.

---

## 9. Practical Setup Checklist (Day 1)

```bash
# Environment
pip install transformer_lens sae_lens transformers datasets accelerate
pip install nnsight  # fallback for larger models
pip install scikit-learn scipy numpy pandas matplotlib seaborn

# Models (HF Hub)
google/gemma-2-2b
google/gemma-2-2b-it
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-3B-Instruct

# SAEs (Gemma Scope via SAELens)
gemma-scope-2b-pt-res        # base, residual-stream SAEs, all layers
gemma-scope-2b-pt-mlp        # base, MLP-out SAEs
gemma-scope-2b-pt-att        # base, attention-out SAEs
gemma-scope-2b-it-res        # instruction-tuned

# Datasets (HF Hub)
McGill-NLP/stereoset
crows_pairs                  # via crows_pairs_v1
akhilayerukola/SeeGULL
heegyu/bbq
```

A single A100 40GB or H100 80GB on RunPod / Lambda / Modal at ~$2–3/hr, run on-demand for Days 1–6, totals roughly $150–250 in compute. Code can be developed locally on CPU and only forwarded to GPU for actual runs.

---

## 10. Closing Note

The plan is deliberately scoped around Gemma-2-2B as the single primary model and four core experiments (probing, DLA, attribution patching, ablation). Llama-3.2-3B replication, cross-cultural extension, and SAE corroboration are layered as stretch goals that strengthen the paper if time allows but are non-load-bearing for the core contribution. The most defensible novelty claim — *"a clean component-wise residual-stream write decomposition of stereotypes on a modern decoder LLM, reconciling DAMA's MLP claim with the attention-head literature"* — is achievable with Experiments 1–4 alone, even on a single model.

If at any point the plan slips, default to writing a tighter short paper rather than scrambling to add experiments. The workshop is a first edition; reviewers will reward a focused, well-executed contribution over an ambitious-but-ragged one.
