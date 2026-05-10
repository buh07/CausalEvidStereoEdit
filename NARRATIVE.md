# Easy to Add, Hard to Erase
## Narrative Blueprint (Revision Pass)

## Title
Easy to Add, Hard to Erase: Causal Evidence on Stereotype Editing in LLMs

## Condensed Abstract (Story-First)
As language models scaled to web-scale data, exhaustive curation stopped being realistic, and stereotype learning became a structural side effect of scale (\citep{nadeem2021stereoset,nangia2020crowspairs,ma2023deciphering,blodgett2021salmon}). The field's practical response has been post-hoc local editing: train first, then edit internal activations to reduce harmful behavior (\citep{vig2020causal,meng2022rome,zou2023repeng,turner2023steering,tan2024steering}).

We test that strategy causally across seven models. Under matched operators, injection-minus-removal asymmetry is significant in 5/7 models on behavioral score, and margin asymmetry is significant in all 7/7. Follow-up diagnostics show why removal is brittle: attribution signs are model-dependent and top-ranked intervention sets can contain suppressor components. Transfer tests across benchmark constructions show no reliable cross-source carryover in this regime.

Conclusion: localization can provide control, but not a debiasing guarantee. Credible debiasing claims must pass causal intervention, sign-reliability, and transfer validation checks.

---

## One-Sentence Story
Local stereotype signals are easier to write than erase, so localization alone is not enough to claim debiasing.

---

## Narrative Spine

### Act 1: Why this matters now
At web scale, hand-curating away every harmful pattern is infeasible. Stereotypes are therefore a predictable byproduct of modern training pipelines, not rare edge cases.

Suggested citations:
- \citep{nadeem2021stereoset,nangia2020crowspairs,zhao2018winobias,ma2023deciphering}
- \citep{blodgett2021salmon}

### Act 2: What the field currently assumes
The dominant local-editing intuition is: once stereotype signal is localized, it can be removed behaviorally by editing the same representation family.

Suggested citations:
- \citep{vig2020causal,chintam2023identifying,chandna2025dissecting,dsouza2026locate}
- \citep{zou2023repeng,turner2023steering,arditi2024refusal,braun2025steering,tan2024steering}

### Act 3: What this paper tests
We audit that assumption with causal interventions:
1. Is stereotype preference easier to increase than decrease under matched operators?
2. If decrease is weak, where does failure enter the pipeline?
3. Do benchmark-local intervention targets transfer across benchmark constructions?

Suggested citations:
- \citep{nanda2022tl,heimersheim2024patching}
- \citep{nanda2022atp,kramar2024atp,syed2024atp,conmy2023acdc}

### Act 4: What we learn
The assumption fails at scale in a specific way: writeability is broad, eraseability is selective. Mechanistic diagnostics explain brittleness, and cross-source transfer remains a practical bottleneck.

Suggested citations:
- \citep{mcgrath2023hydra,rushing2024selfrepair}
- \citep{park2023linear,zou2023repeng}

---

## Core Claims to Carry Through Abstract, Introduction, and Conclusion

1. Majority pattern: under matched operators, score-level inject-minus-remove asymmetry is significant in 5/7 tested models.
2. Confidence asymmetry is broader than behavior asymmetry: margin-level asymmetry is significant in 7/7 models.
3. Mechanistic brittleness is model-dependent: sign reliability and suppressor contamination vary across families.
4. Benchmark-local success does not automatically imply cross-source transfer.
5. Practical implication: debiasing claims need a causal validity standard, not localization alone.

---

## Current Empirical Record (Latest Sweep)

## A) Headline asymmetry (Exp16; matched prediction-position operators)
Primary contrast: `inject_on_anti - remove_on_stereo` on score.

- gemma2b: +0.100, q=0.057 (ns)
- gemma2b-it: +0.1375, q=0.0034
- llama-3.2-3b: +0.2000, q=0.0113
- qwen2.5-3b: +0.1375, q=0.0034
- qwen2.5-3b-instruct: +0.1625, q=0.0044
- mistral-7b-v0.1: +0.0909, q=0.388 (ns)
- olmo-2-7b: +0.2273, q=0.0213

Headline result: score-level asymmetry significant in 5/7 models.

Margin contrast:
- significant in all 7/7 models (q<0.05), supporting a broad confidence-level asymmetry.

Writing implication:
- Phrase the core result as "majority score asymmetry with universal margin asymmetry."

## B) Injection specificity (Exp18)
Contrasts:
- `true_minus_random` (norm-matched random control)
- `true_minus_shuffled` (axis-shuffled control)

Model-level pattern:
- True > random significant in 5/7 models.
- True > shuffled significant in 2/7 models (llama, qwen2.5-3b-instruct), with positive but non-significant estimates in several others.

Writing implication:
- Injection is robustly writable; axis purity is partial and model-dependent.

## C) Attribution sign reliability (Exp14)
DLA sign agreement:
- gemma2b: 23.8% (15/63), q=3.761e-05 (anti-aligned regime)
- gemma2b-it: 60.0% (12/20), q=0.503 (uncertain)
- llama-3.2-3b: 76.1% (35/46), q=5.356e-04 (aligned regime)
- qwen2.5-3b: 51.3% (20/39), q=1.0 (near-random regime)
- qwen2.5-3b-instruct: 52.5% (42/80), q=0.738 (near-random regime)
- mistral-7b-v0.1: 75.0% (15/20), q=0.0414 (aligned regime)
- olmo-2-7b: 50.0% (21/42), q=1.0 (near-random regime)

Writing implication:
- Present sign reliability as three observed regimes (anti-aligned, aligned, near-random), not one universal pattern.

## D) Suppressor contamination (Exp17)
Causal suppressor fractions in top-k sets:
- gemma2b: 26.25%
- gemma2b-it: 1.25%
- llama-3.2-3b: 6.25%
- qwen2.5-3b: 13.75%
- qwen2.5-3b-instruct: 20.00%
- mistral-7b-v0.1: 3.75%
- olmo-2-7b: 13.75%

DLA-sign suppressor fractions:
- 17.5% to 36.25%.

Writing implication:
- Suppressor contamination is common enough to matter, but severity is strongly model-dependent.

## E) Transfer (Exp15)
Cross-source rows:
- none are FDR-significant across all seven models.

Representative cross-source deltas (score):
- gemma2b: 0.000 (q=1.0), +0.0769 (q=1.0)
- gemma2b-it: +0.1053 (q=0.625), +0.3846 (q=0.25)
- llama-3.2-3b: 0.000 (q=1.0), +0.0909 (q=1.0)
- qwen2.5-3b: +0.1923 (q=0.453), 0.000 (q=1.0)
- qwen2.5-3b-instruct: -0.0588 (q=1.0), -0.1667 (q=1.0)
- mistral-7b-v0.1: -0.2500 (q=0.667), 0.000 (q=1.0)
- olmo-2-7b: -0.2000 (q=1.0), +0.4000 (q=1.0)

Within-source anchor:
- strongest trend remains gemma2b crows->crows: -0.1429, q=0.0768.

Writing implication:
- Use "no reliable cross-source carryover detected in this setting."
- Keep power detail in Results/Discussion tables and text, not headline messaging.

## F) Operational status of Exp20-Exp25 (project note, not headline evidence)
- Exp20: 2 completed, 1 failed
- Exp21: 2 completed, 1 failed
- Exp22: 2 completed, 1 failed
- Exp23: 3 completed, 0 failed
- Exp24: 2 completed, 1 failed
- Exp25: 2 completed, 1 failed

Writing implication:
- Keep these in future-work / implementation status, not in headline claims.

---

## Section-by-Section Story Plan for Rewriting

## 1) Introduction
### Story objective
Move from societal motivation -> methodological gap -> clear causal question.

### What to say
- Web-scale training made perfect curation unrealistic.
- Post-hoc local editing became practical default.
- This paper tests whether localized signals are behaviorally erasable.
- Headline: majority (5/7) matched-operator asymmetry.

### What to avoid
- Long caveat taxonomy in Intro.
- Detailed seed/power decomposition in Intro.

### Suggested citations
\citep{nadeem2021stereoset,nangia2020crowspairs,ma2023deciphering,blodgett2021salmon,vig2020causal,zou2023repeng,turner2023steering}

## 2) Methods
### Story objective
Clarify causal design and what is being compared.

### Must be explicit
- Cross-position design (direction extracted at trait positions, intervened at prediction position).
- Canonical asymmetry operator is Exp16 matched-position contrast.
- Statistical authority rule: BH-FDR q-values for inferential claims.

### Suggested citations
\citep{heimersheim2024patching,nanda2022tl,nanda2022atp,kramar2024atp}

## 3) Results I: Write vs Erase
### Story objective
Show the central asymmetry first, then sharpen interpretation.

### Flow
1. Exp16 score asymmetry (5/7).
2. Exp16 margin asymmetry (7/7) to show confidence-vs-behavior distinction.
3. Exp18 specificity controls to distinguish writeability from generic perturbation.

## 4) Results II: Why Erase Is Brittle
### Story objective
Replace vague mechanism talk with concrete failure channels.

### Flow
1. Exp14 sign reliability regimes.
2. Exp17 suppressor contamination rates.
3. Connect both to intervention-set quality and non-monotonic erase outcomes.

## 5) Results III: Portability / Transfer
### Story objective
Answer "does it carry over?" directly.

### Flow
1. Present 2x2 within/cross-source table.
2. State no cross-source FDR significance.
3. Anchor with strongest within-source trend and explain practical interpretation.

## 6) Discussion and Synthesis
### Story objective
Translate findings into a reusable validity standard.

### Synthesis structure
1. What is now established: majority write>erase asymmetry.
2. What explains brittleness in this dataset family: sign regime + suppressors.
3. What remains open: circuit-level mechanism identification.
4. What practitioners should require before claiming debiasing.

## 7) Conclusion
### Story objective
End on clear operational guidance, not hedge lists.

### Closing line to target
"Local edits are a controllability tool, not a debiasing guarantee; claims of debiasing should be accepted only after causal, reliability, and transfer checks pass."

---

## Practical Debiasing Validity Standard (for paper narrative)
Use this as the paper's operational contribution:

1. **Causal efficacy check:** direct inject-minus-remove contrast under matched operators.
2. **Reliability check:** attribution-sign audit before sign-driven editing.
3. **Contamination check:** suppressor-rate audit in candidate intervention sets.
4. **Transfer check:** cross-source validation before deployment-facing generalization claims.

This keeps the takeaway concrete without overloading top-level sections with caveats.

---

## Claim Language Guide

Use:
- "significant in 5/7 tested models"
- "majority asymmetry pattern"
- "model-dependent reliability regimes"
- "no reliable cross-source carryover detected in this setting"
- "localization is not, by itself, a debiasing guarantee"

Avoid:
- "universal asymmetry"
- "transfer is absent"
- "single mechanism proven"

---

## Citation Bank by Claim Type

### Bias benchmarks and evaluation cautions
\citep{nadeem2021stereoset,nangia2020crowspairs,zhao2018winobias,ma2023deciphering,blodgett2021salmon}

### Localization and editing approaches
\citep{vig2020causal,chintam2023identifying,chandna2025dissecting,dsouza2026locate,meng2022rome,zou2023repeng,turner2023steering,arditi2024refusal,tan2024steering,braun2025steering}

### Attribution and circuit tooling
\citep{elhage2021mathematical,olsson2022induction,nanda2022atp,kramar2024atp,syed2024atp,conmy2023acdc,heimersheim2024patching,nanda2022tl}

### Self-repair / contamination framing
\citep{mcgrath2023hydra,rushing2024selfrepair}

### Linear response framing
\citep{park2023linear,zou2023repeng}

### Scope contrast with global approaches
\citep{belrose2023leace,yang2024finetuning}

---

## Final Narrative Acceptance Check
If a reviewer reads only abstract, introduction, and conclusion, they should retain:
1. Why this problem exists at scale.
2. What assumption was audited causally.
3. The core empirical outcome: majority write>erase asymmetry (5/7).
4. Why removal is brittle in practice (reliability + suppressor risks).
5. The practical contribution: a concrete debiasing validity standard.
