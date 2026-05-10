# StereACuLT 2026 Submission Guide and Narrative Clarity Notes

_Last updated: May 7, 2026 (America/New_York)_

## 1) Narrative Clarity Notes (Action Items)

These are the three points to enforce in the main draft:

1. Keep the main thread tight by limiting the main paper to **3-4 headline figures/results bundles**.
- Main text should foreground: Exp04 (asymmetry), Exp09/11 (component causality + sign-aware ablation), Exp14 (sign reliability), Exp15 (cross-dataset transfer).
- Move exhaustive per-axis/per-component tables and secondary ablations to appendix.

2. State the practical takeaway early and plainly.
- Add a direct sentence in the Introduction and Abstract:
  - "Single-site debiasing is often unreliable because stereotype representations are distributed and/or redundant."

3. Declare primary outcomes up front and keep them consistent.
- State early that:
  - `stereotype_score_delta` = primary behavioral outcome
  - `mean_margin_delta` = primary mechanistic/confidence outcome
- Keep this framing consistent to avoid perceived metric switching.

---

## 2) Workshop-Specific Submission Requirements (StereACuLT CFP)

Authoritative CFP page:
- https://sites.google.com/view/stereacult-2026/call-for-papers?authuser=0

Key points from CFP:
- Venue: ACL 2026 Workshop StereACuLT (hybrid)
- Paper types: long and short
- Length:
  - Long: up to **8 pages**
  - Short: up to **4 pages**
  - References and appendices excluded from these limits
- Submission system: OpenReview
  - https://openreview.net/group?id=aclweb.org/ACL/2026/Workshop/StereACuLT
- Deadlines listed on CFP (AoE, UTC-12):
  - Submission: **May 11, 2026**
  - Notification: **June 3, 2026**
  - Camera-ready: **June 14, 2026**

Notes:
- CFP page visibly shows old dates and updated dates on the same lines; use the updated ones above.
- All deadlines explicitly use **11:59 p.m. UTC-12 (Anywhere on Earth)**.
- The OpenReview invitation metadata exposes a technical `duedate` timestamp that may not mirror the CFP wording; treat the CFP and official workshop announcements as authoritative and verify final cutoff directly on the submission form near submission time.

---

## 3) Official ACL Style and Formatting Documents

Use these as the baseline style authority (unless workshop chairs publish stricter overrides):

1. ACL paper formatting guidelines (official)
- https://acl-org.github.io/ACLPUB/formatting.html

2. ACL review-version checklist (anonymization, page limits, etc.)
- https://acl-org.github.io/ACLPUB/review-version.html

3. ACL final-version checklist (de-anonymization, metadata, copyright)
- https://acl-org.github.io/ACLPUB/final-version.html

4. Official ACL style files (LaTeX + Word)
- https://github.com/acl-org/acl-style-files

5. Overleaf ACL template (from official style repo)
- https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj

6. ACL pubcheck (official format checker)
- https://github.com/acl-org/aclpubcheck

---

## 4) OpenReview Submission Form Details (StereACuLT)

OpenReview submission invitation metadata (API):
- `aclweb.org/ACL/2026/Workshop/StereACuLT/-/Submission`

Current form fields include:
- title
- submission type: archival vs non-archival
- authors / authorids
- keywords
- TL;DR (optional)
- abstract
- PDF upload
- non-archival confirmation checkbox (conditional)

Practical implication:
- Decide archival vs non-archival explicitly before submission.
- Ensure metadata text in form fields is clean and consistent with PDF title/abstract.

---

## 5) Formatting Checklist for This Project

### A. Review submission checklist

1. Use official ACL style files without manual style edits.
2. PDF is A4, two-column, line numbers/ruler enabled for review version.
3. Anonymous review version:
- no author names/affiliations in manuscript
- no acknowledgments
- no identifying self-reference phrasing
4. Respect page limits (workshop-specific):
- long <= 8 pages content
- short <= 4 pages content
5. Include required "Limitations" section (ACL guideline).
6. References and appendices formatted per ACL rules.

### B. Technical PDF checks

Run these before upload:

```bash
pdffonts paper.pdf
pdfinfo paper.pdf | rg "Page size"
```

Expected:
- All fonts embedded (`emb` = yes)
- A4 page size (`595.276 x 841.89 pts`)

Optional but strongly recommended:

```bash
uvx --from git+https://github.com/acl-org/aclpubcheck aclpubcheck --paper_type long /path/to/paper.pdf
```

If using short format, set `--paper_type short`.

### C. Camera-ready checklist (post-acceptance)

1. De-anonymize manuscript and self-citations.
2. Add authors/affiliations and acknowledgments (if applicable).
3. Keep within camera-ready page policy given by chairs.
4. Verify metadata (title, authors, abstract) exactly matches OpenReview record.
5. Re-run PDF checks and pubcheck.

---

## 6) Recommended Main-Paper Scope Lock (for 8-page fit)

Main-text bundles:
1. Geometry and asymmetry: Exp04 + compact Exp07/08 + Exp10 summary.
2. Component causality and failure modes: Exp09 + Exp11.
3. Method validity: Exp14 sign reliability.
4. Transfer/generalization: Exp15 matrix.

Appendix-only by default:
- Full per-axis/per-layer/per-component tables
- Extended cultural exploratory tables
- Secondary robustness variants that do not change conclusions
