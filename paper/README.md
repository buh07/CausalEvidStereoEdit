# Paper Folder (ACL / StereACuLT Submission Scaffold)

This folder centralizes the LaTeX manuscript scaffold, official ACL style assets, and workshop submission guideline references.

## Included Files

- `paper.tex`: review submission variant (`\usepackage[review]{acl}`)
- `paper_preprint.tex`: preprint variant (`\usepackage[preprint]{acl}`)
- `paper_final.tex`: camera-ready variant (`\usepackage{acl}`)
- `sections/abstract.tex`: shared abstract text (used by all variants)
- `sections/body.tex`: shared manuscript body (used by all variants)
- `acl.sty`, `acl_natbib.bst`: official ACL style files (vendored locally)
- `refs.bib`: bibliography database
- `Makefile`: build + cleanup + PDF checks
- `templates/`: copied official ACL template sources
- `guidelines/`: workshop/ACL guideline packet and source captures
- `STYLE_PROVENANCE.md`: source URLs + pinned style repo commit used for vendoring

## Build Commands

```bash
cd /jumbo/lisp/f004ndc/StereACL/paper
make review      # builds paper.pdf
make preprint    # builds paper_preprint.pdf
make final       # builds paper_final.pdf
```

## Validation Commands

```bash
make check-review
make check-preprint
make check-final
```

These validate:
- embedded fonts (`pdffonts`)
- A4 page size (`pdfinfo`)

## Optional: ACL Pubcheck

If `uvx` is available in your environment:

```bash
uvx --from git+https://github.com/acl-org/aclpubcheck aclpubcheck --paper_type long /jumbo/lisp/f004ndc/StereACL/paper/paper.pdf
```

## Submission Notes

- Keep review submission anonymized.
- StereACuLT long-paper limit: **8 pages** (excluding references/appendices).
- Include a **Limitations** section (ACL requirement).
- Keep OpenReview metadata (title/authors/abstract) consistent with manuscript metadata.

See:
- `guidelines/INDEX.md`
- `guidelines/WORKSHOP_SUBMISSION_GUIDE.md`
