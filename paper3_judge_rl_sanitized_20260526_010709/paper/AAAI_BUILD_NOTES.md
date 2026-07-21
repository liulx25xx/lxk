# AAAI 2027 Build Notes

This directory keeps the original ACL-style draft and a parallel AAAI 2027 submission build.

## Stable Draft Files

- `main.tex` / `main.pdf`: ACL-style working draft. Technical content ends on page 7 and references start on page 8.
- `supplement.tex` / `supplement.pdf`: ACL-style supplementary material.

## AAAI Submission Files

- `main_aaai2027.tex` / `main_aaai2027.pdf`: official AAAI 2027 style, using `aaai2027.sty` and `aaai2027.bst`.
- `supplement_aaai2027.tex` / `supplement_aaai2027.pdf`: official-style supplementary material.
- `reproducibility_checklist_aaai2027.tex` / `reproducibility_checklist_aaai2027.pdf`: filled AAAI reproducibility checklist.

## Build Commands

Run from this `paper/` directory:

```bash
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error main_aaai2027.tex
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error supplement_aaai2027.tex
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error reproducibility_checklist_aaai2027.tex
```

## Current Checks

- `main_aaai2027.pdf`: 7 pages total; conclusion completes on page 6 and references begin on page 6.
- `supplement_aaai2027.pdf`: 7 pages total; compiled without LaTeX errors, undefined references, or overfull boxes.
- `reproducibility_checklist_aaai2027.pdf`: 2 pages total; compiled without LaTeX errors, undefined references, or overfull boxes.

