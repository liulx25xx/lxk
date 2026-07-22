# AAAI 2027 Build Notes

This directory uses one canonical AAAI 2027 manuscript pair and one canonical supplementary pair. The earlier parallel `*_aaai2027` manuscript copies were removed after their formatting was merged into the latest sources.

## Canonical Files

- `main.tex` / `main.pdf`: latest paper content in official AAAI 2027 style, using `aaai2027.sty` and `aaai2027.bst`.
- `supplement.tex` / `supplement.pdf`: latest standalone supplementary material in the same style.
- `reproducibility_checklist_aaai2027.tex` / `reproducibility_checklist_aaai2027.pdf`: filled AAAI reproducibility checklist.
- `figure1.tex`, `figure_style.tex`, `custom.bib`, and the AAAI style files are build dependencies, not competing manuscript versions.

## Build Commands

Run from this `paper/` directory:

```bash
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error supplement.tex
latexmk -gg -pdf -interaction=nonstopmode -halt-on-error reproducibility_checklist_aaai2027.tex
```

## Current Checks

- `main.pdf`: 9 pages total; technical content and conclusion complete on page 7, and references occupy pages 8--9.
- `supplement.pdf`: 7 pages total.
- Both canonical PDFs compile without LaTeX errors, undefined references/citations, or overfull boxes.
- Page 7 of the main paper intentionally retains space for the next experiment iteration; final page balancing should happen after those results are added.
- `reproducibility_checklist_aaai2027.pdf`: 2 pages total; compiled without LaTeX errors, undefined references, or overfull boxes.
