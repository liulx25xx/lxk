# EMNLP Review Alignment and Revision Record

Updated: 2026-07-22

## Source decision

- `main_neurips.tex` is the active manuscript.
- `main.tex` and `14489_Cascade_Structure_Predic.pdf` are historical sources.
- The submitted PDF has no exact TeX snapshot in this folder.  Its useful
  motivation and diagnostic framing were selectively recovered, but audited
  inconsistencies were not copied forward.

## New paper direction

The manuscript is now organized around **selective recovery**:

> Given only the observable trajectory prefix, choose whether to redirect,
> escalate to richer execution evidence, or abstain.  Optimize incremental task
> utility and intervention harm, not retrospective category accuracy alone.

The current oracle-context results are diagnostic evidence.  They are no longer
presented as a deployable router or proof of software repair.

## Review issue matrix

| Review concern | Revision status | Where handled |
|---|---|---|
| No end-to-end issue-resolution evidence | Planned, not claimed | Selective-recovery contract and P0 server plan |
| Gold category/files and non-first-error prompt construction | Explicitly disclosed | Experimental setup and limitations |
| Weak 0--3 semantic metric | Reanalyzed | Proxy ablations, LLM-judge sensitivity, qualitative proxy failures |
| LOGIC majority lacks a recovery path | Direction changed | Evidence escalation is now a primary action family and P1 experiment |
| PLAN/LOGIC boundary and small PLAN sample | Calibrated | PLAN is explicitly a residual category with n=8 |
| LOC wrong-function inconsistency | Fixed | LOC is file-overlap only; wrong-function claim removed |
| Percentages sum to 101 | Fixed | Taxonomy table reports counts rather than rounded percentages |
| Classifier accuracy, imbalance, latency, reproducibility | Old claim removed | New plan targets prefix-only action value and risk--coverage |
| Missing qualitative examples | Partially fixed | Two artifact-grounded cases added; expansion remains P1 |
| Single agent, Python, SWE-bench Verified | Acknowledged; experiment pending | Limitations and P2 transfer plan |
| Strong ``frontier'' and ``82% waste'' claims | Removed | PTR is a positional statistic; no universal frontier claim |
| Missing usable software/data | Local artifacts organized; public release pending | Local audit/scripts and submission release checklist |

## Valuable content retained from the submitted PDF

- The opening observation that help can compound failure.
- The concrete mismatch example as motivation.
- The distinction among retry loops, localization, applied-but-unresolved edits,
  and plan-level cases.
- The idea that diagnosis should precede a narrow intervention.
- The practical importance of abstention under uncertainty.

These ideas were rewritten around the audited evidence.  The online classifier,
manual-agreement claim, semantic waste claim, fixed scaffoldability frontier,
and end-to-end implications were not restored.

## Active files

- Manuscript: `main_neurips.tex`
- Compiled PDF: `output/pdf/main_neurips.pdf`
- Local evidence audit: `LOCAL_AUDIT.md`, `LOCAL_EXPERIMENTS.md`
- Submission priorities: `AAAI27_EXPERIMENT_PRIORITIES.md`
- Server queue: `SERVER_EXPERIMENTS.md`

