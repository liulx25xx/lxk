# Release Notes

- Anonymous review artifact; no author names or affiliations are included.
- The paper's “confidence proxy” terminology replaces older “calibration reward” shorthand.
- The historical `acc_consist` and `acc_calib` CLI values remain accepted for run compatibility and are documented as proxy modes.
- Aggregate `metrics.json` files are included for integrity checks after machine-specific checkpoint paths are replaced by `<redacted-machine-path>`; summaries are recomputed directly from `eval_results.json`.
- The paper package includes reproducible source for the mechanism figure and a shared style file that fixes color semantics across all figures.
- Licensed source datasets and model checkpoints are excluded.
