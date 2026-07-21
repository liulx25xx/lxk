# External Artifact Manifest

The submission-ready anonymous artifact is staged at `artifact/blackboxnlp2026/` and archived at `output/artifact/blackboxnlp2026_anonymous_artifact.zip`.

The full working repository contains historical experiment notes, orchestration scripts, and machine-specific provenance that are intentionally excluded from external release. Those internal files may use obsolete terminology and should not be submitted.

The staged artifact contains:

- paper source and the publication figure;
- data-conversion, SFT, DPO, GRPO, evaluation, and aggregation scripts;
- raw per-example judge predictions used by the aggregation script;
- regenerated CSV/Markdown summaries and bootstrap uncertainty checks;
- no model checkpoints, licensed source examples, author identity, private hosts, or private filesystem paths.

Before release, verify the archive with:

```bash
unzip -l output/artifact/blackboxnlp2026_anonymous_artifact.zip
rg -n '(/Users/|/data_train/|10\\.[0-9]+\\.[0-9]+\\.[0-9]+)' artifact/blackboxnlp2026
```
