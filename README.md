# encoder-test

This repository is a curated snapshot of the `encoder_first_solver_second_experiment_design.md` implementation and the most relevant experiment artifacts as of April 5, 2026.

## What is included

- `docs/encoder_first_solver_second_experiment_design.md`
  - Original experiment design document.
- `routing_encoder_first/`
  - Standalone implementation of the encoder-first / solver-second CVRP pipeline.
  - Includes source code, configs, and run scripts.
- `artifacts/smoke/`
  - Early smoke-run summaries, tables, and logs.
- `artifacts/fixedval_pilot/`
  - Main fixed-validation training run snapshot.
  - Includes `nohup.log`, `run.log`, and `raw/validation_history.csv`.
- `artifacts/fixedval_listwise_only/`
  - `listwise-only` ablation snapshot with the same log/CSV layout.
- `reports/2026-04-05_fixedval_status.md`
  - Structured analysis of why the current fixed-validation runs are underperforming.

## Environment

The local experiments were run in the `easynco` conda environment on `2 x RTX 3090`.

Minimal Python packages used by `routing_encoder_first`:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `scipy`

Local helper script to install the git tooling plus these Python dependencies:

```bash
bash /public/home/chenrs/utils/install_git.sh
```

## Current status

- The implementation is additive and was developed without modifying the existing `POMO/` and `PolyNet/` codebases.
- The smoke run was successful and showed the full training/evaluation pipeline is functional.
- The larger fixed-validation runs are currently underperforming.
- At snapshot time, both fixed-validation runs had reached step `300`.
- The main issue is that `best-of-M` is not improving over greedy on validation, even though candidate diversity remains high.
- The main pilot and the `listwise-only` ablation behave very similarly, which suggests the core training signal is the bottleneck.

## Notes

- Large checkpoint files are intentionally omitted from this repository snapshot to keep it lightweight.
- The uploaded experiment artifacts are curated snapshots rather than the full local `outputs/` directory.
- Plot images are intentionally omitted from the GitHub snapshot; the most important logs, CSVs, and markdown summaries are included.
- The local training jobs may continue beyond this snapshot; the CSV and log files here should be treated as a GitHub-readable checkpoint of progress, not the final state.
