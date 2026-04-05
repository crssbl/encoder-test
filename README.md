# encoder-test

This repository is a snapshot of the `encoder_first_solver_second_experiment_design.md` implementation and the most relevant experiment artifacts as of 2026-04-05.

## What is included

- `docs/encoder_first_solver_second_experiment_design.md`
  - Original experiment design document.
- `routing_encoder_first/`
  - Standalone implementation of the encoder-first / solver-second CVRP pipeline.
  - Includes source code, configs, and run scripts.
- `artifacts/smoke/`
  - Early smoke-run summaries, tables, and figures.
- `artifacts/fixedval_pilot/`
  - Main fixed-validation training run snapshot.
  - Includes `nohup.log`, `run.log`, `raw/train_history.csv`, `raw/validation_history.csv`, and exported figures.
- `artifacts/fixedval_listwise_only/`
  - `listwise-only` ablation snapshot with the same artifact layout.
- `reports/2026-04-05_fixedval_status.md`
  - Short analysis of why the current training results are not yet strong.

## Current status

- The implementation is additive and was developed without modifying the existing `POMO/` and `PolyNet/` codebases.
- The smoke run was successful and showed the full training/evaluation pipeline is functional.
- The larger fixed-validation runs are currently underperforming.
- The main issue is that `best-of-M` is not improving over greedy on validation, even though candidate diversity remains high.

## Notes

- Large checkpoint files are intentionally omitted from this repository snapshot to keep it lightweight.
- The uploaded experiment artifacts are curated snapshots rather than the full local `outputs/` directory.
