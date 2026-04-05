# Routing Encoder-First

`routing_encoder_first` implements the MVP described in `encoder_first_solver_second_experiment_design.md`:

- transformer-style encoder
- dense directed edge field head
- perturb-and-project candidate generation
- cycle-cover + subtour patch giant-tour projector
- detached exact split DP
- instance-relative listwise training

The implementation is fully additive and does not modify existing `POMO/` or `PolyNet/` code.

## Quick Start

Environment preparation:

```bash
bash /public/home/chenrs/utils/install_git.sh
```

The implementation expects the `easynco` conda environment and uses:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `scipy`

Smoke run:

```bash
bash routing_encoder_first/scripts/run_smoke.sh
```

Main pilot run:

```bash
bash routing_encoder_first/scripts/run_main.sh
```

Background fixed-validation training:

```bash
bash routing_encoder_first/scripts/run_train_nohup.sh
```

Both scripts activate the `easynco` conda environment and expose `CUDA_VISIBLE_DEVICES=0,1`.

## Outputs

Each run writes to:

```text
routing_encoder_first/outputs/<experiment_name>/
├── checkpoints/
├── figs/
├── raw/
├── tables/
├── summary.json
└── summary.md
```

## Notes

- The current version focuses on the main method and internal ablations.
- External solver comparisons are intentionally left as future extensions, but the interfaces are already in place.
