# Fixed-Validation Status on April 5, 2026

This note summarizes the current status of the `CVRP100` fixed-validation experiments and the most likely reasons they are underperforming.

## Snapshot scope

- Source code: `routing_encoder_first/`
- Design document: `docs/encoder_first_solver_second_experiment_design.md`
- Smoke artifacts: `artifacts/smoke/`
- Fixed-validation pilot: `artifacts/fixedval_pilot/`
- Fixed-validation `listwise-only` ablation: `artifacts/fixedval_listwise_only/`

The artifact snapshot in this repository was refreshed when both fixed-validation runs had reached step `300`.

## Key observations

### 1. `best-of-M` is not improving over greedy on validation

On the fixed validation subset, `edge_field_greedy` and `edge_field_best_of_m` are almost identical across the logged checkpoints.

Pilot run:

- step `100`: `18.2583` vs `18.2583`
- step `200`: `18.2483` vs `18.2483`
- step `300`: `18.2494` vs `18.2484`

`listwise-only` run:

- step `100`: `18.2627` vs `18.2627`
- step `200`: `18.2644` vs `18.2644`
- step `300`: `18.2395` vs `18.2393`

The `best-of-M` gain at step `300` is therefore present but still tiny: about `0.0010` for the pilot and `0.0002` for `listwise-only`.

### 2. Main method and `listwise-only` are very close

The full method does not separate clearly from the ablation:

- Pilot best training greedy cost: `17.7847` at step `54`
- `Listwise-only` best training greedy cost: `17.7555` at step `54`
- Pilot latest training greedy cost in the snapshot: `18.1553`
- `Listwise-only` latest training greedy cost in the snapshot: `18.0303`
- Pilot latest validation greedy cost in the snapshot: `18.2494`
- `Listwise-only` latest validation greedy cost in the snapshot: `18.2395`

This suggests the current diversity and consistency terms are not the main limiting factor.

### 3. Training improves early, then stalls or regresses

Both runs improve during the early phase and then flatten or degrade.

Pilot run:

- loss drops from about `1.98` at step `10` to about `1.02` around step `140`
- later rises back to about `1.32` by step `300`

`Listwise-only` run:

- loss drops from about `1.96` at step `10` to about `1.05` around step `130-140`
- later rises back to about `1.39` by step `300`

The diversity metric also steadily falls from about `0.99` to about `0.85`, but this does not translate into a validation gain.

### 4. Candidate diversity exists, but it is not useful diversity

The candidate set remains structurally diverse, yet the best candidate is not better than greedy on validation. That points to a mismatch between:

- what the model is encouraged to score highly
- what the downstream projector and split solver actually turn into low-cost CVRP routes

## Most likely causes

### 1. Detached projector bottleneck

Training currently works as:

1. predict edge scores
2. run a detached giant-tour projector
3. run a detached exact split
4. rank the resulting candidates afterward

Because the projector is discrete and detached, the easiest thing for the model to learn is a stable greedy solution, not a candidate distribution whose noisy variants are truly better after projection.

### 2. The listwise objective is too weak for candidate generation

The current loss only tries to align candidate energy rankings with post-projection cost rankings. It does not directly reward:

- making non-greedy candidates better than greedy
- widening the quality gap between the best candidate and the rest
- preserving exploration pressure once the model finds a stable greedy mode

As a result, the system can minimize loss while collapsing toward a single useful candidate.

### 3. The energy function is very shallow

Candidate energy is currently the negative mean edge score along the chosen successor edges. This is simple and stable, but it is probably too weak to distinguish subtle route-quality differences after cycle-cover projection and split.

### 4. Diversity regularization is misaligned with route quality

The diversity term encourages edge disagreement, but edge disagreement alone does not guarantee better routes. The current runs show that candidates can stay different without producing a better `best-of-M`.

### 5. The high-throughput setup trades iteration speed for resource use

The large-batch configuration increased memory use substantially, which was useful for utilization, but it also made each logged step very slow because CPU projector work still dominates wall-clock time. That makes fast algorithmic iteration harder and likely reduced the value of keeping these long runs going unchanged.

## Practical interpretation

The current results do **not** look like a case where simply training longer will naturally cross a quality threshold. The stronger interpretation is:

- the pipeline is implemented correctly enough to run end-to-end
- the smoke experiment validated the engineering path
- the present training signal is not strong enough to turn candidate diversity into true `best-of-M` gains

## Recommended next changes

1. Stop treating longer training as the default fix.
2. Replace or augment the current listwise loss with an explicit improvement objective, for example a margin-style loss that rewards non-greedy candidates only when they beat greedy.
3. Reduce training batch size to recover faster experimental iteration.
4. Rework projector parallelism so the run does not instantiate unnecessary idle worker pools.
5. Re-run a short pilot of `300-500` steps after the objective change instead of continuing the current setup unchanged.
