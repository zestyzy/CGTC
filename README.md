# Rotate: Supervision–Geometry–Physics Co-Learning

Rotate implements a two-stage TPAC training pipeline that warms a pure supervised model before balancing it with spatial and physics-aware regularisers. The repository now exposes hierarchical ratio guards, feedback-controlled cooling, robustness perturbations, comprehensive diagnostics, and automated ablations so that the workflow can support IJCAI-style method writing.

## Quick start

```bash
# Standard training
python train_tpac.py --cfg configs/default.yaml \
  --root /path/to/project --case C1 --tag demo --device cuda:0 \
  --pts 16384 --batch 2 --workers 4

# Robust/OOD training example
python train_tpac.py --cfg configs/default.yaml \
  --root /path/to/project --case C1 --tag robust_demo \
  --subsample_ratio 0.5 --coord_noise_sigma 0.002 \
  --occlusion_ratio 0.2 --ood_rotation --ood_rot_deg 60

# Ablations + diagnostics + physics eval
bash wo.sh --case C1 --tag ablate_demo --device cuda:0 \
  --no-guard --reverse-hierarchy --ratio-sweep \
  --no-rollback --no-freeze-teacher

# Batch physics evaluation
python eval/eval_physics.py --root /path/to/project --case C1 \
  --cfg configs/default.yaml --split test --pts 16384 \
  --ckpt-dir results/temp_results/C1/demo/weight \
  --out results/temp_results/C1/demo/physics_eval.csv

# Visual diagnostics (curves.pdf/curves.png + legacy PNGs)
python draw_picture.py --metrics results/temp_results/C1/demo/train_val_metrics.csv \
  --out-dir results/temp_results/C1/demo/diagnostics
```

A ready-to-run orchestration script is provided at `scripts/demo.sh`; it chains training, diagnostics, physics evaluation, and ablations using the commands above (adjust `ROOT`, `CASE`, and device flags as needed).

## Method perspective

### Layered soft projection (hierarchical guards)
Spatial and physics regularisers are now formalised as soft projections with explicit caps: spatial loss is limited by `rho_spat * L_sup`, while physics loss is bounded by `rho_phys * L_spat_cap`. Optional `max_ratio_vs_spatial` enforces that physics stays below a fraction of the capped spatial term. The trainer records `L_spat_raw/cap`, `L_phys_raw/cap`, guard scales, and trigger flags at step granularity, producing a schema-compliant `train_val_metrics.csv` that documents how each hierarchy level behaves.

### Feedback-controlled cooling
Whenever a guard saturates or gradient cosines become negative, the trainer triggers cooling: the corresponding blend is multiplied by `gamma`, a cooling window `W` is scheduled, and recovery is prevented until the window expires. Cooling events are exported via metadata for downstream visualisation. The defaults still realise the “Supervision > Spatial > Physics” ordering, while `guard.reverse_hierarchy`, `guard.disable`, `guard.no_rollback`, and `guard.no_freeze_teacher` expose counterfactual behaviours for ablations.

### Two-stage feasible initialisation
Training retains the warm → rollback → warmup pipeline. The warm phase trains pure supervision with EMA teacher updates; once patience fires, the trainer optionally rolls back to the warm-best state, freezes the teacher reference, and enters warmup with blend ramps. Rolling back or freezing can be disabled via guard toggles to study their impact. Stage transitions and rollback markers are embedded into the metrics metadata, allowing `draw_picture.py` to render coloured bands and vertical markers.

### Robustness and diagnostics
Deterministic perturbations (subsampling, noise, occlusion, scaling, OOD rotations) can be applied per split through CLI flags or YAML defaults; all settings are captured inside the CSV metadata header. `draw_picture.py` consumes the enriched schema to render multi-panel diagnostics (loss hierarchy, guard scales, cooling windows, gradient cosines) and exports backwards-compatible PNGs. `eval/eval_physics.py` now supports batch checkpoint evaluation, outputs quantiles and KS distances, and can store histograms per checkpoint.

### Reproducibility assets
- `train_val_metrics.csv`: step/epoch logs with guard/cooling diagnostics, gradient cosines, and metadata headers.
- `curves.pdf` / `curves.png`: staged visualisations with guard triggers and cooling spans.
- `physics_eval.csv`: per-checkpoint divergence/momentum statistics with P50/P90/P95, means, stds, and KS metrics.
- `runs/wo/<CASE>/<TAG>/wo_summary.csv`: consolidated ablation summaries (including toggles, supervision metrics, physics quantiles, guard hit rates, average cooling lengths, epoch timing, and seed).
- `scripts/demo.sh`: end-to-end demo covering training, evaluation, plotting, and ablations.

## Repository layout

- `train_tpac.py` – entry point with deterministic seeding, perturbation plumbing, and DataLoader construction.
- `training/trainer.py` – two-stage trainer with hierarchical guards, gradient cosines, cooling feedback, and CSV writer.
- `draw_picture.py` – stage-aware diagnostics with guard and cooling overlays plus gradient-cos plots.
- `eval/eval_physics.py` – divergence/momentum evaluation with summary CSVs and optional histograms.
- `wo.py` / `wo.sh` – configurable ablation runner that derives configs, launches training/diagnostics/eval, and aggregates results.
- `scripts/demo.sh` – reference script showcasing the full workflow.

Refer to the inline docstrings for implementation details and additional options.
