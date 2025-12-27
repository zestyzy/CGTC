# Training pipeline architecture

## Entry point and dataset preparation
- `train_tpac.py` loads the YAML configuration, normalises nested `SimpleNamespace` entries, and prepares the temporary output structure for a given case/tag before training starts.【F:train_tpac.py†L48-L128】
- The script ensures per-split metadata (`data.pkl`) and sampling manifests (`out.csv`) exist, regenerating them from the CFD text dumps when necessary, then instantiates the `pointdata` dataset for train/validation splits and wraps them in `DataLoader`s with pinned memory, persistent workers, and configurable sampling density.【F:train_tpac.py†L102-L191】
- `pointdata` shuffles each CFD sample, clips to the requested point budget, performs per-sample min-max normalisation (also returning the raw min/max so the PINN can recover physical units), and emits double precision tensors shaped as `(C, N)` for the PointNet-style backbone.【F:data/dataset.py†L144-L224】

## Backbone, teacher, and configuration plumbing
- `build_backbone` is a thin factory around the PointNet++ regressor: it transposes TPAC's `(B, N, 3)` tensors into `(B, 3, N)`, keeps optional features aligned, and always exposes outputs as `{p, u, v, w}` dictionaries which later get restacked into `(B, 4, N)` volumes.【F:models/backbone.py†L12-L49】
- Both the student and (optional) teacher are built from the same configuration block; `train_tpac.py` mirrors the original hyperparameters while allowing command-line overrides for device, sampling density, and batch-level resource settings.【F:train_tpac.py†L193-L227】【F:configs/default.yaml†L3-L127】

## Loss construction and adaptive weighting
- Each batch first constructs spatial attention weights that emphasise wall or interior regions according to the mixed curriculum schedule, normalising them to maintain unit mean.【F:training/losses.py†L11-L44】【F:training/schedules.py†L28-L41】
- Channel-wise error tracking forms an EMA that rescales the supervision weights within bounded ranges, optionally grouping the three velocity components to adapt jointly while keeping pressure independent.【F:training/trainer.py†L160-L178】【F:configs/default.yaml†L21-L38】

## Warmup, teacher guidance, and Boost→Hold PINN scheduling
- The trainer begins in a warm (pure supervision) stage driven by validation patience; once the metric stops improving or the configured warmup ceiling is reached it transitions to the warmup stage, seeds the teacher from the warm best checkpoint, and continues updating it with EMA while blending in the consistency loss schedule.【F:training/trainer.py†L330-L412】【F:training/schedules.py†L43-L48】【F:configs/default.yaml†L70-L96】
- Entering the physics-informed phase triggers a staged Boost→Hold strategy: continuity weight, neighbour count, and sampling density are temporarily amplified until divergence drops below the configured tolerance, after which the trainer switches to Hold and restores the baseline hyperparameters while enabling the momentum penalty.【F:training/trainer.py†L384-L412】【F:configs/default.yaml†L40-L79】
- The continuity multiplier continues to adapt using recent raw divergence statistics until a stable plateau is detected; optional auto-calibration of the divergence target occurs mid-warmup to better match observed scales.【F:training/trainer.py†L440-L455】【F:configs/default.yaml†L61-L75】

## Physics regularisation and gradient control
- When active, the per-epoch `SteadyIncompressiblePINN` augments the batch loss with continuity/momentum residuals evaluated on mixed wall/interior weights, while optionally throttling the pressure gradient with a registered hook until the dedicated release epoch; the trainer now forwards batch-wise min/max statistics gathered from the dataset so the PINN computes residuals in physical space via `norm_mode="denorm_physical"`.【F:training/trainer.py†L136-L212】
- Teacher consistency, spatial rotation, and PINN regularisers are each wrapped in ratio guards that cap their magnitude relative to the supervised MSE; the spatial term also bounds the PINN contribution through `pinn.max_ratio_vs_spatial`, ensuring the hierarchy “supervision → spatial → physics” is preserved even when raw residuals spike.【F:training/trainer.py†L250-L452】【F:configs/default.yaml†L57-L83】
- Validation reuses the same PINN module in eval mode to log weighted and raw residual metrics alongside region-specific MAE/MSE computed via wall/interior quantile masks.【F:training/trainer.py†L241-L316】【F:training/metrics.py†L8-L28】

## Checkpointing, logging, and curve exports
- Every epoch updates a wide `pandas` history table (later rendered to CSV) capturing supervision, physics, teacher consistency, adaptive weights, and scheduling state; this drives the downstream curve plot saved at the end of training.【F:training/trainer.py†L456-L474】【F:train_tpac.py†L211-L227】【F:training/utils.py†L23-L63】
- Multiple checkpoint policies coexist: the first post-warmup epoch that satisfies the divergence envelope, the best validation MAE after that milestone, the physics-balanced combo minimum, periodic snapshots, and the warmup-best teacher copy. The final recommendation symlinks whichever artifact is available in that priority order to `final_reco.pth`.【F:training/trainer.py†L475-L553】
