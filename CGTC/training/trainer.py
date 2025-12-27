# -*- coding: utf-8 -*-
# trainer.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import copy
import json
import math
import time
from collections import deque

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:  
    from CGTC.models.pinn import SteadyIncompressiblePINN
    from CGTC.training.losses import (
        make_scaler,
        make_point_weights,
        compute_weighted_losses,
        losses_per_channel,
    )
    from CGTC.training.schedules import (
        base_lambda_cont,
        current_alpha,
        teacher_consis_weight,
    )
    from CGTC.training.utils import ensure_dir, save_state, symlink_or_copy
    from CGTC.eval.metrics import compute_region_metrics
except Exception:
    try:  
        from CGTC.models.tpac_pinn import SteadyIncompressiblePINN
        from CGTC.training.losses import (
            make_scaler,
            make_point_weights,
            compute_weighted_losses,
            losses_per_channel,
        )
        from CGTC.training.schedules import (
            base_lambda_cont,
            current_alpha,
            teacher_consis_weight,
        )
        from CGTC.training.utils import ensure_dir, save_state, symlink_or_copy
        from CGTC.eval.metrics import compute_region_metrics
    except Exception:
        from models.pinn import SteadyIncompressiblePINN
        from training.losses import (
            make_scaler,
            make_point_weights,
            compute_weighted_losses,
            losses_per_channel,
        )
        from training.schedules import (
            base_lambda_cont,
            current_alpha,
            teacher_consis_weight,
        )
        from training.utils import ensure_dir, save_state, symlink_or_copy
        try:
            from eval.metrics import compute_region_metrics
        except Exception:
            from training.metrics import compute_region_metrics


# ----------------- small helpers -----------------
def stack_preds(preds_dict: dict) -> torch.Tensor:
    # preds_dict: keys "p","u","v","w", each (B,N); return (B,4,N) in [p,u,v,w]
    p = preds_dict["p"].unsqueeze(1)
    u = preds_dict["u"].unsqueeze(1)
    v = preds_dict["v"].unsqueeze(1)
    w = preds_dict["w"].unsqueeze(1)
    return torch.cat([p, u, v, w], dim=1).contiguous()


def forward_backbone(backbone: nn.Module, x_b3n: torch.Tensor) -> torch.Tensor:
    # backbone expects (B,N,3), returns dict with "p","u","v","w" (B,N)
    x_bnc = x_b3n.transpose(1, 2).contiguous()
    preds_dict = backbone(x_bnc)
    return stack_preds(preds_dict)


class Trainer:
    # ----------------- static utilities -----------------
    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
        raise ValueError(f"Unexpected batch format: {type(batch)}")

    @staticmethod
    def _prepare_batch_stats(stats, device: torch.device):
        if stats is None:
            return None
        if not torch.is_tensor(stats):
            stats = torch.as_tensor(stats)
        stats = stats.to(device=device, dtype=torch.float32)

        # squeeze trivial trailing dim
        if stats.dim() == 3 and stats.size(-1) == 1:
            stats = stats.squeeze(-1)

        if stats.dim() == 1:
            stats = stats.view(1, -1)
        elif stats.dim() == 2:
            if stats.size(1) == 1:
                stats = stats.view(stats.size(0), -1)
            elif stats.size(0) == 14 and stats.size(1) != 14:
                stats = stats.transpose(0, 1).contiguous()
        else:
            stats = stats.view(stats.size(0), -1)

        if stats.dim() != 2 or stats.size(1) < 14:
            return None

        if stats.size(1) > 14:
            stats = stats[:, :14]

        x_min = stats[:, 0:3].contiguous()
        x_max = stats[:, 3:6].contiguous()
        y_min = stats[:, 6:10].contiguous()
        y_max = stats[:, 10:14].contiguous()
        return (x_min, x_max, y_min, y_max)

    def _resolve_warm_best_state(self) -> Optional[dict[str, torch.Tensor]]:
        if self.best_warm_state is not None:
            return self.best_warm_state
        if (self.best_warm_path is not None) and self.best_warm_path.exists():
            return torch.load(self.best_warm_path, map_location="cpu")
        return None

    def _refresh_teacher_ref(self, state_dict: Optional[dict[str, torch.Tensor]] = None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        ref = copy.deepcopy(self.model)
        ref.load_state_dict(state_dict, strict=True)
        ref.to(self.device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.teacher_ref = ref

    @staticmethod
    def _small_rotation(x: torch.Tensor, max_deg: float) -> Optional[torch.Tensor]:
        if max_deg <= 0:
            return None
        if x.dim() != 3 or x.size(1) != 3:
            return None

        b, _, _ = x.shape
        max_rad = math.radians(float(max_deg))
        if max_rad <= 0:
            return None

        angles = x.new_empty((b, 3)).uniform_(-max_rad, max_rad)
        cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
        sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])

        rx = torch.zeros((b, 3, 3), device=x.device, dtype=x.dtype)
        rx[:, 0, 0] = 1
        rx[:, 1, 1] = cx
        rx[:, 1, 2] = -sx
        rx[:, 2, 1] = sx
        rx[:, 2, 2] = cx

        ry = torch.zeros((b, 3, 3), device=x.device, dtype=x.dtype)
        ry[:, 1, 1] = 1
        ry[:, 0, 0] = cy
        ry[:, 0, 2] = sy
        ry[:, 2, 0] = -sy
        ry[:, 2, 2] = cy

        rz = torch.zeros((b, 3, 3), device=x.device, dtype=x.dtype)
        rz[:, 2, 2] = 1
        rz[:, 0, 0] = cz
        rz[:, 0, 1] = -sz
        rz[:, 1, 0] = sz
        rz[:, 1, 1] = cz

        rot = torch.bmm(rz, torch.bmm(ry, rx))
        x_rot = torch.bmm(rot, x)
        return x_rot

    # ----------------- gradient helpers -----------------
    def _select_grad_params(self, patterns: list[str]) -> list[torch.nn.Parameter]:
        named = list(self.model.named_parameters())
        if not named:
            return []
        params: list[torch.nn.Parameter] = []
        use_auto = any(pat == "auto" for pat in patterns)

        if use_auto:
            idxs = {0, len(named) - 1, max(0, len(named) // 2)}
            for idx in sorted(idxs):
                params.append(named[idx][1])

        for name, param in named:
            for pat in patterns:
                if pat == "auto":
                    continue
                if pat in name:
                    params.append(param)
                    break

        unique: list[torch.nn.Parameter] = []
        seen = set()
        for p in params:
            if p.requires_grad and id(p) not in seen:
                unique.append(p)
                seen.add(id(p))
        return unique

    @staticmethod
    def _grad_to_vec(grads: tuple[Optional[torch.Tensor], ...]) -> Optional[torch.Tensor]:
        vecs: list[torch.Tensor] = []
        for g in grads:
            if g is None:
                continue
            vecs.append(g.float().reshape(-1))
        if not vecs:
            return None
        return torch.cat(vecs, dim=0)

    def _compute_grad_cos(self, loss_a: Optional[torch.Tensor], loss_b: Optional[torch.Tensor]) -> float:
        if loss_a is None or loss_b is None:
            return float("nan")
        if (not getattr(loss_a, "requires_grad", False)) or (not getattr(loss_b, "requires_grad", False)):
            return float("nan")
        if not self.grad_params:
            return float("nan")

        grads_a = torch.autograd.grad(loss_a, self.grad_params, retain_graph=True, allow_unused=True)
        grads_b = torch.autograd.grad(loss_b, self.grad_params, retain_graph=True, allow_unused=True)
        va = self._grad_to_vec(grads_a)
        vb = self._grad_to_vec(grads_b)
        if va is None or vb is None:
            return float("nan")
        denom = va.norm() * vb.norm()
        if float(denom.item()) < self.gradcos_eps:
            return float("nan")
        cos = torch.dot(va, vb) / (denom + 1e-12)
        return float(cos.item())

    # --------- scheme2: grad-ratio guard (PINN) ---------
    def _grad_norm(self, loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
        """
        Return L2 norm of grads wrt params (float). If fails -> nan.
        """
        try:
            if loss is None or (not getattr(loss, "requires_grad", False)):
                return float("nan")
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            s = torch.tensor(0.0, device=loss.device)
            cnt = 0
            for g in grads:
                if g is None:
                    continue
                if not torch.isfinite(g).all():
                    continue
                s = s + torch.sum(g.float() * g.float())
                cnt += 1
            if cnt == 0:
                return float("nan")
            val = torch.sqrt(s + 1e-12)
            return float(val.detach().item())
        except Exception:
            return float("nan")

    @staticmethod
    def _cosine_scale(
        cos_val: float,
        low: float,
        high: float,
        min_scale: float,
    ) -> float:
        if not math.isfinite(cos_val):
            return 1.0
        if high <= low:
            return float(min_scale)
        t = (cos_val - low) / (high - low)
        t = float(max(0.0, min(1.0, t)))
        return float(min_scale + (1.0 - min_scale) * t)

    def _cosine_control(self, cos_val: float) -> tuple[str, float, Optional[str]]:
        if (not self.cos_control) or (not math.isfinite(cos_val)):
            return "off", 1.0, None

        if cos_val >= self.cos_trust:
            return "trust", 1.0, None
        if cos_val >= self.cos_surgery:
            scale = self._cosine_scale(
                cos_val,
                self.cos_surgery,
                self.cos_trust,
                self.cos_scale_min,
            )
            return "scale", scale, None
        if cos_val >= self.cos_freeze:
            scale = self._cosine_scale(
                cos_val,
                self.cos_freeze,
                self.cos_surgery,
                self.cos_scale_min,
            )
            return "surgery", scale, self.cos_surgery_solver
        return "freeze", 0.0, None

    def _apply_grad_ratio_guard(
        self,
        raw_loss: torch.Tensor,
        sup_loss: Optional[torch.Tensor],
        tau: float,
        *,
        eps: float = 1e-8,
        params: Optional[list[torch.nn.Parameter]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Scheme-2: cap by gradient norm ratio:
            r = ||g_phys|| / (||g_sup|| + eps)
            enforce r <= tau by scaling phys loss with s = min(1, tau*(||g_sup||+eps)/(||g_phys||+eps))
        """
        info = {
            "scale": 1.0,
            "triggered": False,
            "raw": float(raw_loss.detach().mean().item()),
            "cap": float(raw_loss.detach().mean().item()),
            "limit": None,  # for compatibility; here we store tau
            "ratio": float(tau or 0.0),
            "caps": [],
            "metric": "grad",
            "g_sup": float("nan"),
            "g_phys": float("nan"),
            "g_ratio": float("nan"),
        }

        if tau is None or float(tau) <= 0.0 or sup_loss is None:
            return raw_loss, info

        if params is None:
            # prefer grad_params (subset) for speed/stability
            params = self.grad_params if self.grad_params else [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return raw_loss, info

        g_sup = self._grad_norm(sup_loss, params)
        g_phys = self._grad_norm(raw_loss, params)
        info["g_sup"] = float(g_sup)
        info["g_phys"] = float(g_phys)

        if (not math.isfinite(g_sup)) or (not math.isfinite(g_phys)):
            return raw_loss, info
        if g_phys <= 0.0:
            return raw_loss, info

        ratio_now = g_phys / (g_sup + float(eps))
        info["g_ratio"] = float(ratio_now)

        scale_val = 1.0
        if ratio_now > float(tau):
            scale_val = max(min(float(tau) * (g_sup + float(eps)) / (g_phys + float(eps)), 1.0), 0.0)
            info["triggered"] = True
            info["limit"] = float(tau)

        loss_term = raw_loss * float(scale_val)
        info["scale"] = float(scale_val)
        info["cap"] = float(loss_term.detach().mean().item())
        return loss_term, info

    # ----------------- cooling / logging -----------------
    def _trigger_cooling(self, term: str, reason: str) -> str:
        event = f"{term}:{reason}"
        if self.disable_guard:
            return event

        if term == "pinn":
            self.pinn_blend = max(0.0, float(self.pinn_blend) * self.guard_gamma_pinn)
            window = max(1, self.cooling_window_pinn)
            self.guard_cooldown = max(self.guard_cooldown, window)
            state = self.cooling_state["pinn"]
            state["active"] = True
            state["remaining"] = max(int(state.get("remaining", 0)), window)
            state["reason"] = reason
            self.cooling_events.append(
                {
                    "step": self.global_step,
                    "term": "pinn",
                    "event": "start",
                    "reason": reason,
                    "window": window,
                    "gamma": self.guard_gamma_pinn,
                }
            )
        elif term == "spatial":
            self.spatial_blend = max(0.0, float(self.spatial_blend) * self.spatial_gamma)
            window = max(0, self.spatial_cooling_window)
            self.spatial_cooldown = max(self.spatial_cooldown, window)
            state = self.cooling_state["spatial"]
            state["active"] = True
            state["remaining"] = max(int(state.get("remaining", 0)), window)
            state["reason"] = reason
            self.cooling_events.append(
                {
                    "step": self.global_step,
                    "term": "spatial",
                    "event": "start",
                    "reason": reason,
                    "window": window,
                    "gamma": self.spatial_gamma,
                }
            )
        return event

    def _update_cooling_states_after_epoch(self, in_pinn: bool) -> None:
        # PINN cooling
        state_p = self.cooling_state["pinn"]
        if in_pinn:
            if self.guard_cooldown > 0:
                state_p["active"] = True
                state_p["remaining"] = self.guard_cooldown
            else:
                if state_p.get("active"):
                    self.cooling_events.append(
                        {"step": self.global_step, "term": "pinn", "event": "end"}
                    )
                state_p.update({"active": False, "remaining": 0, "reason": None})
        else:
            if state_p.get("active"):
                self.cooling_events.append(
                    {"step": self.global_step, "term": "pinn", "event": "end"}
                )
            state_p.update({"active": False, "remaining": 0, "reason": None})

        # spatial cooling
        state_s = self.cooling_state["spatial"]
        if self.spatial_cooldown > 0:
            self.spatial_cooldown -= 1
            state_s["active"] = True
            state_s["remaining"] = self.spatial_cooldown
            if self.spatial_cooldown == 0:
                self.cooling_events.append(
                    {"step": self.global_step, "term": "spatial", "event": "end"}
                )
                state_s.update({"active": False, "remaining": 0, "reason": None})
        else:
            if state_s.get("active"):
                self.cooling_events.append(
                    {"step": self.global_step, "term": "spatial", "event": "end"}
                )
            state_s.update({"active": False, "remaining": 0, "reason": None})
            if self.spatial_blend < 1.0:
                self.spatial_blend = min(1.0, self.spatial_blend + self.spatial_recover_step)

    def _record_train_step(
        self,
        *,
        epoch: int,
        mse_val: float,
        mae_val: float,
        spatial_info: dict,
        phys_info: dict,
        rho_spat: float,
        rho_phys: float,
        blend: float,
        spatial_blend: float,
        lr_now: float,
        cooling_flags: list[str],
        gradcos_spat: float,
        gradcos_phys: float,
        neg_cos: bool,
    ) -> None:
        if (self.global_step % self.train_log_interval) != 0:
            return

        rmse_val = math.sqrt(mse_val) if mse_val >= 0.0 else float("nan")
        record = {
            "global_step": self.global_step,
            "epoch": epoch,
            "phase": self.stage,
            "split": "train",
            "L_sup": float(mse_val),
            "L_sup_mae": float(mae_val),
            "L_sup_rmse": float(rmse_val),
            "L_spat_raw": float(spatial_info.get("raw", float("nan"))),
            "L_spat_cap": float(spatial_info.get("cap", float("nan"))),
            "L_phys_raw": float(phys_info.get("raw", float("nan"))),
            "L_phys_cap": float(phys_info.get("cap", float("nan"))),
            "rho_spat": float(rho_spat),
            "rho_phys": float(rho_phys),
            "guard_spat_triggered": bool(spatial_info.get("triggered", False)),
            "guard_phys_triggered": bool(phys_info.get("triggered", False)),
            "guard_spat_scale": float(spatial_info.get("scale", 1.0)),
            "guard_phys_scale": float(phys_info.get("scale", 1.0)),
            "pinn_blend": float(blend),
            "spat_blend": float(spatial_blend),
            "gamma": float(self.guard_gamma_pinn),
            "cooling_active": bool(self.cooling_state["pinn"]["active"]),
            "cooling_remaining": int(self.cooling_state["pinn"].get("remaining", 0)),
            "spatial_cooling_active": bool(self.cooling_state["spatial"]["active"]),
            "spatial_cooling_remaining": int(self.cooling_state["spatial"].get("remaining", 0)),
            "cos_sup_spat": float(gradcos_spat),
            "cos_sup_phys": float(gradcos_phys),
            "neg_cos_ind": bool(neg_cos),
            "rollback_flag": bool(self.just_rolled_back),
            "freeze_teacher_flag": bool(self.teacher_frozen),
            "lr": float(lr_now),
            "cooling_trigger": ";".join(cooling_flags),
            # scheme2 extra diagnostics (safe: may be nan)
            "phys_guard_metric": str(phys_info.get("metric", "")),
            "g_sup": float(phys_info.get("g_sup", float("nan"))),
            "g_phys": float(phys_info.get("g_phys", float("nan"))),
            "g_ratio": float(phys_info.get("g_ratio", float("nan"))),
            "cos_stage": str(phys_info.get("cos_stage", "")),
            "cos_scale": float(phys_info.get("cos_scale", float("nan"))),
        }
        self.step_records.append(record)

    def _record_val_epoch(
        self,
        *,
        epoch: int,
        val_mse: float,
        val_mae: float,
        val_loss: float,
        pc_raw: float,
        pm_raw: float,
        consis_raw: float,
        blend: float,
        spatial_blend: float,
        lr_now: float,
        epoch_time: float,
    ) -> None:
        rmse_val = math.sqrt(val_mse) if val_mse >= 0.0 else float("nan")
        record = {
            "global_step": self.global_step,
            "epoch": epoch,
            "phase": self.stage,
            "split": "val",
            "L_sup": float(val_mse),
            "L_sup_mae": float(val_mae),
            "L_sup_rmse": float(rmse_val),
            "L_spat_raw": float("nan"),
            "L_spat_cap": float("nan"),
            "L_phys_raw": float(pc_raw),
            "L_phys_cap": float(pc_raw),
            "rho_spat": float(self.spatial_guard_ratio),
            "rho_phys": float(self.pinn_guard_ratio),
            "guard_spat_triggered": False,
            "guard_phys_triggered": False,
            "guard_spat_scale": 1.0,
            "guard_phys_scale": 1.0,
            "pinn_blend": float(blend),
            "spat_blend": float(spatial_blend),
            "gamma": float(self.guard_gamma_pinn),
            "cooling_active": bool(self.cooling_state["pinn"]["active"]),
            "cooling_remaining": int(self.cooling_state["pinn"].get("remaining", 0)),
            "spatial_cooling_active": bool(self.cooling_state["spatial"]["active"]),
            "spatial_cooling_remaining": int(self.cooling_state["spatial"].get("remaining", 0)),
            "cos_sup_spat": float("nan"),
            "cos_sup_phys": float("nan"),
            "neg_cos_ind": False,
            "rollback_flag": bool(self.just_rolled_back),
            "freeze_teacher_flag": bool(self.teacher_frozen),
            "lr": float(lr_now),
            "cooling_trigger": "",
            "epoch_time": float(epoch_time),
            "val_loss": float(val_loss),
            "pinn_cont_raw": float(pc_raw),
            "pinn_mom_raw": float(pm_raw),
            "consis_raw": float(consis_raw),
            "cos_stage": "",
            "cos_scale": float("nan"),
        }
        self.step_records.append(record)

    def _csv_metadata_lines(self) -> list[str]:
        lines: list[str] = []
        if self.robust_meta:
            lines.append(f"robust={json.dumps(self.robust_meta)}")
        if self.stage_events:
            lines.append(f"stage_events={json.dumps(self.stage_events)}")
        if self.cooling_events:
            lines.append(f"cooling_events={json.dumps(self.cooling_events)}")
        if self.seed is not None:
            lines.append(f"seed={json.dumps(self.seed)}")
        if getattr(self, "mo_solver", None) is not None:
            lines.append(f"mo_solver={json.dumps(self.mo_solver)}")
        if getattr(self, "pinn_guard_metric", None) is not None:
            lines.append(f"pinn_guard_metric={json.dumps(self.pinn_guard_metric)}")
        return lines

    def _finalise_events(self) -> None:
        for term, state in self.cooling_state.items():
            if state.get("active"):
                self.cooling_events.append(
                    {
                        "step": self.global_step,
                        "term": term,
                        "event": "end",
                    }
                )
                state.update({"active": False, "remaining": 0, "reason": None})

    def _write_metrics_csv(self, df: pd.DataFrame, csv_path: Path) -> None:
        ensure_dir(csv_path.parent)
        meta_lines = self._csv_metadata_lines()
        with open(csv_path, "w", encoding="utf-8") as f:
            for line in meta_lines:
                f.write(f"# {line}\n")
            df.to_csv(f, index=False)

    # ----------------- multi-objective gradient utils -----------------
    @staticmethod
    def _merge_mean_grads(grads_list: list[list[Optional[torch.Tensor]]]) -> list[Optional[torch.Tensor]]:
        if not grads_list:
            return []
        T = len(grads_list)
        P = len(grads_list[0])
        merged: list[Optional[torch.Tensor]] = []

        for p_idx in range(P):
            acc = None
            cnt = 0
            for t in range(T):
                g = grads_list[t][p_idx]
                if g is None:
                    continue
                if acc is None:
                    acc = g.clone()
                else:
                    acc = acc + g
                cnt += 1
            if cnt == 0:
                merged.append(None)
            else:
                merged.append(acc / float(cnt))
        return merged

    def _merge_pcgrad(self, grads_list: list[list[Optional[torch.Tensor]]]) -> list[Optional[torch.Tensor]]:
        if not grads_list:
            return []
        if len(grads_list) == 1:
            return self._merge_mean_grads(grads_list)

        T = len(grads_list)
        P = len(grads_list[0])

        device = None
        for t in range(T):
            for g in grads_list[t]:
                if g is not None:
                    device = g.device
                    break
            if device is not None:
                break
        if device is None:
            return self._merge_mean_grads(grads_list)

        proj = [
            [g.clone() if g is not None else None for g in grads_list[t]]
            for t in range(T)
        ]

        for i in range(T):
            order = torch.randperm(T, device=device).tolist()
            for j in order:
                if j == i:
                    continue
                gi = proj[i]
                gj = proj[j]
                dot_ij = torch.tensor(0.0, device=device)
                gj_norm_sq = torch.tensor(0.0, device=device)
                for g_i, g_j in zip(gi, gj):
                    if g_i is None or g_j is None:
                        continue
                    dot_ij += torch.sum(g_i * g_j)
                    gj_norm_sq += torch.sum(g_j * g_j)
                if gj_norm_sq <= 0:
                    continue
                if dot_ij < 0:
                    coeff = dot_ij / (gj_norm_sq + 1e-12)
                    for k in range(P):
                        if gi[k] is not None and gj[k] is not None:
                            gi[k] = gi[k] - coeff * gj[k]

        return self._merge_mean_grads(proj)

    @staticmethod
    def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
        if v.numel() == 1:
            return torch.ones_like(v)
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        if not torch.any(cond):
            return torch.ones_like(v) / v.numel()
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = torch.clamp(v - theta, min=0.0)
        s = w.sum()
        if s <= 0:
            return torch.ones_like(w) / v.numel()
        return w / s

    def _merge_cagrad(self, grads_list: list[list[Optional[torch.Tensor]]]) -> list[Optional[torch.Tensor]]:
        if not grads_list:
            return []
        T = len(grads_list)
        if T == 1:
            return self._merge_mean_grads(grads_list)

        P = len(grads_list[0])

        device = None
        for t in range(T):
            for g in grads_list[t]:
                if g is not None:
                    device = g.device
                    break
            if device is not None:
                break
        if device is None:
            return self._merge_mean_grads(grads_list)

        Q = torch.zeros(T, T, device=device)
        for i in range(T):
            for j in range(T):
                s = torch.tensor(0.0, device=device)
                for g_i, g_j in zip(grads_list[i], grads_list[j]):
                    if g_i is None or g_j is None:
                        continue
                    s = s + torch.sum(g_i * g_j)
                Q[i, j] = s

        w = torch.ones(T, device=device) / float(T)
        lr = 0.1
        for _ in range(25):
            grad_w = Q @ w
            w = w - lr * grad_w
            w = self._project_to_simplex(w)

        alpha = float(self.cagrad_alpha)
        alpha = max(0.0, min(1.0, alpha))

        merged: list[Optional[torch.Tensor]] = []
        for p_idx in range(P):
            g_mean = None
            g_c = None
            cnt = 0
            for t in range(T):
                g_t = grads_list[t][p_idx]
                if g_t is None:
                    continue
                if g_mean is None:
                    g_mean = g_t.clone()
                    g_c = w[t] * g_t
                else:
                    g_mean = g_mean + g_t
                    g_c = g_c + w[t] * g_t
                cnt += 1
            if cnt == 0:
                merged.append(None)
            else:
                g_mean = g_mean / float(cnt)
                g_final = (1.0 - alpha) * g_mean + alpha * g_c
                merged.append(g_final)
        return merged

    def _backward_multi_objective(
        self,
        objectives: list[torch.Tensor],
        params: list[torch.nn.Parameter],
        solver_override: Optional[str] = None,
    ) -> None:
        # explicit FP32 multi-objective backward; AMP is disabled here
        valid_objs: list[torch.Tensor] = []
        for obj in objectives:
            if obj is None:
                continue
            if not torch.isfinite(obj):
                continue
            valid_objs.append(obj)
        if not valid_objs:
            return

        grads_list: list[list[Optional[torch.Tensor]]] = []
        for obj in valid_objs:
            grads = torch.autograd.grad(
                obj,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            grads_list.append([g if g is not None else None for g in grads])

        solver = solver_override or self.mo_solver
        if len(grads_list) == 1 or solver == "sum":
            merged = self._merge_mean_grads(grads_list)
        elif solver == "pcgrad":
            merged = self._merge_pcgrad(grads_list)
        elif solver == "cagrad":
            merged = self._merge_cagrad(grads_list)
        else:
            merged = self._merge_mean_grads(grads_list)

        for p, g in zip(params, merged):
            if (not p.requires_grad) or (g is None):
                continue
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad.copy_(g.detach())

        if getattr(self.cfg.train, "grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.train.grad_clip,
            )

        self.optimizer.step()

    # ----------------- init -----------------
    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module | None,
        cfg,
        pinn_norm_cfg: Optional[dict] = None,
        robust_meta: Optional[dict] = None,
    ):
        self.model = model
        self.teacher = teacher_model
        self.cfg = cfg
        self.robust_meta = robust_meta or {}
        runtime_cfg = getattr(cfg, "runtime", SimpleNamespace())
        self.seed = getattr(runtime_cfg, "seed", None)

        # multi-objective solver config
        mo_cfg = getattr(cfg, "mo", SimpleNamespace())
        if isinstance(mo_cfg, dict):
            mo_cfg = SimpleNamespace(**mo_cfg)
        solver = getattr(mo_cfg, "solver", "sum")
        if solver is None:
            solver = "sum"
        self.mo_solver = str(solver).lower()
        self.cagrad_alpha = float(getattr(mo_cfg, "cagrad_alpha", 0.5))

        # PINN norm mode
        self.pinn_norm_cfg = pinn_norm_cfg
        default_norm_mode = "denorm_physical" if pinn_norm_cfg is not None else "none"
        cfg_norm_mode = getattr(cfg.pinn, "norm_mode", default_norm_mode)
        self.pinn_norm_mode = cfg_norm_mode or default_norm_mode
        if isinstance(self.pinn_norm_mode, str):
            self.pinn_norm_mode = self.pinn_norm_mode.lower()

        # scheme2 switch (default: loss)
        self.pinn_guard_metric = getattr(cfg.pinn, "guard_metric", "loss")
        if self.pinn_guard_metric is None:
            self.pinn_guard_metric = "loss"
        if isinstance(self.pinn_guard_metric, str):
            self.pinn_guard_metric = self.pinn_guard_metric.lower()

        self.device = torch.device(cfg.train.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.eta_min
        )
        self.scaler = make_scaler(enabled=True, device=self.device)

        cols = [
            "epoch",
            "lambda_cont_now",
            "lcont_mult",
            "train_loss",
            "train_mae",
            "train_mse",
            "val_loss",
            "val_mae",
            "val_mse",
            "val_mae_wall",
            "val_mse_wall",
            "val_mae_interior",
            "val_mse_interior",
            "pinn_cont_w",
            "pinn_mom_w",
            "pinn_cont_raw",
            "pinn_mom_raw",
            "consis_raw",
            "consis_wt",
            "combo_score",
            "w_p",
            "w_u",
            "w_v",
            "w_w",
            "alpha_now",
            "nonfinite_train_batches",
            "amp_enabled",
            "lr",
            "focus_mode",
            "stage",
            "pinn_blend",
            "pinn_guard",
            "teacher_guard",
            "spatial_guard",
        ]
        self.hist = pd.DataFrame(columns=cols)
        self.step_records: list[dict[str, Any]] = []
        self.stage_events: list[dict[str, Any]] = [{"phase": "warm", "epoch": 0, "global_step": 0}]
        self.cooling_events: list[dict[str, Any]] = []
        self.global_step: int = 0
        self.just_rolled_back = False

        # guard config
        guard_cfg = getattr(cfg, "guard", SimpleNamespace())
        self.gradcos_interval = max(1, int(getattr(guard_cfg, "gradcos_interval", 10)))
        self.gradcos_eps = float(getattr(guard_cfg, "eps", 1e-8))
        self.disable_guard = bool(getattr(guard_cfg, "disable", False))
        self.reverse_hierarchy = bool(getattr(guard_cfg, "reverse_hierarchy", False))
        self.allow_rollback = not bool(getattr(guard_cfg, "no_rollback", False))
        self.allow_freeze_teacher = not bool(getattr(guard_cfg, "no_freeze_teacher", False))
        self.cos_control = bool(getattr(guard_cfg, "cos_control", False))
        self.cos_trust = float(getattr(guard_cfg, "cos_trust", 0.2))
        self.cos_surgery = float(getattr(guard_cfg, "cos_surgery", 0.0))
        self.cos_freeze = float(getattr(guard_cfg, "cos_freeze", -0.3))
        self.cos_scale_min = float(getattr(guard_cfg, "cos_scale_min", 0.2))
        self.cos_scale_min = max(0.0, min(1.0, self.cos_scale_min))
        self.cos_surgery_solver = str(getattr(guard_cfg, "cos_surgery_solver", "pcgrad")).lower()
        cos_bounds = sorted([self.cos_freeze, self.cos_surgery, self.cos_trust])
        self.cos_freeze, self.cos_surgery, self.cos_trust = cos_bounds
        layers_cfg = getattr(guard_cfg, "layers", ["auto"])
        if isinstance(layers_cfg, str):
            layers_cfg = [layers_cfg]
        self.gradcos_layers = list(layers_cfg)
        self.grad_params = self._select_grad_params(self.gradcos_layers)
        self.train_log_interval = max(1, int(getattr(cfg.train, "log_interval", 1)))

        # best checkpoints
        self.best_warm_val = float("inf")
        self.best_warm_state: Optional[dict[str, torch.Tensor]] = None
        self.best_warm_path: Optional[Path] = None

        self.best_balance = float("inf")
        self.path_balance: Optional[Path] = None

        self.first_ok_epoch: Optional[int] = None
        self.path_first_ok: Optional[Path] = None
        self.best_after_ok_mae: float = float("inf")
        self.path_best_after_ok: Optional[Path] = None

        # per-channel EMA for adaptive weights
        self.ema_mae = torch.ones(4, device=self.device, dtype=torch.float32) * 0.1

        # pc_raw buffer for plateau detection
        self.recent_pcraw = deque(maxlen=8)
        self.lcont_frozen = False

        # teacher warmup
        self.teacher_ref: Optional[nn.Module] = None
        self.freeze_teacher_after_warm = bool(
            getattr(cfg.teacher, "freeze_after_warm", True)
        ) and self.allow_freeze_teacher
        self.teacher_frozen = False

        # spatial consistency config
        spatial_cfg = getattr(cfg.teacher, "spatial", SimpleNamespace())
        if isinstance(spatial_cfg, dict):
            spatial_cfg = SimpleNamespace(**spatial_cfg)
        if not hasattr(spatial_cfg, "use"):
            spatial_cfg.use = False
        if not hasattr(spatial_cfg, "max_deg"):
            spatial_cfg.max_deg = 5.0
        if not hasattr(spatial_cfg, "weight"):
            spatial_cfg.weight = 0.1
        self.spatial_cfg = spatial_cfg
        self.spatial_blend = 1.0
        self.spatial_gamma = float(getattr(self.spatial_cfg, "gamma", 0.8))
        self.spatial_cooling_window = int(getattr(self.spatial_cfg, "cooling_window", 0))
        self.spatial_recover_step = float(getattr(self.spatial_cfg, "recover_step", 0.1))
        self.spatial_cooldown = 0

        # PINN guard
        self.guard_gamma_pinn = float(getattr(cfg.pinn, "guard_gamma", 0.7))
        self.cooling_window_pinn = int(getattr(cfg.pinn, "cooling_window", 3))
        self.cooling_state = {
            "pinn": {"active": False, "remaining": 0, "reason": None},
            "spatial": {"active": False, "remaining": 0, "reason": None},
        }

        self.csv_meta = {
            "robust": self.robust_meta,
            "guard_layers": self.gradcos_layers,
            "gradcos_interval": self.gradcos_interval,
            "pinn_guard_metric": self.pinn_guard_metric,
        }

        if self.teacher is not None:
            self.teacher.load_state_dict(self.model.state_dict(), strict=True)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()

        self.stage = "warm"
        # optional override: start stage immediately (if you set cfg.train.start_stage)
        start_stage = getattr(cfg.train, "start_stage", None)
        if isinstance(start_stage, str) and start_stage.strip():
            self.stage = start_stage.strip()

        self.dynamic_warmup = int(getattr(cfg.pinn, "warmup", 0))
        self.warm_no_improve = 0
        smooth_epochs_cfg = getattr(cfg.pinn, "smooth_epochs", None)
        if smooth_epochs_cfg is None:
            ramp_ref = int(getattr(cfg.pinn, "ramp", 40))
            smooth_epochs_cfg = max(5, max(1, ramp_ref) // 4)
        self.pinn_blend = 0.0
        self.pinn_blend_step = 1.0 / max(1, int(smooth_epochs_cfg))
        self.pinn_guard_ratio = float(getattr(cfg.pinn, "max_loss_ratio", 0.0) or 0.0)
        self.pinn_guard_vs_spatial = float(getattr(cfg.pinn, "max_ratio_vs_spatial", 0.0) or 0.0)
        self.guard_blend_reset = float(getattr(cfg.pinn, "guard_blend_reset", 0.65))
        self.guard_backoff = float(getattr(cfg.pinn, "guard_backoff", 0.4))
        self.guard_adapt_min = float(getattr(cfg.pinn, "guard_adapt_min", 0.35))
        self.last_guard_mean = 1.0
        self.guard_blend_decay = float(getattr(cfg.pinn, "guard_blend_decay", 0.5))
        self.guard_cooldown_default = int(getattr(cfg.pinn, "guard_cooldown", 3))
        self.guard_cooldown = 0

        self.teacher_guard_ratio = float(getattr(cfg.teacher, "max_ratio", 0.0) or 0.0)
        self.spatial_guard_ratio = float(getattr(self.spatial_cfg, "max_ratio", 0.0) or 0.0)

        if self.disable_guard:
            self.pinn_guard_ratio = 0.0
            self.pinn_guard_vs_spatial = 0.0
            self.teacher_guard_ratio = 0.0
            self.spatial_guard_ratio = 0.0
            self.guard_gamma_pinn = 1.0
            self.cooling_window_pinn = 0
            self.spatial_gamma = 1.0
            self.spatial_cooling_window = 0

    # ----------------- ratio guard (loss-ratio; legacy) -----------------
    def _apply_ratio_guard(
        self,
        raw_loss: torch.Tensor,
        anchor: Optional[torch.Tensor],
        ratio: float,
        *,
        extra_caps: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, dict]:
        # Softly cap raw_loss based on anchor and extra caps.
        info = {
            "scale": 1.0,
            "triggered": False,
            "raw": float(raw_loss.detach().mean().item()),
            "cap": float(raw_loss.detach().mean().item()),
            "limit": None,
            "ratio": float(ratio or 0.0),
            "caps": [],
            "metric": "loss",
            "g_sup": float("nan"),
            "g_phys": float("nan"),
            "g_ratio": float("nan"),
        }

        caps: list[float] = []

        # anchor-based cap
        if ratio > 0.0 and torch.is_tensor(anchor):
            anchor_det = anchor.detach()
            if anchor_det.numel() > 1:
                anchor_det = anchor_det.mean()
            anchor_val = float(anchor_det.abs().clamp_min(1e-8))
            if math.isfinite(anchor_val) and anchor_val > 0.0:
                caps.append(anchor_val * ratio)

        # additional caps
        if extra_caps:
            for cap in extra_caps:
                if cap is None:
                    continue
                cap_val = float(cap)
                if math.isfinite(cap_val) and cap_val > 0.0:
                    caps.append(cap_val)

        scale_val = 1.0
        cap_limit = None
        if caps:
            cap_val = min(caps)
            loss_val = float(raw_loss.detach().mean().abs().clamp_min(1e-12))
            if math.isfinite(loss_val) and loss_val > cap_val > 0.0:
                scale_val = max(min(cap_val / loss_val, 1.0), 0.0)
                info["triggered"] = True
                cap_limit = cap_val

        loss_term = raw_loss * scale_val
        info["scale"] = float(scale_val)
        info["cap"] = float(loss_term.detach().mean().item())
        info["limit"] = cap_limit
        info["caps"] = caps
        return loss_term, info

    # ----------------- one epoch train -----------------
    def train_one_epoch(
        self,
        dl,
        epoch_idx: int,
        lambda_cont_now: float,
        focus_mode: str,
        alpha_now: float,
        cw_default,
        cfg,
        # overrides
        lambda_mom_now: float,
        pinn_k_now: int,
        pinn_m_now: int,
        alpha_override: float | None,
        consis_mult: float,
        p_grad_scale_now: float,
        warmup_edge: int,
    ):
        m = self.model
        m.train()
        loss_sum = mae_sum = mse_sum = 0.0
        nonfinite = 0
        batch_idx = 0

        # use AMP only if solver == "sum"
        amp_enabled = (
            torch.cuda.is_available()
            and self.device.type == "cuda"
            and self.scaler.is_enabled()
            and (self.mo_solver == "sum")
        )

        pinn = None
        if cfg.pinn.use and lambda_cont_now > 0.0:
            pinn = SteadyIncompressiblePINN(
                k_neighbors=int(pinn_k_now),
                m_samples=int(pinn_m_now),
                lambda_cont=float(lambda_cont_now),
                lambda_mom=float(lambda_mom_now),
                rho=cfg.pinn.rho,
                nu_eff=cfg.pinn.nu_eff,
                residual_weight_mode=("mixed" if cfg.mixed.use else "uniform"),
                residual_alpha=(alpha_override if cfg.mixed.use else None),
                residual_sigma=cfg.mixed.wall_sigma,
                norm_mode=self.pinn_norm_mode,
                norm_cfg=self.pinn_norm_cfg,
            )

        chw_now = torch.tensor(cw_default, device=self.device, dtype=torch.float32)

        pbar = tqdm(dl, desc=f"Train {epoch_idx:03d}", leave=False)
        guard_scales: list[float] = []
        teacher_guard_scales: list[float] = []
        spatial_guard_scales: list[float] = []

        for batch in pbar:
            batch_idx += 1
            x, labels, stats = self._unpack_batch(batch)
            x = x.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            batch_stats = self._prepare_batch_stats(stats, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            cooling_flags: list[str] = []
            spatial_info = {"raw": float("nan"), "cap": float("nan"), "scale": 1.0, "triggered": False}
            phys_info = {
                "raw": float("nan"),
                "cap": float("nan"),
                "scale": 1.0,
                "triggered": False,
                "metric": "",
                "g_sup": float("nan"),
                "g_phys": float("nan"),
                "g_ratio": float("nan"),
                "cos_stage": "off",
                "cos_scale": 1.0,
            }
            spatial_raw_term_for_grad: Optional[torch.Tensor] = None
            phys_raw_term_for_grad: Optional[torch.Tensor] = None

            sup_obj_for_mo: Optional[torch.Tensor] = None
            spatial_term_eff: Optional[torch.Tensor] = None
            phys_term_eff: Optional[torch.Tensor] = None

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = forward_backbone(m, x)

                # channel-wise EMA for adaptive weight
                w_pts, _ = make_point_weights(
                    x,
                    mode=("mixed" if cfg.mixed.use else focus_mode),
                    sigma=cfg.mixed.wall_sigma,
                )
                mae_c, _ = losses_per_channel(pred, labels, w_pts)
                with torch.no_grad():
                    self.ema_mae.mul_(cfg.adapt.ema_beta).add_(
                        (1 - cfg.adapt.ema_beta) * mae_c.detach()
                    )
                    if cfg.adapt.enable and (epoch_idx >= cfg.adapt.start_epoch):
                        rel = (self.ema_mae / self.ema_mae.mean().clamp_min(1e-8)).clamp_min(1e-6)
                        w_adapt = (1.0 / rel) ** cfg.adapt.tau
                        if cfg.adapt.group_velocity:
                            w_p = w_adapt[0:1]
                            w_uvw = w_adapt[1:4].mean(0, keepdim=True)
                            w_adapt = torch.cat([w_p, w_uvw.repeat(3)], dim=0)
                        w_adapt = w_adapt.clamp(cfg.adapt.w_min, cfg.adapt.w_max)
                        w_adapt = w_adapt / w_adapt.mean()
                        chw_now = w_adapt.detach()
                    else:
                        chw_now = torch.tensor(cw_default, device=self.device, dtype=torch.float32)

                mse, mae, _ = compute_weighted_losses(
                    pred,
                    labels,
                    x,
                    focus_mode=("mixed" if cfg.mixed.use else focus_mode),
                    channel_weights=chw_now,
                    alpha=(alpha_now if cfg.mixed.use else None),
                    wall_sigma=cfg.mixed.wall_sigma,
                )

                total = mse
                sup_obj_for_mo = mse

                spatial_anchor_val: Optional[float] = None
                teacher_guard_scale = 1.0

                # teacher consistency
                consis_w = teacher_consis_weight(
                    epoch_idx,
                    cfg.teacher.use,
                    warmup_edge,
                    cfg.teacher.max_w,
                    cfg.teacher.decay_eps,
                ) * float(consis_mult)
                teacher_model = None
                if self.stage != "warm" and self.teacher_ref is not None:
                    teacher_model = self.teacher_ref
                elif self.teacher is not None:
                    teacher_model = self.teacher

                teacher_term: Optional[torch.Tensor] = None
                if (teacher_model is not None) and (consis_w > 0.0):
                    with torch.no_grad():
                        pred_t = forward_backbone(teacher_model, x)
                    loss_cons = nn.functional.mse_loss(pred, pred_t, reduction="mean")
                    raw_teacher_term = consis_w * loss_cons
                    teacher_term, teacher_guard = self._apply_ratio_guard(
                        raw_teacher_term, mse, self.teacher_guard_ratio
                    )
                    teacher_guard_scale = teacher_guard["scale"]
                    total = total + teacher_term
                    sup_obj_for_mo = sup_obj_for_mo + teacher_term
                teacher_guard_scales.append(teacher_guard_scale)

                # spatial teacher consistency
                spatial_pending = False
                spatial_raw_term_for_grad = None
                if (
                    self.stage != "warm"
                    and self.teacher_ref is not None
                    and getattr(self.spatial_cfg, "use", False)
                    and float(getattr(self.spatial_cfg, "weight", 0.0)) > 0.0
                ):
                    x_rot = self._small_rotation(x, float(getattr(self.spatial_cfg, "max_deg", 0.0)))
                    if x_rot is not None:
                        with torch.cuda.amp.autocast(enabled=amp_enabled):
                            pred_rot = forward_backbone(m, x_rot)
                        with torch.no_grad():
                            pred_t_rot = forward_backbone(self.teacher_ref, x_rot)
                        loss_spatial = nn.functional.mse_loss(pred_rot, pred_t_rot, reduction="mean")
                        w_spatial = float(self.spatial_cfg.weight)
                        spatial_raw_term = loss_spatial * w_spatial
                        spatial_raw_term_for_grad = spatial_raw_term
                        spatial_info["raw"] = float(spatial_raw_term.detach().mean().item())
                        if not self.reverse_hierarchy:
                            spatial_term, spatial_guard = self._apply_ratio_guard(
                                spatial_raw_term, mse, self.spatial_guard_ratio
                            )
                            spatial_info.update(spatial_guard)
                            spatial_anchor_val = float(spatial_guard["cap"])
                            if spatial_guard.get("triggered"):
                                cooling_flags.append(self._trigger_cooling("spatial", "overbound"))
                            spatial_guard_scales.append(spatial_guard["scale"])
                            spatial_term_eff = spatial_term * self.spatial_blend
                            total = total + spatial_term_eff
                        else:
                            spatial_pending = True
                            spatial_info["cap"] = float(spatial_raw_term.detach().mean().item())
                    else:
                        spatial_guard_scales.append(1.0)
                else:
                    spatial_guard_scales.append(1.0)

                # PINN loss
                phys_guard = None
                phys_term_pending: Optional[torch.Tensor] = None
                if pinn is not None:
                    pred_for_pinn = pred.clone()
                    if epoch_idx <= cfg.p_grad.release_epoch:
                        pred_for_pinn[:, 0, :] = pred_for_pinn[:, 0, :].detach()
                    else:
                        p_slice = pred_for_pinn[:, 0, :]

                        def _scale_hook(g):
                            return g * float(p_grad_scale_now)

                        p_slice.requires_grad_(True)
                        p_slice.register_hook(_scale_hook)
                        pred_for_pinn[:, 0, :] = p_slice

                    pinn_out = pinn(x, pred_for_pinn, batch_stats=batch_stats)
                    if torch.isfinite(pinn_out["loss_pinn"]):
                        phys_raw_term_for_grad = pinn_out["loss_pinn"]
                        phys_info["raw"] = float(phys_raw_term_for_grad.detach().mean().item())

                        if not self.reverse_hierarchy:
                            # --- scheme2: grad-ratio guard if enabled ---
                            if (self.pinn_guard_metric == "grad") and (self.pinn_guard_ratio > 0.0):
                                # use sup mse as anchor (strict supervision-first)
                                phys_term, phys_guard = self._apply_grad_ratio_guard(
                                    phys_raw_term_for_grad,
                                    mse,
                                    self.pinn_guard_ratio,
                                    eps=self.gradcos_eps,
                                    params=None,
                                )
                            else:
                                extra_caps: list[float] = []
                                if (
                                    self.pinn_guard_vs_spatial > 0.0
                                    and spatial_anchor_val is not None
                                    and spatial_anchor_val > 0.0
                                ):
                                    extra_caps.append(spatial_anchor_val * self.pinn_guard_vs_spatial)
                                phys_term, phys_guard = self._apply_ratio_guard(
                                    phys_raw_term_for_grad,
                                    mse,
                                    self.pinn_guard_ratio,
                                    extra_caps=extra_caps,
                                )

                            if phys_guard is not None:
                                phys_info.update(phys_guard)
                                if phys_guard.get("triggered") and (not self.cos_control):
                                    cooling_flags.append(self._trigger_cooling("pinn", "overbound"))
                                guard_scales.append(float(phys_guard.get("scale", 1.0)))

                            phys_term_eff = phys_term
                            total = total + phys_term_eff
                        else:
                            phys_term_pending = phys_raw_term_for_grad
                            phys_info["cap"] = float(phys_raw_term_for_grad.detach().mean().item())
                    else:
                        guard_scales.append(1.0)
                else:
                    guard_scales.append(1.0)

                if self.reverse_hierarchy:
                    if phys_term_pending is not None:
                        if (self.pinn_guard_metric == "grad") and (self.pinn_guard_ratio > 0.0):
                            phys_term, phys_guard = self._apply_grad_ratio_guard(
                                phys_term_pending,
                                mse,
                                self.pinn_guard_ratio,
                                eps=self.gradcos_eps,
                                params=None,
                            )
                        else:
                            phys_term, phys_guard = self._apply_ratio_guard(
                                phys_term_pending,
                                mse,
                                self.pinn_guard_ratio,
                            )

                        if phys_guard is not None:
                            phys_info.update(phys_guard)
                            if phys_guard.get("triggered") and (not self.cos_control):
                                cooling_flags.append(self._trigger_cooling("pinn", "overbound"))
                            guard_scales.append(float(phys_guard.get("scale", 1.0)))

                        phys_term_eff = phys_term
                        total = total + phys_term_eff

                    if spatial_pending and spatial_raw_term_for_grad is not None:
                        extra_caps2: list[float] = []
                        cap_val = phys_info.get("cap")
                        if cap_val is not None and self.pinn_guard_vs_spatial > 0.0:
                            extra_caps2.append(float(cap_val) * self.pinn_guard_vs_spatial)
                        spatial_term, spatial_guard = self._apply_ratio_guard(
                            spatial_raw_term_for_grad,
                            None,
                            0.0,
                            extra_caps=extra_caps2,
                        )
                        spatial_info.update(spatial_guard)
                        if spatial_guard.get("triggered"):
                            cooling_flags.append(self._trigger_cooling("spatial", "overbound"))
                        spatial_guard_scales.append(spatial_guard["scale"])
                        spatial_term_eff = spatial_term * self.spatial_blend
                        total = total + spatial_term_eff
                    elif spatial_pending:
                        spatial_guard_scales.append(1.0)

            gradcos_spat = float("nan")
            gradcos_phys = float("nan")
            neg_cos = False
            cos_stage = "off"
            cos_scale = 1.0
            cos_solver_override: Optional[str] = None
            if self.grad_params and (self.global_step % self.gradcos_interval == 0):
                gradcos_spat = self._compute_grad_cos(mse, spatial_raw_term_for_grad)
                gradcos_phys = self._compute_grad_cos(mse, phys_raw_term_for_grad)
                if math.isfinite(gradcos_spat) and gradcos_spat < 0:
                    cooling_flags.append(self._trigger_cooling("spatial", "neg-cos"))
                    neg_cos = True
                if math.isfinite(gradcos_phys) and gradcos_phys < 0:
                    if not self.cos_control:
                        cooling_flags.append(self._trigger_cooling("pinn", "neg-cos"))
                    neg_cos = True

            if self.cos_control and math.isfinite(gradcos_phys) and phys_term_eff is not None:
                cos_stage, cos_scale, cos_solver_override = self._cosine_control(gradcos_phys)
                phys_info["cos_stage"] = cos_stage
                phys_info["cos_scale"] = float(cos_scale)
                if cos_stage == "freeze":
                    total = total - phys_term_eff
                    phys_term_eff = None
                    cooling_flags.append(self._trigger_cooling("pinn", "cos-freeze"))
                elif cos_stage in {"scale", "surgery"}:
                    if cos_scale < 1.0:
                        total = total - phys_term_eff
                        phys_term_eff = phys_term_eff * float(cos_scale)
                        total = total + phys_term_eff
                    if cos_stage == "surgery":
                        cooling_flags.append(self._trigger_cooling("pinn", "cos-surgery"))
                    else:
                        cooling_flags.append(self._trigger_cooling("pinn", "cos-scale"))

            # multi-objectives for solver: sup + spatial + physics
            mo_objectives: list[torch.Tensor] = []
            if sup_obj_for_mo is not None:
                mo_objectives.append(sup_obj_for_mo)
            if spatial_term_eff is not None:
                mo_objectives.append(spatial_term_eff)
            if phys_term_eff is not None:
                mo_objectives.append(phys_term_eff)

            if not torch.isfinite(total):
                nonfinite += 1
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix_str("non-finite batch")
                continue

            cooling_active = (
                bool(self.cooling_state["pinn"].get("active", False))
                or bool(self.cooling_state["spatial"].get("active", False))
                or bool(phys_info.get("triggered", False))
                or bool(spatial_info.get("triggered", False))
                or bool(neg_cos)
            )
            use_mo = (
                (cooling_active or cos_stage == "surgery")
                and (self.mo_solver is not None)
                and (self.mo_solver != "sum")
                and (len(mo_objectives) >= 2)
            )

            if use_mo:
                params = [p for p in self.model.parameters() if p.requires_grad]
                self._backward_multi_objective(
                    mo_objectives,
                    params,
                    solver_override=cos_solver_override,
                )
            else:
                self.scaler.scale(total).backward()
                if cfg.train.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            lr_now = self.optimizer.param_groups[0]["lr"]
            self._record_train_step(
                epoch=epoch_idx,
                mse_val=float(mse.detach().item()),
                mae_val=float(mae.detach().item()),
                spatial_info=spatial_info,
                phys_info=phys_info,
                rho_spat=self.spatial_guard_ratio,
                rho_phys=self.pinn_guard_ratio,
                blend=float(self.pinn_blend),
                spatial_blend=float(self.spatial_blend),
                lr_now=float(lr_now),
                cooling_flags=cooling_flags,
                gradcos_spat=gradcos_spat,
                gradcos_phys=gradcos_phys,
                neg_cos=neg_cos,
            )

            self.global_step += 1
            self.just_rolled_back = False

            loss_sum += total.item()
            mae_sum += mae.item()
            mse_sum += mse.item()
            pbar.set_postfix(loss=f"{loss_sum / max(batch_idx, 1):.4f}")

        guard_mean = (sum(guard_scales) / len(guard_scales)) if guard_scales else 1.0
        teacher_guard_mean = (
            sum(teacher_guard_scales) / len(teacher_guard_scales) if teacher_guard_scales else 1.0
        )
        spatial_guard_mean = (
            sum(spatial_guard_scales) / len(spatial_guard_scales) if spatial_guard_scales else 1.0
        )

        return (
            loss_sum / max(batch_idx, 1),
            mae_sum / max(batch_idx, 1),
            mse_sum / max(batch_idx, 1),
            nonfinite,
            chw_now.detach(),
            guard_mean,
            teacher_guard_mean,
            spatial_guard_mean,
        )

    # ----------------- evaluate -----------------
    @torch.no_grad()
    def evaluate(
        self,
        dl,
        lambda_cont_now: float,
        focus_mode: str,
        alpha_now: float,
        cw_eval,
        cfg,
        consis_w: float = 0.0,
        # overrides
        lambda_mom_now: float = 0.0,
        pinn_k_now: int = 24,
        pinn_m_now: int = 4096,
        alpha_override: float | None = None,
    ):
        self.model.eval()
        mae_sum = mse_sum = 0.0
        wall_mae_sum = wall_mse_sum = int_mae_sum = int_mse_sum = 0.0
        pinn_cont_w = pinn_mom_w = 0.0
        pinn_cont_raw = pinn_mom_raw = 0.0
        consis_raw = consis_wt = 0.0

        pinn_eval = None
        if cfg.pinn.use and lambda_cont_now > 0.0:
            pinn_eval = SteadyIncompressiblePINN(
                k_neighbors=int(pinn_k_now),
                m_samples=int(pinn_m_now),
                lambda_cont=float(lambda_cont_now),
                lambda_mom=float(lambda_mom_now),
                rho=cfg.pinn.rho,
                nu_eff=cfg.pinn.nu_eff,
                residual_weight_mode=("mixed" if cfg.mixed.use else "uniform"),
                residual_alpha=(alpha_override if cfg.mixed.use else None),
                residual_sigma=cfg.mixed.wall_sigma,
                norm_mode=self.pinn_norm_mode,
                norm_cfg=self.pinn_norm_cfg,
            )

        pbar = tqdm(dl, desc="Valid", leave=False)
        for batch in pbar:
            x, labels, stats = self._unpack_batch(batch)
            x = x.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            batch_stats = self._prepare_batch_stats(stats, self.device)

            pred = forward_backbone(self.model, x)

            mse, mae, d = compute_weighted_losses(
                pred,
                labels,
                x,
                ("mixed" if cfg.mixed.use else focus_mode),
                channel_weights=cw_eval,
                alpha=(alpha_now if cfg.mixed.use else None),
                wall_sigma=cfg.mixed.wall_sigma,
            )

            teacher_model = None
            if self.stage != "warm" and self.teacher_ref is not None:
                teacher_model = self.teacher_ref
            elif self.teacher is not None:
                teacher_model = self.teacher
            if teacher_model is not None and consis_w > 0.0:
                pred_t = forward_backbone(teacher_model, x)
                cr = nn.functional.mse_loss(pred, pred_t, reduction="mean")
                consis_raw += float(cr)
                consis_wt += float(consis_w) * float(cr)

            if pinn_eval is not None:
                po = pinn_eval(x, pred, batch_stats=batch_stats)
                if torch.isfinite(po["loss_cont"]) and torch.isfinite(po["loss_mom"]):
                    pinn_cont_w += float(po["loss_cont"])
                    pinn_mom_w += float(po["loss_mom"])
                if torch.isfinite(po["cont_raw_mean"]) and torch.isfinite(po["mom_raw_mean"]):
                    pinn_cont_raw += float(po["cont_raw_mean"])
                    pinn_mom_raw += float(po["mom_raw_mean"])

            mae_sum += mae.item()
            mse_sum += mse.item()
            reg = compute_region_metrics(pred, labels, d)
            wall_mae_sum += reg["mae_wall"]
            wall_mse_sum += reg["mse_wall"]
            int_mae_sum += reg["mae_interior"]
            int_mse_sum += reg["mse_interior"]

            pbar.set_postfix(loss=f"{mse_sum / max(1, len(self.hist) + 1):.4f}")

        n = max(len(dl), 1)
        val_mae = mae_sum / n
        val_mse = mse_sum / n
        val_loss = val_mse

        return (
            val_loss,
            val_mae,
            val_mse,
            wall_mae_sum / n,
            wall_mse_sum / n,
            int_mae_sum / n,
            int_mse_sum / n,
            pinn_cont_w / n if pinn_eval is not None else 0.0,
            pinn_mom_w / n if pinn_eval is not None else 0.0,
            pinn_cont_raw / n if pinn_eval is not None else 0.0,
            pinn_mom_raw / n if pinn_eval is not None else 0.0,
            consis_raw / n,
            consis_wt / n,
        )

    # ----------------- main training loop -----------------
    def run(self, dl_train, dl_val, paths, cfg):
        ensure_dir(paths.save_dir)
        ensure_dir(paths.csv_path.parent)
        ensure_dir(paths.curve_dir)

        try:
            num_train_batches = len(dl_train)
        except TypeError:
            num_train_batches = None
        try:
            num_val_batches = len(dl_val)
        except TypeError:
            num_val_batches = None

        if not num_train_batches:
            raise RuntimeError(
                "Training DataLoader produced zero batches. Please check out.csv / dataset contents before training."
            )
        if not num_val_batches:
            raise RuntimeError(
                "Validation DataLoader produced zero batches. Please ensure the validation split has data."
            )

        log_txt = Path(paths.save_dir) / "train_log.txt"
        with open(log_txt, "w") as f:
            f.write(
                "epoch,lambda_cont,lcont_mult,train_loss,val_loss,val_mae,pc_raw,pc_w,combo,alpha,lr,mode\n"
            )

        lcont_mult = cfg.pinn.lcont_mult_init
        last_adjust = 0
        div_target_calibrated = False
        div_target_value = cfg.pinn.div_target

        print("Start Training...")

        BOOST_N = getattr(cfg.pinn, "boost_epochs", 30)

        last_epoch_executed = 0
        saved_any_checkpoint = False

        for epoch in range(1, cfg.train.epochs + 1):
            epoch_start = time.time()
            frac = epoch / cfg.train.epochs
            focus_mode = (
                "mixed"
                if cfg.mixed.use
                else ("wall_focus" if (frac < cfg.curriculum.split) else "interior_focus")
            )
            alpha_now = current_alpha(
                epoch,
                cfg.mixed.use,
                cfg.mixed.alpha_start,
                cfg.mixed.alpha_end,
                cfg.mixed.decay_start,
                cfg.mixed.decay_epochs,
                cfg.mixed.schedule,
            )

            warm_edge = int(self.dynamic_warmup)
            lambda_base = base_lambda_cont(
                epoch,
                cfg.pinn.use,
                warm_edge,
                cfg.pinn.ramp,
                cfg.pinn.lmin,
                cfg.pinn.lmax,
                cfg.pinn.schedule,
                cfg.pinn.mode,
            )
            lambda_cont_now = lambda_base * (lcont_mult if cfg.pinn.adapt else 1.0)
            if cfg.pinn.adapt:
                lambda_cont_now = max(
                    0.0, min(lambda_cont_now, cfg.pinn.lmax * cfg.pinn.lcont_max_mult)
                )

            update_teacher = False
            if self.teacher is not None:
                if self.stage == "warm":
                    update_teacher = True
                elif not self.freeze_teacher_after_warm:
                    update_teacher = True
                elif not self.teacher_frozen:
                    update_teacher = True
            if update_teacher:
                ema_now = float(cfg.teacher.ema_beta)
                if self.stage != "warm":
                    ema_now = float(
                        getattr(cfg.teacher, "ema_beta_balance", cfg.teacher.ema_beta)
                    )
                with torch.no_grad():
                    td = self.teacher.state_dict()
                    msd = self.model.state_dict()
                    for k in td.keys():
                        td[k].copy_(ema_now * td[k] + (1 - ema_now) * msd[k])

            target_now = div_target_value if div_target_calibrated else cfg.pinn.div_target
            hi = target_now * (1.0 + cfg.pinn.div_tol)
            hi2 = target_now * (1.0 + 2.0 * cfg.pinn.div_tol)

            in_pinn = epoch > warm_edge
            if self.stage == "warm":
                in_pinn = False
            if self.stage == "warmup":
                in_pinn = epoch > warm_edge

            lambda_mom_now = 0.0
            pinn_k_now = int(cfg.pinn.k)
            pinn_m_now = int(cfg.pinn.samples)
            alpha_override = alpha_now if cfg.mixed.use else None
            consis_mult = 1.0
            p_grad_scale_now = cfg.p_grad.scale

            prev_pc_raw = float(self.hist.iloc[-1]["pinn_cont_raw"]) if len(self.hist) > 0 else 1e9
            if in_pinn:
                stage_epoch = max(1, epoch - warm_edge)
                if stage_epoch <= BOOST_N:
                    lambda_mom_now = 0.0
                    lambda_cont_now = min(
                        cfg.pinn.lmax * cfg.pinn.lcont_max_mult,
                        lambda_cont_now * 1.25,
                    )
                    pinn_k_now = max(pinn_k_now, 40)
                    pinn_m_now = max(pinn_m_now, 8192)
                    alpha_override = max(alpha_now, 0.95) if cfg.mixed.use else None
                    consis_mult = 0.3
                    p_grad_scale_now = min(p_grad_scale_now, 0.02)
                else:
                    if prev_pc_raw <= hi:
                        lambda_mom_now = max(cfg.pinn.lambda_mom, 1e-4)
                        lambda_cont_now = max(lambda_cont_now, cfg.pinn.lmin)
                        pinn_k_now = int(cfg.pinn.k)
                        pinn_m_now = int(cfg.pinn.samples)
                        alpha_override = max(alpha_now, 0.85) if cfg.mixed.use else None
                        consis_mult = 0.6
                        p_grad_scale_now = cfg.p_grad.scale
                    else:
                        lambda_mom_now = 0.0
                        pinn_k_now = max(pinn_k_now, 36)
                        pinn_m_now = max(pinn_m_now, 6144)
                        alpha_override = max(alpha_now, 0.9) if cfg.mixed.use else None
                        consis_mult = 0.4
                        if prev_pc_raw > hi2:
                            lambda_cont_now = min(
                                cfg.pinn.lmax * cfg.pinn.lcont_max_mult,
                                lambda_cont_now * 1.15,
                            )
                            p_grad_scale_now = min(p_grad_scale_now, 0.02)

            if in_pinn:
                if self.guard_cooldown > 0:
                    self.guard_cooldown -= 1
                else:
                    self.pinn_blend = min(1.0, self.pinn_blend + self.pinn_blend_step)
            else:
                self.pinn_blend = 0.0
                self.guard_cooldown = 0

            self._update_cooling_states_after_epoch(in_pinn)
            blend = self.pinn_blend if in_pinn else 0.0
            lambda_cont_now *= blend
            lambda_mom_now *= blend
            pinn_active = in_pinn and (blend > 0.0) and (lambda_cont_now > 0.0)

            consis_w_for_log = teacher_consis_weight(
                epoch,
                cfg.teacher.use,
                warm_edge,
                cfg.teacher.max_w,
                cfg.teacher.decay_eps,
            ) * consis_mult

            (
                tr_loss,
                tr_mae,
                tr_mse,
                nonfinite,
                cw_epoch,
                guard_mean,
                teacher_guard_mean,
                spatial_guard_mean,
            ) = self.train_one_epoch(
                dl_train,
                epoch,
                lambda_cont_now,
                focus_mode,
                alpha_now,
                cfg.loss.channel_weights,
                cfg,
                lambda_mom_now=lambda_mom_now,
                pinn_k_now=pinn_k_now,
                pinn_m_now=pinn_m_now,
                alpha_override=alpha_override,
                consis_mult=consis_mult,
                p_grad_scale_now=p_grad_scale_now,
                warmup_edge=warm_edge,
            )

            (
                val_loss,
                val_mae,
                val_mse,
                v_mae_w,
                v_mse_w,
                v_mae_i,
                v_mse_i,
                pc_w,
                pm_w,
                pc_raw,
                pm_raw,
                consis_raw,
                consis_wt,
            ) = self.evaluate(
                dl_val,
                lambda_cont_now,
                focus_mode,
                alpha_now,
                cw_epoch,
                cfg,
                consis_w=consis_w_for_log,
                lambda_mom_now=lambda_mom_now,
                pinn_k_now=pinn_k_now,
                pinn_m_now=pinn_m_now,
                alpha_override=alpha_override,
            )

            self._record_val_epoch(
                epoch=epoch,
                val_mse=float(val_mse),
                val_mae=float(val_mae),
                val_loss=float(val_loss),
                pc_raw=float(pc_raw),
                pm_raw=float(pm_raw),
                consis_raw=float(consis_raw),
                blend=float(self.pinn_blend),
                spatial_blend=float(self.spatial_blend),
                lr_now=float(self.optimizer.param_groups[0]["lr"]),
                epoch_time=float(time.time() - epoch_start),
            )

            self.last_guard_mean = float(guard_mean)

            if in_pinn and (self.last_guard_mean < self.guard_blend_reset):
                self.pinn_blend = min(
                    self.pinn_blend * self.guard_blend_decay,
                    self.last_guard_mean,
                )
                self.guard_cooldown = max(
                    self.guard_cooldown,
                    self.guard_cooldown_default,
                )

                self.recent_pcraw.append(float(pc_raw))
                if (
                    (not self.lcont_frozen)
                    and (len(self.recent_pcraw) == self.recent_pcraw.maxlen)
                ):
                    vals = list(self.recent_pcraw)
                    mean_v = sum(vals) / max(1e-12, len(vals))
                    var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)
                    std_v = math.sqrt(max(0.0, var_v))
                    if (all(v <= hi for v in vals)) and (mean_v > 0) and (std_v / mean_v < 0.06):
                        self.lcont_frozen = True

                if (
                    cfg.pinn.auto_calib
                    and (not div_target_calibrated)
                    and (epoch == max(5, max(1, warm_edge // 2)))
                ):
                    if pc_raw > 0:
                        div_target_value = float(
                            max(1e-8, pc_raw * cfg.pinn.calib_mult)
                        )
                        div_target_calibrated = True

            combo_score = val_loss + cfg.loss.phys_alpha * float(pc_raw)

            lr_now = self.optimizer.param_groups[0]["lr"]
            w_list = cw_epoch.detach().cpu().tolist()
            self.hist.loc[len(self.hist)] = [
                epoch,
                lambda_cont_now,
                (lcont_mult if cfg.pinn.adapt else 1.0),
                tr_loss,
                tr_mae,
                tr_mse,
                val_loss,
                val_mae,
                val_mse,
                v_mae_w,
                v_mse_w,
                v_mae_i,
                v_mse_i,
                pc_w,
                pm_w,
                pc_raw,
                pm_raw,
                consis_raw,
                consis_wt,
                combo_score,
                w_list[0],
                w_list[1],
                w_list[2],
                w_list[3],
                (0.0 if alpha_now is None else alpha_now),
                nonfinite,
                self.scaler.is_enabled(),
                lr_now,
                focus_mode,
                self.stage,
                self.pinn_blend if in_pinn else 0.0,
                guard_mean,
                teacher_guard_mean,
                spatial_guard_mean,
            ]

            last_epoch_executed = epoch

            # ---- warm phase -> warmup transition ----
            if self.stage == "warm":
                patience_cfg = int(getattr(cfg.early, "patience", 0))
                patience_cfg = max(0, patience_cfg)
                delta_early = float(getattr(cfg.early, "delta", 1.0e-4))

                metric_now = float(val_loss)
                if metric_now + delta_early < self.best_warm_val:
                    self.best_warm_val = metric_now
                    self.warm_no_improve = 0

                    self.best_warm_state = copy.deepcopy(self.model.state_dict())
                    if self.allow_rollback:
                        self.best_warm_path = (
                            paths.save_dir
                            / f"best_warm_val{self.best_warm_val:.6f}_ep{epoch:03d}.pth"
                        )
                        save_state(self.best_warm_state, self.best_warm_path)
                else:
                    self.warm_no_improve += 1

                should_switch = False
                if patience_cfg > 0:
                    if self.warm_no_improve >= patience_cfg:
                        should_switch = True
                else:
                    if (cfg.pinn.warmup > 0) and (epoch >= cfg.pinn.warmup):
                        should_switch = True

                if should_switch:
                    rollback_event = {
                        "phase": "rollback",
                        "epoch": epoch,
                        "global_step": self.global_step,
                    }
                    warm_state = None
                    if self.allow_rollback:
                        warm_state = self._resolve_warm_best_state()
                        if warm_state is not None:
                            self.model.load_state_dict(warm_state, strict=True)
                            if self.teacher is not None:
                                self.teacher.load_state_dict(warm_state, strict=True)
                                for p in self.teacher.parameters():
                                    p.requires_grad_(False)
                                self.teacher.eval()
                        else:
                            warm_state = {
                                k: v.detach().clone()
                                for k, v in self.model.state_dict().items()
                            }
                    else:
                        rollback_event["skipped"] = True
                        warm_state = {
                            k: v.detach().clone()
                            for k, v in self.model.state_dict().items()
                        }

                    self.stage_events.append(rollback_event)

                    self._refresh_teacher_ref(warm_state)
                    if self.teacher is not None and self.freeze_teacher_after_warm:
                        self.teacher_frozen = True
                    if not self.allow_freeze_teacher:
                        self.teacher_frozen = False

                    self.stage = "warmup"
                    self.stage_events.append(
                        {
                            "phase": "warmup",
                            "epoch": epoch,
                            "global_step": self.global_step,
                        }
                    )
                    self.just_rolled_back = True

                    self.dynamic_warmup = int(epoch)
                    self.pinn_blend = 0.0
                    self.warm_no_improve = 0

            # ---- λ_cont adapt ----
            if (
                (not self.lcont_frozen)
                and cfg.pinn.adapt
                and epoch >= cfg.pinn.adapt_after
                and (lambda_base > 0)
                and (self.last_guard_mean >= self.guard_adapt_min)
            ):
                if (epoch - last_adjust) >= cfg.pinn.adapt_interval:
                    target = div_target_value if div_target_calibrated else cfg.pinn.div_target
                    hi_adapt = target * (1 + cfg.pinn.div_tol)
                    lo_adapt = target * (1 - cfg.pinn.div_tol)
                    if pc_raw > hi_adapt:
                        lcont_mult = min(
                            cfg.pinn.lcont_max_mult, lcont_mult * cfg.pinn.up_rate
                        )
                    elif pc_raw < lo_adapt:
                        lcont_mult = max(
                            cfg.pinn.lcont_min_mult, lcont_mult * cfg.pinn.down_rate
                        )
                    last_adjust = epoch
            elif self.last_guard_mean < self.guard_backoff:
                lcont_mult = max(
                    cfg.pinn.lcont_min_mult,
                    lcont_mult * max(self.last_guard_mean, self.guard_backoff),
                )

            # ---- checkpoint selection ----
            div_ok = (
                div_target_value if div_target_calibrated else cfg.pinn.div_target
            ) * (1.0 + cfg.save.ok_margin)
            if pinn_active and (pc_raw <= div_ok):
                if self.first_ok_epoch is None:
                    self.first_ok_epoch = epoch
                    self.path_first_ok = (
                        paths.save_dir
                        / f"best_firstOK_pc{pc_raw:.6f}_ep{epoch:03d}.pth"
                    )
                    save_state(self.model.state_dict(), self.path_first_ok)
                    saved_any_checkpoint = True
                    self.best_after_ok_mae = float(val_mae)
                    self.path_best_after_ok = (
                        paths.save_dir
                        / f"best_afterOK_mae{self.best_after_ok_mae:.6f}_ep{epoch:03d}.pth"
                    )
                    symlink_or_copy(self.path_first_ok, self.path_best_after_ok)
                else:
                    if (val_mae + 1e-12) < (self.best_after_ok_mae - cfg.early.delta):
                        self.best_after_ok_mae = float(val_mae)
                        self.path_best_after_ok = (
                            paths.save_dir
                            / f"best_afterOK_mae{self.best_after_ok_mae:.6f}_ep{epoch:03d}.pth"
                        )
                        save_state(self.model.state_dict(), self.path_best_after_ok)
                        saved_any_checkpoint = True

            if pinn_active and (pc_raw <= div_ok):
                if combo_score < self.best_balance - cfg.early.delta:
                    self.best_balance = combo_score
                    self.path_balance = (
                        paths.save_dir
                        / f"best_balance_{self.best_balance:.6f}_ep{epoch:03d}.pth"
                    )
                    save_state(self.model.state_dict(), self.path_balance)
                    saved_any_checkpoint = True

            if (epoch % 20) == 0:
                ckpt = (
                    paths.save_dir
                    / f"epoch{epoch:03d}_tl{tr_loss:.6f}_vl{val_loss:.6f}_pcRaw{pc_raw:.6f}.pth"
                )
                save_state(self.model.state_dict(), ckpt)
                saved_any_checkpoint = True

            with open(log_txt, "a") as f:
                f.write(
                    f"{epoch},{lambda_cont_now:.6g},{lcont_mult:.3f},{tr_loss:.6f},{val_loss:.6f},{val_mae:.6f},"
                    f"{pc_raw:.6f},{pc_w:.6f},{combo_score:.6f},{0.0 if alpha_now is None else alpha_now:.3f},"
                    f"{lr_now:.3e},{focus_mode}\n"
                )

            self.scheduler.step()

        if len(self.hist) == 0:
            raise RuntimeError(
                "Training finished without executing any epochs. Check the training configuration (epochs) and dataloaders."
            )

        if not saved_any_checkpoint:
            fallback = paths.save_dir / f"final_epoch{last_epoch_executed:03d}.pth"
            save_state(self.model.state_dict(), fallback)
            saved_any_checkpoint = True
            if self.path_first_ok is None:
                self.path_first_ok = fallback

        final_src = None
        if self.path_best_after_ok is not None and Path(self.path_best_after_ok).exists():
            final_src = self.path_best_after_ok
        elif self.path_first_ok is not None and Path(self.path_first_ok).exists():
            final_src = self.path_first_ok
        elif self.path_balance is not None and self.path_balance.exists():
            final_src = self.path_balance
        else:
            last_ckpts = sorted(Path(paths.save_dir).glob("epoch*.pth"))
            if len(last_ckpts) > 0:
                final_src = last_ckpts[-1]
            elif self.best_warm_path is not None and self.best_warm_path.exists():
                final_src = self.best_warm_path

        if final_src is None:
            fallback = paths.save_dir / f"final_epoch{last_epoch_executed:03d}.pth"
            save_state(self.model.state_dict(), fallback)
            final_src = fallback

        if final_src is not None:
            symlink_or_copy(final_src, paths.final_reco)

        self._finalise_events()
        if self.step_records:
            df_steps = pd.DataFrame(self.step_records)
            self._write_metrics_csv(df_steps, paths.csv_path)

        return self.hist