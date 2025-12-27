# -*- coding: utf-8 -*-
"""
tpac_pinn_adapter.py
把 TPAC 的 (B,N,3) + {"p","u","v","w"} 适配为 SteadyIncompressiblePINN 的
(B,3,N) + (B,4,N) 接口；返回值完全沿用原 PINN 的字典键。
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from models.tpac_pinn import SteadyIncompressiblePINN

__all__ = ["TPACPINNAdapter", "build_pinn"]


class TPACPINNAdapter(nn.Module):
    """
    适配器：仅做维度转换与通道拼接；同时可选地托管 PINN 的 norm_mode/norm_cfg 与动态 λ。

    输入：
        preds: dict，包含键 "p","u","v","w"，支持张量形状：
               (B,N) 或 (B,1,N) 或 (B,N,1)
        xyz_bnc: torch.Tensor, 形状 (B,N,3)

    常用：
        adapter.update_lambdas(lambda_cont_now, lambda_mom_now)
        out = adapter.loss(preds, xyz_bnc)
    """
    def __init__(self, *, norm_mode: str = "none", norm_cfg: Optional[Dict] = None, **pinn_kwargs: Any):
        super().__init__()
        # 透传 norm_mode / norm_cfg
        self.pinn = SteadyIncompressiblePINN(norm_mode=norm_mode, norm_cfg=(norm_cfg or {}), **pinn_kwargs)

    # ---- 动态更新 λ（可选，但推荐这样做而不是每个 epoch 重建 PINN）----
    @torch.no_grad()
    def update_lambdas(self, lambda_cont: Optional[float] = None, lambda_mom: Optional[float] = None):
        if lambda_cont is not None:
            self.pinn.lambda_cont = float(lambda_cont)
        if lambda_mom is not None:
            self.pinn.lambda_mom = float(lambda_mom)

    @staticmethod
    def _to_b1n(t: torch.Tensor) -> torch.Tensor:
        """
        将 (B,N) / (B,1,N) / (B,N,1) 统一为 (B,1,N)，保留梯度。
        """
        assert t.dim() in (2, 3), f"pred tensor must be 2D or 3D, got {t.shape}"
        if t.dim() == 2:
            # (B,N) -> (B,1,N)
            return t.unsqueeze(1)
        # 3D:
        if t.size(1) == 1:      # (B,1,N)
            return t
        if t.size(2) == 1:      # (B,N,1) -> (B,1,N)
            return t.transpose(1, 2)
        raise AssertionError(f"Expect (B,1,N) or (B,N,1) for 3D, got {t.shape}")

    @staticmethod
    def _pack_pred(preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        dict -> (B,4,N)，通道顺序固定为 [p,u,v,w]，保留计算图。
        """
        for k in ("p", "u", "v", "w"):
            assert k in preds, f"preds missing key: '{k}'"
        p = TPACPINNAdapter._to_b1n(preds["p"]).contiguous()
        u = TPACPINNAdapter._to_b1n(preds["u"]).contiguous()
        v = TPACPINNAdapter._to_b1n(preds["v"]).contiguous()
        w = TPACPINNAdapter._to_b1n(preds["w"]).contiguous()
        # (B,4,N), 保留梯度
        y = torch.cat([p, u, v, w], dim=1)
        return y

    @staticmethod
    def _pack_xyz(xyz_bnc: torch.Tensor) -> torch.Tensor:
        """
        (B,N,3) -> (B,3,N)，保留计算图。
        """
        assert xyz_bnc.dim() == 3 and xyz_bnc.size(-1) == 3, f"xyz must be (B,N,3), got {xyz_bnc.shape}"
        return xyz_bnc.transpose(1, 2).contiguous()

    def loss(self, preds: Dict[str, torch.Tensor], xyz_bnc: torch.Tensor,
             batch_stats: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        训练与验证阶段统一使用的入口。
        返回键与 SteadyIncompressiblePINN.forward 完全一致。
        """
        device = xyz_bnc.device
        dtype  = xyz_bnc.dtype

        y_pred_b4n = self._pack_pred(preds).to(device=device, dtype=dtype)
        xyz_bcn    = self._pack_xyz(xyz_bnc).to(device=device, dtype=dtype)

        # 不要 no_grad，需要 PINN 的损失回传到 preds（除了对 p 的自定义 hook）
        return self.pinn(xyz_bcn, y_pred_b4n, batch_stats=batch_stats)

    # 别名：便于某些框架 .forward 调用
    def forward(self, preds: Dict[str, torch.Tensor], xyz_bnc: torch.Tensor,
                batch_stats: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        return self.loss(preds, xyz_bnc, batch_stats=batch_stats)


def _getattr_safe(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_pinn(cfg: Any,
               *,
               norm_mode: Optional[str] = None,
               norm_cfg: Optional[Dict] = None,
               # 允许外部先给一个初始 λ，之后每个 epoch 用 update_lambdas 动态改
               lambda_cont_init: Optional[float] = None,
               lambda_mom_init: Optional[float] = None) -> TPACPINNAdapter:
    """
    工厂：从 cfg 读取超参并实例化 TPACPINNAdapter（内含 SteadyIncompressiblePINN）。

    兼容字段名：
      cfg.pinn.k / k_neighbors
      cfg.pinn.samples / m_samples
      cfg.pinn.lambda_mom
      cfg.pinn.rho / nu_eff
      cfg.mixed.use / cfg.mixed.wall_sigma
    """
    p = _getattr_safe(cfg, "pinn", None)
    m = _getattr_safe(cfg, "mixed", None)

    # 兼容字段名
    k_neighbors = _getattr_safe(p, "k", _getattr_safe(p, "k_neighbors", 16))
    m_samples   = _getattr_safe(p, "samples", _getattr_safe(p, "m_samples", 4096))
    lambda_mom  = _getattr_safe(p, "lambda_mom", 0.0 if p is None else 0.0)
    rho         = _getattr_safe(p, "rho", 1.0)
    nu_eff      = _getattr_safe(p, "nu_eff", 0.0)

    # 残差空间权重：与训练一致
    res_mode  = "mixed" if _getattr_safe(m, "use", False) else "uniform"
    res_alpha = None  # 由 Trainer 每个 epoch 计算 PINN 时注入更准确；这里先用 None
    res_sigma = _getattr_safe(m, "wall_sigma", 0.08)

    # 初始 λ（真正训练中建议每个 epoch 调整：adapter.update_lambdas(...)）
    lambda_cont = 0.0 if lambda_cont_init is None else float(lambda_cont_init)
    lambda_mom  = float(lambda_mom_init) if (lambda_mom_init is not None) else float(lambda_mom)

    adapter = TPACPINNAdapter(
        k_neighbors=int(k_neighbors),
        m_samples=int(m_samples),
        lambda_cont=lambda_cont,
        lambda_mom=lambda_mom,
        lambda_mom_floor=_getattr_safe(p, "lambda_mom_floor", 1e-4),
        vel_magnitude_weight=_getattr_safe(p, "vel_mag_weight", None),
        vel_gradient_weight=_getattr_safe(p, "vel_grad_weight", None),
        rho=float(rho),
        nu_eff=float(nu_eff),
        residual_weight_mode=res_mode,
        residual_alpha=res_alpha,
        residual_sigma=float(res_sigma),
        norm_mode=(norm_mode or "none"),
        norm_cfg=(norm_cfg or {}),
        ridge_init=_getattr_safe(p, "ridge_init", 1e-6),
        ridge_retry=_getattr_safe(p, "ridge_retry", 3),
    )
    return adapter
