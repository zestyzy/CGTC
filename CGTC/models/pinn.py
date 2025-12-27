# Steady incompressible PINN for point clouds (robust, with raw metrics, with in-PINN normalization handling)
from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

# ---------------------- helpers ----------------------
@torch.no_grad()
def _rand_subset_idx(N: int, m: int, device):
    m = int(min(max(1, m), N))
    return torch.randperm(N, device=device)[:m]

def _pairwise_cdist(x, y):
    return torch.cdist(x, y, p=2)

def _knn_indices(query_xyz, ref_xyz, k: int):
    """
    基于欧氏距离的近邻索引：
    - 下限 >= 6（稳住最小二乘）
    - 上限严格 < N（避免把“自身”当邻居）
    - N 很小时自动退化；N<=1 时返回占位索引
    """
    with torch.no_grad():
        d = _pairwise_cdist(query_xyz, ref_xyz)  # (B,M,N)
        N = int(ref_xyz.shape[1])
        if N <= 1:
            B, M, _ = d.shape
            return torch.zeros(B, M, 1, dtype=torch.long, device=d.device)
        k_lower = 6
        k_upper = max(1, N - 1)
        k_eff = min(k_upper, max(k_lower, int(k)))
        _, idx = torch.topk(d, k=k_eff, dim=-1, largest=False, sorted=False)
    return idx  # (B,M,k_eff)

def _gather_neighbors(t, idx):
    """
    t: (B,N) or (B,N,C), idx: (B,M,k)
    return: (B,M,k) or (B,M,k,C)
    """
    if t.dim() == 2:
        B, N = t.shape
        B2, M, k = idx.shape
        assert B == B2
        batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(-1, M, k)
        out = t[batch_idx, idx]
        return out
    elif t.dim() == 3:
        B, N, C = t.shape
        B2, M, k = idx.shape
        assert B == B2
        batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(-1, M, k)
        out = t[batch_idx, idx, :]
        return out
    else:
        raise ValueError("t must be (B,N) or (B,N,C)")

# ---------------------- robust solvers ----------------------
def _least_squares_grad(
    query_xyz, ref_xyz, f, idx, ridge: float = 1e-6, ridge_retry: int = 3
):
    """
    最小二乘估计 ∇f（batched），稳健流程：
      1) Cholesky 解 (X^T X + λI) g = X^T y，λ 自适应放大（最多 ridge_retry 次）
      2) 若仍失败，用 pinv 兜底
    注：对 y（即 f）可传梯度。
    """
    B, M, _ = query_xyz.shape
    device = query_xyz.device
    dtype  = query_xyz.dtype

    nbr_xyz = _gather_neighbors(ref_xyz, idx)                       # (B,M,k,3)
    nbr_f   = _gather_neighbors(f.unsqueeze(-1), idx).squeeze(-1)   # (B,M,k)

    # 中心化 + 极小扰动，避免精确共线
    q  = query_xyz.unsqueeze(2)                                     # (B,M,1,3)
    dq = (nbr_xyz - q)                                              # (B,M,k,3)
    dq = dq + 1e-8 * torch.randn_like(dq)

    # 用 ref 最近点值近似 f(q)
    with torch.no_grad():
        d_all  = _pairwise_cdist(query_xyz, ref_xyz) + 1e-12        # (B,M,N)
        minidx = torch.argmin(d_all, dim=-1)                        # (B,M)
    f_q = f.gather(1, minidx)                                       # (B,M)
    df  = nbr_f - f_q.unsqueeze(-1)                                 # (B,M,k)

    # 正规方程
    XT  = dq.transpose(-1, -2)                                      # (B,M,3,k)
    XTX = XT @ dq                                                   # (B,M,3,3)
    XTy = XT @ df.unsqueeze(-1)                                     # (B,M,3,1)

    # 自适应 ridge
    I = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3)
    base = (XTX.abs().mean(dim=(-1, -2), keepdim=True) + 1e-12)     # (B,M,1,1)

    ridge_list = [float(ridge)]
    for _ in range(max(0, int(ridge_retry))):
        ridge_list.append(ridge_list[-1] * 10.0)
    ridge_list = [max(1e-12, r) for r in ridge_list] + [1e-3]

    g = None
    solved = False
    for lam in ridge_list:
        A = (XTX.to(torch.float32) + (lam * base) * I)              # (B,M,3,3)
        L, info = torch.linalg.cholesky_ex(A)
        if (info == 0).all():
            g = torch.cholesky_solve(XTy.to(torch.float32), L).squeeze(-1)  # (B,M,3)
            solved = True
            break

    if not solved:
        A = (XTX.to(torch.float32) + (1e-3 * base) * I)
        A_pinv = torch.linalg.pinv(A)                               # (B,M,3,3)
        g = (A_pinv @ XTy.to(torch.float32)).squeeze(-1)            # (B,M,3)

    return g.to(dtype)

def _graph_laplacian(query_xyz, ref_xyz, f, idx, eps: float = 1e-8):
    """
    稳健图拉普拉斯 Δf：
      - 反距离平方权重
      - 对每个点局部权重做 99% 分位截断，抑制极端近邻主导
    """
    nbr_xyz = _gather_neighbors(ref_xyz, idx)                                # (B,M,k,3)
    d = torch.linalg.norm(nbr_xyz - query_xyz.unsqueeze(2), dim=-1)          # (B,M,k)
    w = 1.0 / (d * d + eps)                                                  # (B,M,k)
    # 分位截断（沿 k 维）
    w = torch.minimum(w, torch.quantile(w.float(), 0.99, dim=-1, keepdim=True))

    f_nbr = _gather_neighbors(f.unsqueeze(-1), idx).squeeze(-1)              # (B,M,k)

    # 用最近邻把中心值对齐（减少偏置）
    d_all  = _pairwise_cdist(query_xyz, ref_xyz) + eps
    minidx = torch.argmin(d_all, dim=-1)                                     # (B,M)
    f_q = f.gather(1, minidx).unsqueeze(-1)                                   # (B,M,1)

    num = (w * (f_nbr - f_q)).sum(dim=-1)                                     # (B,M)
    den = w.sum(dim=-1).clamp_min(eps)                                        # (B,M)
    return num / den

# ---------- spatial weighting for PINN residuals ----------
def _distance_to_wall(xyz_bn3):
    """
    xyz in [0,1]^3, return d = min{x,1-x,y,1-y,z,1-z} as (B,1,N)
    """
    x = xyz_bn3[..., 0:1]; y = xyz_bn3[..., 1:2]; z = xyz_bn3[..., 2:3]
    d = torch.minimum(torch.minimum(x, 1 - x),
        torch.minimum(torch.minimum(y, 1 - y), torch.minimum(z, 1 - z)))
    return d.transpose(1, 2)  # (B,1,N)

def _make_pinn_weights(xyz_bn3, mode="uniform", sigma=0.08, alpha=None, eps=1e-8):
    """
    仅作用于 PINN 残差的空间权重：
      uniform: 1
      wall   : 1 + exp(-d/sigma)
      mixed  : alpha * (1 + exp(-d/sigma)) + (1-alpha) * 1
    输出两份：(B,1,N) for continuity, (B,1,N) for momentum（当前相同，可分开定制）
    """
    d = _distance_to_wall(xyz_bn3)  # (B,1,N)
    if mode == "uniform":
        w = torch.ones_like(d)
    elif mode == "wall":
        w = 1.0 + torch.exp(-d / max(sigma, eps))
    elif mode == "mixed":
        a = float(alpha if alpha is not None else 1.0)
        a = max(0.0, min(1.0, a))
        w_wall = 1.0 + torch.exp(-d / max(sigma, eps))
        w = a * w_wall + (1.0 - a) * 1.0
    else:
        w = torch.ones_like(d)
    # 归一化到均值=1
    w = w / w.mean(dim=2, keepdim=True).clamp_min(eps)
    return w, w

# ---------------------- PINN ----------------------
class SteadyIncompressiblePINN(nn.Module):
    """
    Steady incompressible PINN for point clouds:
        continuity:  div(u) = 0
        momentum:    (u · ∇)u + (1/ρ)∇p - νΔu = 0

    新增：
      - norm_mode: "none" | "denorm_physical" | "nondim"
      - norm_cfg : dict，可包含：
           x_min/x_max: [3], y_min/y_max: [4]     # 用于 min-max 反归一化
           L_scale/U_scale/P_scale                # 直接给尺度（nondim 更方便）
           rho/nu_eff                             # 物理常数（覆盖构造参数）
      - lambda_mom_floor / vel_mag_weight / vel_grad_weight:
            PINN 动量支路的最小权重，以及对速度幅值/梯度的直接惩罚项权重
    """
    def __init__(self, k_neighbors=16, m_samples=4096,
                 lambda_cont=1.0, lambda_mom=0.0,
                 lambda_mom_floor: float = 1e-4,
                 vel_magnitude_weight: Optional[float] = None,
                 vel_gradient_weight: Optional[float] = None,
                 rho=1.0, nu_eff=0.0,
                 # 残差空间权重
                 residual_weight_mode: str = "uniform",   # 'uniform' | 'wall' | 'mixed'
                 residual_alpha: Optional[float] = None,  # for 'mixed'
                 residual_sigma: float = 0.08,
                 # 数值稳健性
                 ridge_init: float = 1e-6,
                 ridge_retry: int = 3,
                 # ===== 新增：尺度处理 =====
                 norm_mode: str = "none",                 # 'none' | 'denorm_physical' | 'nondim'
                 norm_cfg: Optional[Dict] = None):
        super().__init__()
        self.k = max(int(k_neighbors), 6)
        self.m = int(m_samples)
        self.lambda_cont = float(lambda_cont)
        self.lambda_mom  = float(lambda_mom)
        self.lambda_mom_floor = max(0.0, float(lambda_mom_floor))

        if vel_magnitude_weight is None:
            vel_magnitude_weight = 0.0
        if vel_gradient_weight is None:
            vel_gradient_weight = 0.0
        self.vel_mag_weight = max(0.0, float(vel_magnitude_weight))
        self.vel_grad_weight = max(0.0, float(vel_gradient_weight))

        # 默认常数（若 norm_mode 不改写，则用它们）
        self.rho = float(rho)
        self.nu  = float(nu_eff)

        self.res_mode  = residual_weight_mode
        self.res_alpha = residual_alpha
        self.res_sigma = residual_sigma

        self.ridge_init  = float(ridge_init)
        self.ridge_retry = int(max(0, ridge_retry))

        # 尺度与模式
        self.norm_mode = (norm_mode or "none").lower()
        self.norm_cfg  = norm_cfg or {}

        # 把可能提供的常数覆盖掉
        self.rho_phys = float(self.norm_cfg.get("rho", self.rho))
        self.nu_phys  = float(self.norm_cfg.get("nu_eff", self.nu))

        # 缓存 min/max / 尺度
        def _to_tensor(x):
            if x is None: return None
            t = torch.as_tensor(x, dtype=torch.float32)
            return t
        self.x_min = _to_tensor(self.norm_cfg.get("x_min", None))  # (3,)
        self.x_max = _to_tensor(self.norm_cfg.get("x_max", None))
        self.y_min = _to_tensor(self.norm_cfg.get("y_min", None))  # (4,)
        self.y_max = _to_tensor(self.norm_cfg.get("y_max", None))

        self.L_scale = self.norm_cfg.get("L_scale", None)
        self.U_scale = self.norm_cfg.get("U_scale", None)
        self.P_scale = self.norm_cfg.get("P_scale", None)
        self.P_ref   = self.norm_cfg.get("p_ref", None)

    # ---------- 工具：解析尺度 ----------
    def _resolve_scales(self, *,
                         B: int,
                         device: torch.device,
                         x_min: Optional[torch.Tensor] = None,
                         x_max: Optional[torch.Tensor] = None,
                         y_min: Optional[torch.Tensor] = None,
                         y_max: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回逐 batch 的 (L_scale, U_scale, P_scale)。优先使用逐样本统计，
        其次使用构造时提供的全局尺度，最后退化为常数 1。
        """

        def _fallback_tensor(val: Optional[float], default: float) -> torch.Tensor:
            base = float(val) if val is not None else float(default)
            return torch.full((B,), base, device=device, dtype=torch.float32)

        L_batch: Optional[torch.Tensor] = None
        if (x_min is not None) and (x_max is not None):
            rng = (x_max - x_min).abs()
            L_batch = rng.amax(dim=1)
            L_batch = L_batch.clamp_min(1e-6)
        elif self.x_min is not None and self.x_max is not None:
            rng = (self.x_max - self.x_min).abs()
            L_batch = torch.full((B,), float(torch.max(rng).item()), device=device, dtype=torch.float32)
        elif self.L_scale is not None:
            L_batch = _fallback_tensor(self.L_scale, 1.0)

        U_batch: Optional[torch.Tensor] = None
        if (y_min is not None) and (y_max is not None):
            rng = (y_max[:, 1:4] - y_min[:, 1:4]).abs()
            U_batch = rng.amax(dim=1)
            U_batch = U_batch.clamp_min(1e-6)
        elif self.y_min is not None and self.y_max is not None:
            rng = (self.y_max[1:4] - self.y_min[1:4]).abs()
            U_batch = torch.full((B,), float(torch.max(rng).item()), device=device, dtype=torch.float32)
        elif self.U_scale is not None:
            U_batch = _fallback_tensor(self.U_scale, 1.0)

        if L_batch is None:
            L_batch = _fallback_tensor(self.L_scale, 1.0)
        if U_batch is None:
            U_batch = _fallback_tensor(self.U_scale, 1.0)

        if self.P_scale is not None:
            P_batch = _fallback_tensor(self.P_scale, float(self.rho_phys))
        else:
            P_batch = (self.rho_phys * (U_batch ** 2)).clamp_min(1e-6)

        return L_batch, U_batch, P_batch

    def _resolve_p_ref(self, *,
                       B: int,
                       device: torch.device,
                       y_min: Optional[torch.Tensor] = None,
                       y_max: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.P_ref is not None:
            return torch.full((B,), float(self.P_ref), device=device, dtype=torch.float32)
        if (y_min is not None) and (y_max is not None):
            return 0.5 * (y_min[:, 0] + y_max[:, 0])
        if self.y_min is not None and self.y_max is not None:
            return torch.full((B,), float(0.5 * (self.y_min[0] + self.y_max[0])), device=device, dtype=torch.float32)
        return torch.zeros(B, device=device, dtype=torch.float32)

    # ---------- 反归一化 ----------
    def _expand_stat(self, value, fallback, batch: int, ch: int, device: torch.device):
        base = value if value is not None else fallback
        if base is None:
            return None
        t = torch.as_tensor(base, dtype=torch.float32, device=device)
        if t.dim() == 1:
            if t.numel() != ch:
                raise ValueError(f"stat expected len {ch}, got {t.numel()}")
            return t.view(1, ch).expand(batch, ch).contiguous()
        if t.dim() == 2:
            if t.size(1) != ch:
                raise ValueError(f"stat expected second dim {ch}, got {t.size(1)}")
            if t.size(0) == 1:
                return t.expand(batch, ch).contiguous()
            if t.size(0) != batch:
                raise ValueError(f"stat batch mismatch: expect {batch}, got {t.size(0)}")
            return t.contiguous()
        raise ValueError("stat tensor must be 1D or 2D")

    def _resolve_minmax(self, batch_stats, B: int,
                         device_x: torch.device, device_y: torch.device):
        x_min_o = x_max_o = y_min_o = y_max_o = None
        if batch_stats is not None:
            if isinstance(batch_stats, dict):
                x_min_o = batch_stats.get("x_min")
                x_max_o = batch_stats.get("x_max")
                y_min_o = batch_stats.get("y_min")
                y_max_o = batch_stats.get("y_max")
            elif isinstance(batch_stats, (list, tuple)) and len(batch_stats) >= 4:
                x_min_o, x_max_o, y_min_o, y_max_o = batch_stats[:4]

        x_min = self._expand_stat(x_min_o, self.x_min, B, 3, device_x)
        x_max = self._expand_stat(x_max_o, self.x_max, B, 3, device_x)
        y_min = self._expand_stat(y_min_o, self.y_min, B, 4, device_y)
        y_max = self._expand_stat(y_max_o, self.y_max, B, 4, device_y)
        return x_min, x_max, y_min, y_max

    def _denorm_minmax(self, X_b3n: torch.Tensor, Y_b4n: torch.Tensor, batch_stats=None):
        """
        使用 min-max 对 (X,Y) 做反归一化到物理单位：
          X_phys = x_min + X_norm * (x_max - x_min)
          Y_phys = y_min + Y_norm * (y_max - y_min)
        期望 X 形状 (B,3,N), Y (B,4,N)
        """
        B = X_b3n.shape[0]
        x_min, x_max, y_min, y_max = self._resolve_minmax(batch_stats, B, X_b3n.device, Y_b4n.device)
        if (x_min is None) or (x_max is None) or (y_min is None) or (y_max is None):
            raise ValueError("[PINN] denorm_physical 需要提供逐样本或全局的 min/max。")

        try:
            x_min = x_min.view(B, 3, 1)
            x_max = x_max.view(B, 3, 1)
        except RuntimeError as err:
            raise RuntimeError(
                f"[PINN] x_min/x_max 形状异常：x_min={tuple(x_min.shape)}, x_max={tuple(x_max.shape)}, batch={B}"
            ) from err
        try:
            y_min = y_min.view(B, 4, 1)
            y_max = y_max.view(B, 4, 1)
        except RuntimeError as err:
            raise RuntimeError(
                f"[PINN] y_min/y_max 形状异常：y_min={tuple(y_min.shape)}, y_max={tuple(y_max.shape)}, batch={B}"
            ) from err

        x_rng = (x_max - x_min).clamp_min(1e-12)
        y_rng = (y_max - y_min).clamp_min(1e-12)

        X_phys = x_min + X_b3n * x_rng
        Y_phys = y_min + Y_b4n * y_rng
        return X_phys, Y_phys, (
            x_min.view(B, 3), x_max.view(B, 3),
            y_min.view(B, 4), y_max.view(B, 4)
        )

    # ---------- forward ----------
    def forward(self, xyz, y_pred, batch_stats=None) -> Dict[str, torch.Tensor]:
        """
        xyz:    (B,3,N)  —— 传入的是“归一化”坐标
        y_pred: (B,4,N)  —— 传入的是“归一化/网络输出尺度”的 [p,u,v,w]

        根据 norm_mode 选择：
          - none:            沿用原实现
          - denorm_physical: 反归一化到物理单位，用 rho_phys/nu_phys 计算残差
          - nondim:          直接在归一化变量上算，但常数改 rho*=1, nu*=nu/(U*L)
        """
        device = xyz.device
        dtype  = xyz.dtype
        B, _, N = xyz.shape
        assert y_pred.shape[:2] == (B, 4), "y_pred must be (B,4,N) with [p,u,v,w]"

        # 关闭时直接返回 0
        lambda_mom_eff = max(self.lambda_mom, self.lambda_mom_floor)
        mom_branch_active = (lambda_mom_eff > 0.0) or (self.vel_mag_weight > 0.0) or (self.vel_grad_weight > 0.0)

        if (self.lambda_cont <= 0.0) and (not mom_branch_active):
            z = torch.zeros((), device=device, dtype=dtype)
            return {
                "loss_pinn": z, "loss_cont": z, "loss_mom": z,
                "cont_raw_mean": z, "mom_raw_mean": z,
                "mom_mag_mean": z, "mom_grad_mean": z
            }

        # === 1) 准备权重坐标（固定使用归一化坐标估墙距） ===
        xyz_for_weight = xyz.transpose(1, 2).contiguous()  # (B,N,3)

        # === 2) 准备用于 KNN/梯度/拉普拉斯 的坐标 & y（可能做尺度变换）===
        mode = self.norm_mode
        if mode == "denorm_physical":
            # 物理坐标/物理变量 -> 转换到无量纲尺度计算残差
            X_phys, Y_phys, stats_used = self._denorm_minmax(xyz, y_pred, batch_stats=batch_stats)
            x_min_t, x_max_t, y_min_t, y_max_t = stats_used

            xyz_bn3 = xyz.transpose(1, 2).contiguous()               # (B,N,3) —— 原始归一化坐标

            L_scale, U_scale, P_scale = self._resolve_scales(
                B=B, device=device, x_min=x_min_t, x_max=x_max_t, y_min=y_min_t, y_max=y_max_t
            )
            p_ref = self._resolve_p_ref(B=B, device=device, y_min=y_min_t, y_max=y_max_t)

            L_scale = L_scale.clamp_min(1e-6).to(dtype=dtype)
            U_scale = U_scale.clamp_min(1e-6).to(dtype=dtype)
            P_scale = P_scale.clamp_min(1e-6).to(dtype=dtype)
            p_ref   = p_ref.to(dtype=dtype)

            p = (Y_phys[:, 0, :] - p_ref.unsqueeze(1)) / P_scale.unsqueeze(1)
            u = Y_phys[:, 1, :] / U_scale.unsqueeze(1)
            v = Y_phys[:, 2, :] / U_scale.unsqueeze(1)
            w = Y_phys[:, 3, :] / U_scale.unsqueeze(1)

            rho_use = torch.ones(B, device=device, dtype=dtype)
            nu_base = torch.full((B,), float(self.nu_phys), device=device, dtype=dtype)
            nu_use  = nu_base / (U_scale * L_scale).clamp_min(1e-6)
        elif mode == "nondim":
            # 无量纲：直接在归一化变量上计算，常数转换为 ρ*=1, ν*=ν/(U*L)
            xyz_bn3 = xyz.transpose(1, 2).contiguous()               # (B,N,3)

            L_scale, U_scale, P_scale = self._resolve_scales(B=B, device=device)
            L_scale = L_scale.clamp_min(1e-6).to(dtype=dtype)
            U_scale = U_scale.clamp_min(1e-6).to(dtype=dtype)
            P_scale = P_scale.clamp_min(1e-6).to(dtype=dtype)

            p = y_pred[:, 0, :] / P_scale.unsqueeze(1)
            u = y_pred[:, 1, :] / U_scale.unsqueeze(1)
            v = y_pred[:, 2, :] / U_scale.unsqueeze(1)
            w = y_pred[:, 3, :] / U_scale.unsqueeze(1)

            rho_use = torch.ones(B, device=device, dtype=dtype)
            nu_base = torch.full((B,), float(self.nu_phys), device=device, dtype=dtype)
            nu_use  = nu_base / (U_scale * L_scale).clamp_min(1e-6)
        else:
            # none：保持原实现（即在“网络输出尺度”上算）
            xyz_bn3 = xyz.transpose(1, 2).contiguous()               # (B,N,3)
            p = y_pred[:, 0, :]; u = y_pred[:, 1, :]; v = y_pred[:, 2, :]; w = y_pred[:, 3, :]
            rho_use = torch.full((B,), float(self.rho), device=device, dtype=dtype)
            nu_use  = torch.full((B,), float(self.nu), device=device, dtype=dtype)

        U = torch.stack([u, v, w], dim=-1)                           # (B,N,3)

        # PINN 残差空间权重（基于归一化坐标）
        w_cont_full, w_mom_full = _make_pinn_weights(
            xyz_for_weight, mode=self.res_mode, sigma=self.res_sigma, alpha=self.res_alpha
        )  # (B,1,N), (B,1,N)

        cont_losses = []
        mom_losses  = []
        cont_raw_list = []
        mom_raw_list  = []
        mom_mag_list  = []
        mom_grad_list = []
        mom_mag_raw_list  = []
        mom_grad_raw_list = []

        for b in range(B):
            Nb = xyz_bn3.shape[1]
            m  = min(self.m, Nb)
            if m <= 0:
                continue

            I = _rand_subset_idx(Nb, m, device)
            qxyz = xyz_bn3[b:b+1, I, :]             # (1,m,3)
            rxyz = xyz_bn3[b:b+1, :, :]             # (1,Nb,3)

            idx_knn = _knn_indices(qxyz, rxyz, self.k)  # (1,m,k)

            # === gradients (robust) ===
            gu = _least_squares_grad(qxyz, rxyz, u[b:b+1], idx_knn,
                                     ridge=self.ridge_init, ridge_retry=self.ridge_retry)  # (1,m,3)
            gv = _least_squares_grad(qxyz, rxyz, v[b:b+1], idx_knn,
                                     ridge=self.ridge_init, ridge_retry=self.ridge_retry)
            gw = _least_squares_grad(qxyz, rxyz, w[b:b+1], idx_knn,
                                     ridge=self.ridge_init, ridge_retry=self.ridge_retry)

            # ---------- continuity: div(u) ----------
            div = gu[..., 0] + gv[..., 1] + gw[..., 2]          # (1,m)
            wc  = w_cont_full[b:b+1, :, I].squeeze(1)           # (1,m)->(m)

            # raw（未裁剪、未加权）
            div2_raw = (div * div).float()                      # (1,m)
            cont_raw_mean = div2_raw.mean()                     # 标量

            # 裁剪 + 加权（稳健）
            q99  = torch.quantile(div2_raw, 0.99, dim=-1, keepdim=True)  # (1,1)
            div2_clip = torch.minimum(div2_raw, q99)
            loss_cont = (div2_clip * wc.float()).mean()
            if not torch.isfinite(loss_cont):
                loss_cont = torch.zeros((), device=device, dtype=dtype)

            cont_losses.append(loss_cont.to(dtype))
            cont_raw_list.append(cont_raw_mean.to(dtype))

            # ---------- momentum（可选） ----------
            if mom_branch_active:
                gradU = torch.stack([gu, gv, gw], dim=2).squeeze(0)  # (m,3,3)
                U_q   = U[b, I, :].unsqueeze(1)                       # (m,1,3)
                adv   = torch.matmul(U_q, gradU).squeeze(1)           # (m,3)

                gp   = _least_squares_grad(qxyz, rxyz, p[b:b+1], idx_knn,
                                            ridge=self.ridge_init, ridge_retry=self.ridge_retry).squeeze(0)  # (m,3)
                rho_b = float(rho_use[b]) if torch.is_tensor(rho_use) else float(rho_use)
                pres = (1.0 / max(rho_b, 1e-12)) * gp

                lap_u = _graph_laplacian(qxyz, rxyz, u[b:b+1], idx_knn).squeeze(0)
                lap_v = _graph_laplacian(qxyz, rxyz, v[b:b+1], idx_knn).squeeze(0)
                lap_w = _graph_laplacian(qxyz, rxyz, w[b:b+1], idx_knn).squeeze(0)
                nu_b = float(nu_use[b]) if torch.is_tensor(nu_use) else float(nu_use)
                visc  = nu_b * torch.stack([lap_u, lap_v, lap_w], dim=-1)         # (m,3)

                mom_res  = adv + pres - visc                                         # (m,3)
                wm = w_mom_full[b:b+1, :, I].squeeze(1)                               # (1,m)->(m)

                mom_res_sq = (mom_res ** 2).sum(dim=-1).float()
                mom_raw_mean = mom_res_sq.mean()
                loss_mom = (mom_res_sq * wm.float()).mean()

                vel_mag_mean = torch.zeros((), device=device, dtype=dtype)
                vel_grad_mean = torch.zeros((), device=device, dtype=dtype)
                vel_mag_loss = torch.zeros((), device=device, dtype=dtype)
                vel_grad_loss = torch.zeros((), device=device, dtype=dtype)

                if self.vel_mag_weight > 0.0:
                    speed_sq = (U[b, I, :] ** 2).sum(dim=-1).float()
                    vel_mag_mean = speed_sq.mean()
                    vel_mag_loss = (speed_sq * wm.float()).mean()

                if self.vel_grad_weight > 0.0:
                    grad_norm_sq = (gradU.float() ** 2).sum(dim=(-1, -2))
                    vel_grad_mean = grad_norm_sq.mean()
                    vel_grad_loss = (grad_norm_sq * wm.float()).mean()

                if not torch.isfinite(loss_mom):
                    loss_mom = torch.zeros((), device=device, dtype=dtype)
                if not torch.isfinite(mom_raw_mean):
                    mom_raw_mean = torch.zeros((), device=device, dtype=dtype)
                if not torch.isfinite(vel_mag_loss):
                    vel_mag_loss = torch.zeros((), device=device, dtype=dtype)
                if not torch.isfinite(vel_grad_loss):
                    vel_grad_loss = torch.zeros((), device=device, dtype=dtype)
                if not torch.isfinite(vel_mag_mean):
                    vel_mag_mean = torch.zeros((), device=device, dtype=dtype)
                if not torch.isfinite(vel_grad_mean):
                    vel_grad_mean = torch.zeros((), device=device, dtype=dtype)
            else:
                loss_mom     = torch.zeros((), device=device, dtype=dtype)
                mom_raw_mean = torch.zeros((), device=device, dtype=dtype)
                vel_mag_loss = torch.zeros((), device=device, dtype=dtype)
                vel_grad_loss = torch.zeros((), device=device, dtype=dtype)
                vel_mag_mean = torch.zeros((), device=device, dtype=dtype)
                vel_grad_mean = torch.zeros((), device=device, dtype=dtype)

            mom_losses.append(loss_mom.to(dtype))
            mom_raw_list.append(mom_raw_mean.to(dtype))
            mom_mag_list.append(vel_mag_loss.to(dtype))
            mom_grad_list.append(vel_grad_loss.to(dtype))
            mom_mag_raw_list.append(vel_mag_mean.to(dtype))
            mom_grad_raw_list.append(vel_grad_mean.to(dtype))

        mom_mag_loss = torch.stack(mom_mag_list).mean() if mom_mag_list else torch.zeros((), device=device, dtype=dtype)
        mom_grad_loss = torch.stack(mom_grad_list).mean() if mom_grad_list else torch.zeros((), device=device, dtype=dtype)
        mom_mag_raw = torch.stack(mom_mag_raw_list).mean() if mom_mag_raw_list else torch.zeros((), device=device, dtype=dtype)
        mom_grad_raw = torch.stack(mom_grad_raw_list).mean() if mom_grad_raw_list else torch.zeros((), device=device, dtype=dtype)

        # 汇总
        loss_cont = torch.stack(cont_losses).mean() if cont_losses else torch.zeros((), device=device, dtype=dtype)
        loss_mom  = torch.stack(mom_losses ).mean() if mom_losses  else torch.zeros((), device=device, dtype=dtype)

        cont_raw  = torch.stack(cont_raw_list).mean() if cont_raw_list else torch.zeros((), device=device, dtype=dtype)
        mom_raw   = torch.stack(mom_raw_list ).mean() if mom_raw_list  else torch.zeros((), device=device, dtype=dtype)

        loss_pinn = self.lambda_cont * loss_cont
        if mom_branch_active:
            if lambda_mom_eff > 0.0:
                loss_pinn = loss_pinn + lambda_mom_eff * loss_mom
            if self.vel_mag_weight > 0.0:
                loss_pinn = loss_pinn + self.vel_mag_weight * mom_mag_loss
            if self.vel_grad_weight > 0.0:
                loss_pinn = loss_pinn + self.vel_grad_weight * mom_grad_loss
        if not torch.isfinite(loss_pinn):
            loss_pinn = torch.zeros((), device=device, dtype=dtype)

        return {
            "loss_pinn": loss_pinn,       # 训练用：连续性 + 动量残差 + 幅值/梯度正则
            "loss_cont": loss_cont,       # 评估用：裁剪+加权（未乘 λ）
            "loss_mom" : loss_mom,        # 评估用：加权（未乘 λ）
            "cont_raw_mean": cont_raw,    # 原始 E[div^2]（未裁剪、未加权、未乘 λ）
            "mom_raw_mean" : mom_raw,     # 原始 ‖mom_res‖²（未加权、未乘 λ）
            "mom_mag_mean": mom_mag_raw.detach() if mom_branch_active else torch.zeros((), device=device, dtype=dtype),
            "mom_grad_mean": mom_grad_raw.detach() if mom_branch_active else torch.zeros((), device=device, dtype=dtype),
        }
