# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Dict

# PointNet++ 回归器
try:  # pragma: no cover - prefer absolute import when package is installed
    from Rotate.models.pointnet2pp import PointNet2Regressor as _PN2Reg
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    from models.pointnet2pp import PointNet2Regressor as _PN2Reg

# PointTransformer 回归器
try:  # pragma: no cover
    from Rotate.models.pointTransformer import (
        PointTransformerRegressor as _PTReg,
    )
except ModuleNotFoundError:  # pragma: no cover
    try:
        from models.pointTransformer import (
            PointTransformerRegressor as _PTReg,
        )
    except ModuleNotFoundError:
        _PTReg = None

__all__ = ["PointNet2Adapter", "PointTransformerAdapter", "build_backbone"]


class PointNet2Adapter(nn.Module):
    """
    适配器：把 TPAC 统一输入 (B,N,3)/(B,N,C0) 转成 PointNet2 的 (B,3,N)/(B,C0,N)
    输出改为 dict: {"p","u","v","w"}，每个张量形状 (B,N)
    —— 仅做维度转置与通道拆分，不改变任何训练策略/结构。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.net = _PN2Reg(**kwargs)

    def forward(
        self,
        xyz_bnc: torch.Tensor,
        feat0_bnc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        xyz_bnc:   (B,N,3) —— TPAC 统一输入
        feat0_bnc: (B,N,C0) —— 可选
        返回: dict{"p","u","v","w"}，每个 (B,N)
        """
        assert xyz_bnc.dim() == 3 and xyz_bnc.size(-1) == 3, \
            f"xyz must be (B,N,3), got {tuple(xyz_bnc.shape)}"

        # 转成模型输入格式 (B,3,N) / (B,C0,N)
        xyz_bcn = xyz_bnc.transpose(1, 2).contiguous()
        feat0 = None
        if feat0_bnc is not None:
            assert (
                feat0_bnc.dim() == 3
                and feat0_bnc.size(0) == xyz_bnc.size(0)
                and feat0_bnc.size(1) == xyz_bnc.size(1)
            ), (
                "feat0 must align with xyz on (B,N,*), "
                f"got {tuple(feat0_bnc.shape)} vs {tuple(xyz_bnc.shape)}"
            )
            feat0 = feat0_bnc.transpose(1, 2).contiguous()

        # 模型输出: (B,4,N)，通道顺序 [p,u,v,w]
        out_b4n = self.net(xyz_bcn, feat0)  # (B,4,N)
        assert out_b4n.dim() == 3 and out_b4n.size(1) == 4, \
            f"backbone must output (B,4,N), got {tuple(out_b4n.shape)}"

        p = out_b4n[:, 0, :]  # (B,N)
        u = out_b4n[:, 1, :]
        v = out_b4n[:, 2, :]
        w = out_b4n[:, 3, :]

        return {
            "p": p.contiguous(),
            "u": u.contiguous(),
            "v": v.contiguous(),
            "w": w.contiguous(),
        }


class PointTransformerAdapter(nn.Module):
    """
    PointTransformer 适配器：
      - 输入仍然使用 TPAC 统一格式 (B,N,3)/(B,N,C0)
      - PointTransformerRegressor 接口与 PointNet2Regressor 一致：
          输入 (B,3,N)/(B,C0,N)，输出 (B,4,N) 通道顺序 [p,u,v,w]
      - 只做维度转置与通道拆分，不改动训练逻辑。
    """
    def __init__(self, **kwargs):
        super().__init__()
        if _PTReg is None:
            raise ImportError(
                "PointTransformerRegressor is not available. "
                "Please ensure `tpac_pointTransformer.py` is in "
                "Rotate.models or models and defines PointTransformerRegressor."
            )
        self.net = _PTReg(**kwargs)

    def forward(
        self,
        xyz_bnc: torch.Tensor,
        feat0_bnc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        xyz_bnc:   (B,N,3)
        feat0_bnc: (B,N,C0) —— 可选
        返回: dict{"p","u","v","w"}，每个 (B,N)
        """
        assert xyz_bnc.dim() == 3 and xyz_bnc.size(-1) == 3, \
            f"xyz must be (B,N,3), got {tuple(xyz_bnc.shape)}"

        # 转成 PointTransformer 接口格式 (B,3,N)/(B,C0,N)
        xyz_bcn = xyz_bnc.transpose(1, 2).contiguous()
        feat0 = None
        if feat0_bnc is not None:
            assert (
                feat0_bnc.dim() == 3
                and feat0_bnc.size(0) == xyz_bnc.size(0)
                and feat0_bnc.size(1) == xyz_bnc.size(1)
            ), (
                "feat0 must align with xyz on (B,N,*), "
                f"got {tuple(feat0_bnc.shape)} vs {tuple(xyz_bnc.shape)}"
            )
            feat0 = feat0_bnc.transpose(1, 2).contiguous()

        out_b4n = self.net(xyz_bcn, feat0)  # 期望 (B,4,N)
        assert out_b4n.dim() == 3 and out_b4n.size(1) == 4, \
            f"backbone must output (B,4,N), got {tuple(out_b4n.shape)}"

        p = out_b4n[:, 0, :]
        u = out_b4n[:, 1, :]
        v = out_b4n[:, 2, :]
        w = out_b4n[:, 3, :]

        return {
            "p": p.contiguous(),
            "u": u.contiguous(),
            "v": v.contiguous(),
            "w": w.contiguous(),
        }


def build_backbone(cfg):
    """
    统一工厂。配置里写:
      models.backbone.name: pointnet2pp_ssg / pointtransformer
      models.backbone.args: {... 原始超参 ...}

    返回：
      - PointNet2Adapter 或 PointTransformerAdapter
      - 接口统一为：
          forward(xyz_bnc, feat0_bnc=None) -> dict{"p","u","v","w"}
    """
    name = getattr(getattr(cfg, "models", None), "backbone", None)
    if name is None:
        backbone_name = "pointnet2pp_ssg"
        args = {}
    else:
        backbone_name = getattr(
            cfg.models.backbone, "name", "pointnet2pp_ssg"
        ).lower()
        args = dict(getattr(cfg.models.backbone, "args", {}))

    # ---------------- PointNet++ SSG ----------------
    if backbone_name in ("pointnet2pp_ssg", "pn2", "pointnet2"):
        # 与 tpac_pointnet2pp.get_model 的默认值保持一致
        defaults = dict(
            out_channels=4,
            p_drop=0.3,
            npoint1=2048,
            npoint2=512,
            npoint3=128,
            k1=16,
            k2=16,
            k3=16,
            c1=128,
            c2=256,
            c3=512,
            groups_gn=32,
            use_pe=False,
            use_msg=False,
            in_ch0=0,
            p_head_ch=(128, 64),
            uvw_head_ch=(128, 64),
            use_graph_refine=True,
            refine_k=16,
            refine_layers=2,
            refine_hidden=128,
            refine_residual=True,
        )
        defaults.update(args)
        return PointNet2Adapter(**defaults)

    # ---------------- PointTransformer ----------------
    if backbone_name in ("pointtransformer", "point_transformer", "pt", "pt_reg"):
        # 与刚刚实现的 PointTransformerRegressor 构造函数保持一致
        defaults = dict(
            out_channels=4,
            p_drop=0.3,
            dim=128,
            depth=4,
            k_neighbors=16,
            groups_gn=32,
            in_ch0=0,
            p_head_ch=(128, 64),
            uvw_head_ch=(128, 64),
            use_graph_refine=True,
            refine_k=16,
            refine_layers=2,
            refine_hidden=128,
            refine_residual=True,
        )
        defaults.update(args)
        return PointTransformerAdapter(**defaults)

    raise ValueError(f"Unknown backbone: {backbone_name}")
