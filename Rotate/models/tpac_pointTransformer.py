# tpac_pointtransformer.py
# PointTransformer with dual-head [p] & [u,v,w] (+ 可选动态图卷积细化)

import torch
import torch.nn as nn

# ----------------- public factory (保持兼容) -----------------
def get_model(**kwargs):
    """
    兼容调用方式：model = pointtransformer.get_model()

    关键超参：
      - dim:       主特征维度（默认 128）
      - depth:     PointTransformer 层数（默认 4）
      - k_neighbors: 每层局部注意力的 kNN K（默认 16）
      - use_graph_refine: 是否启用 EdgeConv 图细化（默认 True）
      - refine_k / refine_layers / refine_hidden / refine_residual:
          同 PointNet2Regressor 中 GraphRefiner 的超参

    其余参数见 PointTransformerRegressor 注释。
    """
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
        # 图细化模块（可选，接口与 PointNet2Regressor 保持一致）
        use_graph_refine=True,
        refine_k=16,
        refine_layers=2,
        refine_hidden=128,
        refine_residual=True,
    )
    defaults.update(kwargs)
    return PointTransformerRegressor(**defaults)


# ----------------- small utils -----------------
def gn(ch, groups=32):
    """GroupNorm with fallback to a single group when not divisible."""
    assert ch > 0
    g = min(groups, ch)
    g = g if ch % g == 0 else 1
    return nn.GroupNorm(g, ch)


def knn_idx_full(feat, k: int):
    """
    基于特征空间 L2 距离构建 kNN 索引。
    feat: (B,C,N)  -> return indices (B,N,k)
    这里可以传坐标 (B,3,N)，也可以传高维特征。
    """
    B, C, N = feat.shape
    k = min(k, N)
    # pairwise distance: (B,N,N) = ||x_i - x_j||^2
    xx = (feat ** 2).sum(dim=1, keepdim=False)  # (B,N)
    dist = xx.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * feat.transpose(1, 2) @ feat  # (B,N,N)
    dist = dist.clamp_min(0)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return idx  # (B,N,k)


def gather_by_idx(feat, idx):
    """
    feat: (B,C,N), idx: (B,N,k) -> out: (B,C,N,k)
    """
    B, C, N = feat.shape
    _, N2, K = idx.shape
    assert N2 == N
    batch = torch.arange(B, device=feat.device).view(B, 1, 1).expand(B, N, K)
    nbr = feat.transpose(1, 2)[batch, idx, :]   # (B,N,k,C)
    return nbr.permute(0, 3, 1, 2).contiguous()  # (B,C,N,k)


# ----------------- Dynamic EdgeConv (DGCNN-style) -----------------
class EdgeConvBlock(nn.Module):
    """
    One EdgeConv block:
      - build kNN on input features
      - edge feature: phi([x_i, x_j - x_i])
      - aggregation: max over neighbors
      - GN + ReLU
      - residual (optional)
    """
    def __init__(self, in_ch, out_ch, k=16, groups_gn=32, residual=True):
        super().__init__()
        self.k = int(max(1, k))
        self.residual = residual and (in_ch == out_ch)
        self.theta = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            gn(out_ch, groups_gn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            gn(out_ch, groups_gn),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B,C,N)
        return: (B,C_out,N)
        """
        B, C, N = x.shape
        idx = knn_idx_full(x, self.k)                     # (B,N,k)
        xj = gather_by_idx(x, idx)                        # (B,C,N,k)
        xi = x.unsqueeze(-1).expand(-1, -1, -1, self.k)   # (B,C,N,k)
        edge = torch.cat([xi, xj - xi], dim=1)            # (B,2C,N,k)
        y = self.theta(edge).amax(dim=-1)                 # (B,C_out,N)
        if self.residual:
            y = y + x
        return y


class GraphRefiner(nn.Module):
    """
    Stack of EdgeConv blocks that refines per-point features.
    """
    def __init__(self, in_ch=128, hidden=128, layers=2, k=16, groups_gn=32, residual=True):
        super().__init__()
        blocks = []
        ch = in_ch
        for _ in range(layers):
            blocks.append(
                EdgeConvBlock(
                    ch,
                    hidden,
                    k=k,
                    groups_gn=groups_gn,
                    residual=(residual and ch == hidden),
                )
            )
            ch = hidden
        self.blocks = nn.ModuleList(blocks)
        # optional fusion back to in_ch
        self.fuse = nn.Sequential(
            nn.Conv1d(ch, in_ch, 1),
            gn(in_ch, groups_gn),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_1d):
        """
        feat_1d: (B,C,N)  (1D conv layout)
        """
        x = feat_1d
        f = x
        for blk in self.blocks:
            f = blk(f)          # (B,hidden,N)
        f = self.fuse(f)        # (B,in_ch,N)
        return x + f            # residual to stabilize


# ----------------- PointTransformer Block -----------------
class PointTransformerLayer(nn.Module):
    """
    一个局部 PointTransformer 层：
      - 基于 xyz 的 kNN 建图
      - 使用相对坐标 positional encoding
      - 邻域注意力聚合
      - 残差 + GN + ReLU
    """
    def __init__(self, dim, k=16, groups_gn=32):
        super().__init__()
        self.k = int(max(1, k))
        self.dim = dim

        self.to_q = nn.Conv1d(dim, dim, 1)
        self.to_k = nn.Conv1d(dim, dim, 1)
        self.to_v = nn.Conv1d(dim, dim, 1)

        # relative position encoding: (B,3,N,k) -> (B,dim,N,k)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            gn(dim, groups_gn),
            nn.ReLU(inplace=True),
        )

        # attention weight MLP: (B,dim,N,k) -> (B,dim,N,k)
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gn(dim, groups_gn),
            nn.ReLU(inplace=True),
        )

        self.norm = gn(dim, groups_gn)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz_bcn, feat):
        """
        xyz_bcn: (B,3,N)
        feat:    (B,dim,N)
        return:  (B,dim,N)
        """
        B, _, N = xyz_bcn.shape
        assert feat.shape[2] == N and feat.shape[1] == self.dim

        # kNN based on xyz
        idx = knn_idx_full(xyz_bcn, self.k)   # (B,N,k)

        # project to q,k,v
        q = self.to_q(feat)                   # (B,dim,N)
        k = self.to_k(feat)
        v = self.to_v(feat)

        # gather neighbors
        q_i = q.unsqueeze(-1).expand(-1, -1, -1, self.k)     # (B,dim,N,k)
        k_j = gather_by_idx(k, idx)                          # (B,dim,N,k)
        v_j = gather_by_idx(v, idx)                          # (B,dim,N,k)

        # relative position encoding
        xyz = xyz_bcn
        pos_j = gather_by_idx(xyz, idx)                      # (B,3,N,k)
        pos_i = xyz.unsqueeze(-1).expand(-1, -1, -1, self.k) # (B,3,N,k)
        delta = pos_j - pos_i                                # (B,3,N,k)
        pos_enc = self.pos_mlp(delta)                        # (B,dim,N,k)

        # attention logits
        attn = self.attn_mlp(k_j - q_i + pos_enc)            # (B,dim,N,k)
        attn = attn.sum(dim=1)                               # (B,N,k)
        attn = torch.softmax(attn, dim=-1)                   # (B,N,k)
        attn = attn.unsqueeze(1)                             # (B,1,N,k)

        # aggregate neighbors
        agg = (v_j + pos_enc) * attn                         # (B,dim,N,k)
        agg = agg.sum(dim=-1)                                # (B,dim,N)

        # residual + norm + act
        out = feat + agg
        out = self.act(self.norm(out))
        return out


# ----------------- PointTransformer Regressor -----------------
class PointTransformerRegressor(nn.Module):
    """
    输入:
      - xyz_bcn: (B,3,N) 归一化坐标
      - feat0:   (B,C0,N) 额外点特征（可选, 默认 None）

    输出:
      - (B,4,N) 逐点回归（通道顺序: [p,u,v,w]）

    结构:
      [Conv1d 输入投影] -> [L 层 PointTransformerLayer] -> [GraphRefiner*] -> Dual Head
    """
    def __init__(
        self,
        out_channels=4,
        p_drop=0.3,
        dim=128,
        depth=4,
        k_neighbors=16,
        groups_gn=32,
        in_ch0=0,
        p_head_ch=(128, 64),
        uvw_head_ch=(128, 64),
        # Graph refine
        use_graph_refine=True,
        refine_k=16,
        refine_layers=2,
        refine_hidden=128,
        refine_residual=True,
    ):
        super().__init__()
        assert out_channels == 4, "该实现固定输出 4 通道 [p,u,v,w]"
        self.dim = dim
        self.k_neighbors = int(k_neighbors)

        # 输入映射: [xyz(3) + feat0(in_ch0)] -> dim
        in_total = 3 + int(in_ch0)
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_total, dim, 1),
            gn(dim, groups_gn),
            nn.ReLU(inplace=True),
        )

        # PointTransformer 主干
        blocks = []
        for _ in range(depth):
            blocks.append(
                PointTransformerLayer(dim=dim, k=self.k_neighbors, groups_gn=groups_gn)
            )
        self.blocks = nn.ModuleList(blocks)

        # Graph refine (可选，接口保持与 PointNet2Regressor 一致)
        self.use_graph_refine = use_graph_refine
        if use_graph_refine:
            self.refiner = GraphRefiner(
                in_ch=dim,
                hidden=refine_hidden,
                layers=refine_layers,
                k=refine_k,
                groups_gn=groups_gn,
                residual=refine_residual,
            )

        # Dual head
        ph1, ph2 = p_head_ch
        uh1, uh2 = uvw_head_ch
        self.p_head = nn.Sequential(
            nn.Conv1d(dim, ph1, 1), gn(ph1, groups_gn), nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Conv1d(ph1, ph2, 1), gn(ph2, groups_gn), nn.ReLU(inplace=True),
            nn.Conv1d(ph2, 1, 1),
        )
        self.uvw_head = nn.Sequential(
            nn.Conv1d(dim, uh1, 1), gn(uh1, groups_gn), nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Conv1d(uh1, uh2, 1), gn(uh2, groups_gn), nn.ReLU(inplace=True),
            nn.Conv1d(uh2, 3, 1),
        )

        # 更稳定的起点
        nn.init.zeros_(self.p_head[-1].weight)
        nn.init.zeros_(self.p_head[-1].bias)
        nn.init.zeros_(self.uvw_head[-1].weight)
        nn.init.zeros_(self.uvw_head[-1].bias)

    def forward(self, xyz_bcn, feat0=None):
        """
        xyz_bcn: (B,3,N)
        feat0:   (B,C0,N) 或 None
        """
        assert xyz_bcn.dim() == 3 and xyz_bcn.size(1) == 3, \
            f"xyz_bcn must be (B,3,N), got {tuple(xyz_bcn.shape)}"
        B, _, N = xyz_bcn.shape

        if feat0 is not None:
            assert feat0.dim() == 3 and feat0.size(2) == N, \
                f"feat0 must be (B,C0,N) with same N, got {tuple(feat0.shape)}"
            x = torch.cat([xyz_bcn, feat0], dim=1)  # (B,3+C0,N)
        else:
            x = xyz_bcn

        # 输入映射
        x = self.in_proj(x)  # (B,dim,N)

        # PointTransformer 层堆叠
        for blk in self.blocks:
            x = blk(xyz_bcn, x)  # (B,dim,N)

        # 可选图细化
        if self.use_graph_refine:
            x = self.refiner(x)  # (B,dim,N)

        # Dual-head 回归
        p = self.p_head(x)      # (B,1,N)
        uvw = self.uvw_head(x)  # (B,3,N)
        out = torch.cat([p, uvw], dim=1)  # (B,4,N)
        return out
