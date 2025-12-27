# PointNet++ (+ 可选动态图卷积细化) with dual-head [p] & [u,v,w]
import torch
import torch.nn as nn

# ----------------- public factory (保持兼容) -----------------
def get_model(**kwargs):
    """
    兼容旧调用方式：model = pointnet2pp_ssg.get_model()
    关键开关：
      - use_graph_refine: 是否启用动态图卷积细化（默认 True）
      - refine_k:         EdgeConv 的 kNN K（默认 16）
      - refine_layers:    EdgeConv 层数（默认 2）
    其余参数见 PointNet2Regressor 注释。
    """
    defaults = dict(
        out_channels=4,
        p_drop=0.3,
        npoint1=2048, npoint2=512, npoint3=128,
        k1=16, k2=16, k3=16,
        c1=128, c2=256, c3=512,
        groups_gn=32,
        use_pe=False,
        use_msg=False,
        in_ch0=0,
        p_head_ch=(128, 64),
        uvw_head_ch=(128, 64),
        # === 新增的图细化超参 ===
        use_graph_refine=True,
        refine_k=16,
        refine_layers=2,
        refine_hidden=128,
        refine_residual=True,
    )
    defaults.update(kwargs)
    return PointNet2Regressor(**defaults)

# ----------------- small utils -----------------
def gn(ch, groups=32):
    """GroupNorm with fallback to a single group when not divisible."""
    assert ch > 0
    g = min(groups, ch)
    g = g if ch % g == 0 else 1
    return nn.GroupNorm(g, ch)

@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    npoint = int(min(npoint, N))
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_ind = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_ind, farthest, :].view(B, 1, 3)
        d = torch.sum((xyz - centroid) ** 2, -1)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, -1)[1]
    return idx

def knn_group(xyz, centroids, k):
    B, N, _ = xyz.shape
    k = min(k, N)
    dists = torch.cdist(centroids, xyz, p=2)               # (B,M,N)
    idx = torch.topk(dists, k=k, dim=-1, largest=False)[1] # (B,M,k)
    return idx

def group_gather(feat, idx):
    B, M, K = idx.shape
    batch = torch.arange(B, device=feat.device).view(B, 1, 1).expand(B, M, K)
    if feat.dim() == 2:
        return feat[batch, idx]
    else:
        return feat[batch, idx, :]

# -------- Fourier positional encoding (optional) --------
class FourierPE(nn.Module):
    def __init__(self, freqs=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32), persistent=False)

    def forward(self, x):
        if self.freqs.numel() == 0: return x
        outs = [x]
        for b in self.freqs:
            outs += [torch.sin(b * x), torch.cos(b * x)]
        return torch.cat(outs, dim=-1)

# ----------------- Set Abstraction (SSG / MSG) -----------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_ch, out_ch, groups_gn=32, use_pe=False, msg_branch_ratio=0.5):
        """
        npoint: 采样点数
        k:      若传入 int => 单尺度；若传入 list/tuple => 多尺度分支
        """
        super().__init__()
        self.npoint = int(npoint)
        self.use_pe = use_pe
        self.pe = FourierPE() if use_pe else None

        if isinstance(k, int):
            self.k_list = [k]
        else:
            self.k_list = list(k)
            assert len(self.k_list) > 0 and all(kk > 0 for kk in self.k_list)

        # 为多分支分配通道
        if len(self.k_list) == 1:
            branch_out = [out_ch]
        else:
            base = int(out_ch * msg_branch_ratio / max(1, len(self.k_list) - 1))
            remain = out_ch
            branch_out = []
            for _ in range(len(self.k_list) - 1):
                ch = max(32, base); branch_out.append(ch); remain -= ch
            branch_out.append(max(32, remain))
            s = sum(branch_out)
            if s != out_ch: branch_out[-1] += (out_ch - s)

        self.branches = nn.ModuleList()
        for bo in branch_out:
            pe_mul = (1 + 2 * 4) if use_pe else 1
            in_total = 3 * pe_mul + in_ch
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_total, bo // 2, 1), gn(bo // 2, groups_gn), nn.ReLU(inplace=True),
                nn.Conv2d(bo // 2, bo, 1),       gn(bo, groups_gn),       nn.ReLU(inplace=True),
            ))

    def forward(self, xyz, feat):
        B, N, _ = xyz.shape
        M = int(min(self.npoint, N))
        fps_idx = farthest_point_sample(xyz, M)  # (B,M)
        centroids = xyz.gather(1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))

        outs = []
        for bi, k in enumerate(self.k_list):
            idx = knn_group(xyz, centroids, k)            # (B,M,k)
            group_xyz = group_gather(xyz, idx)            # (B,M,k,3)
            delta = group_xyz - centroids.unsqueeze(2)    # (B,M,k,3)
            if self.use_pe: delta = self.pe(delta)
            if feat is None:
                group_feat = delta
            else:
                gfeat = group_gather(feat, idx)                  # (B,M,k,C)
                group_feat = torch.cat([delta, gfeat], dim=-1)   # (B,M,k,3(+PE)+C)
            group_feat = group_feat.permute(0, 3, 1, 2).contiguous()  # (B,C',M,k)
            out = self.branches[bi](group_feat).max(dim=-1)[0]        # (B,Bo,M)
            outs.append(out)

        new_feat = torch.cat(outs, dim=1).permute(0, 2, 1).contiguous()  # (B,M,out_ch)
        new_xyz = centroids
        return new_xyz, new_feat

# ----------------- Feature Propagation -----------------
class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, out_ch, groups_gn=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1), gn(out_ch, groups_gn), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 1), gn(out_ch, groups_gn), nn.ReLU(inplace=True),
        )

    def forward(self, xyz_coarse, feat_coarse, xyz_fine, feat_skip):
        B, Nf, _ = xyz_fine.shape
        dists = torch.cdist(xyz_fine, xyz_coarse, p=2)  # (B,Nf,Mc)
        K = min(3, xyz_coarse.shape[1])
        d3, i3 = torch.topk(dists, k=K, dim=-1, largest=False)

        zero_mask = (d3 <= 1e-12).any(dim=-1, keepdim=True)
        w = 1.0 / (d3 + 1e-12); w = w / w.sum(dim=-1, keepdim=True)
        if zero_mask.any():
            oh = torch.zeros_like(w); min_idx = d3.argmin(dim=-1, keepdim=True)
            oh.scatter_(-1, min_idx, 1.0)
            w = torch.where(zero_mask.expand_as(w), oh, w)

        batch = torch.arange(B, device=xyz_fine.device).view(B, 1, 1).expand(B, Nf, K)
        interp = feat_coarse[batch, i3, :].mul(w.unsqueeze(-1)).sum(dim=2)  # (B,Nf,Cc)

        if feat_skip is None: feat = interp.transpose(1, 2)
        else:                  feat = torch.cat([interp, feat_skip], dim=-1).transpose(1, 2)
        feat = self.mlp(feat).transpose(1, 2).contiguous()
        return feat

# ----------------- Dynamic EdgeConv (DGCNN-style) -----------------
def knn_idx_full(feat, k: int):
    """
    feat: (B,C,N)  -> return indices (B,N,k) based on L2 in feature space
    """
    B, C, N = feat.shape
    k = min(k, N)
    # pairwise distance: (B,N,N) = ||x_i - x_j||^2
    # use (x^2 - 2xTy + y^2) trick
    xx = (feat**2).sum(dim=1, keepdim=False)                              # (B,N)
    dist = xx.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * feat.transpose(1,2) @ feat  # (B,N,N)
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
    batch = torch.arange(B, device=feat.device).view(B,1,1).expand(B,N,K)
    nbr = feat.transpose(1,2)[batch, idx, :]   # (B,N,k,C)
    return nbr.permute(0,3,1,2).contiguous()   # (B,C,N,k)

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
            nn.Conv2d(in_ch*2, out_ch, 1),
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
        xi = x.unsqueeze(-1).expand(-1,-1,-1,self.k)     # (B,C,N,k)
        edge = torch.cat([xi, xj - xi], dim=1)           # (B,2C,N,k)
        y = self.theta(edge).amax(dim=-1)                 # (B,C_out,N)
        if self.residual: y = y + x
        return y

class GraphRefiner(nn.Module):
    """
    Stack of EdgeConv blocks that refines per-point features after FP.
    """
    def __init__(self, in_ch=128, hidden=128, layers=2, k=16, groups_gn=32, residual=True):
        super().__init__()
        blocks = []
        ch = in_ch
        for li in range(layers):
            blocks.append(EdgeConvBlock(ch, hidden, k=k, groups_gn=groups_gn, residual=(residual and ch==hidden)))
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
        feat_1d: (B,128,N)  (1D conv layout)
        """
        x = feat_1d  # (B,128,N)
        f = x
        for blk in self.blocks:
            f = blk(f)          # (B,hidden,N)
        f = self.fuse(f)        # (B,128,N)
        return x + f            # residual to stabilize

# ----------------- PointNet++ Model (+GraphRefiner) -----------------
class PointNet2Regressor(nn.Module):
    """
    输入:
      - xyz_bcn: (B,3,N) 归一化坐标
      - feat0:   (B,C0,N) 额外点特征（可选, 默认 None）
    输出:
      - (B,4,N) 逐点回归（通道顺序: [p,u,v,w]）

    结构:
      SA1 -> SA2 -> SA3 -> FP3 -> FP2 -> FP1(xyz skip) -> [GraphRefiner*] -> Dual Head
    """
    def __init__(
        self,
        out_channels=4,
        p_drop=0.3,
        npoint1=2048, npoint2=512, npoint3=128,
        k1=16, k2=16, k3=16,
        c1=128, c2=256, c3=512,
        groups_gn=32,
        use_pe=False,
        use_msg=False,
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

        # ---------- 统一的 k 选择规则 ----------
        # use_msg=True  : 允许多尺度；若传入 int，则包成 (k,) 作为单分支 MSG
        # use_msg=False : 强制单尺度；若传入 list/tuple，则取 max(k) 作为代表
        def _resolve_k(k, use_msg_flag):
            if use_msg_flag:
                return k if isinstance(k, (list, tuple)) else (k,)
            else:
                return k if isinstance(k, int) else max(k)

        rk1 = _resolve_k(k1, use_msg)
        rk2 = _resolve_k(k2, use_msg)
        rk3 = _resolve_k(k3, use_msg)

        # SA
        self.sa1 = PointNetSetAbstraction(npoint1, rk1, in_ch=in_ch0, out_ch=c1, groups_gn=groups_gn, use_pe=use_pe)
        self.sa2 = PointNetSetAbstraction(npoint2, rk2, in_ch=c1,    out_ch=c2, groups_gn=groups_gn, use_pe=use_pe)
        self.sa3 = PointNetSetAbstraction(npoint3, rk3, in_ch=c2,    out_ch=c3, groups_gn=groups_gn, use_pe=use_pe)

        # FP
        self.fp3 = FeaturePropagation(in_ch=c3 + c2, out_ch=c2, groups_gn=groups_gn)
        self.fp2 = FeaturePropagation(in_ch=c2 + c1, out_ch=c1, groups_gn=groups_gn)
        self.fp1 = FeaturePropagation(in_ch=c1 + 3,  out_ch=128, groups_gn=groups_gn)  # concat xyz

        # Graph refine (可选)
        self.use_graph_refine = use_graph_refine
        if use_graph_refine:
            self.refiner = GraphRefiner(in_ch=128, hidden=refine_hidden,
                                        layers=refine_layers, k=refine_k,
                                        groups_gn=groups_gn, residual=refine_residual)

        # Dual head
        ph1, ph2 = p_head_ch
        uh1, uh2 = uvw_head_ch
        self.p_head = nn.Sequential(
            nn.Conv1d(128, ph1, 1), gn(ph1, groups_gn), nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Conv1d(ph1, ph2, 1), gn(ph2, groups_gn), nn.ReLU(inplace=True),
            nn.Conv1d(ph2, 1, 1)
        )
        self.uvw_head = nn.Sequential(
            nn.Conv1d(128, uh1, 1), gn(uh1, groups_gn), nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Conv1d(uh1, uh2, 1), gn(uh2, groups_gn), nn.ReLU(inplace=True),
            nn.Conv1d(uh2, 3, 1)
        )
        # 更稳定的起点
        nn.init.zeros_(self.p_head[-1].weight);   nn.init.zeros_(self.p_head[-1].bias)
        nn.init.zeros_(self.uvw_head[-1].weight); nn.init.zeros_(self.uvw_head[-1].bias)

    def forward(self, xyz_bcn, feat0=None):
        assert xyz_bcn.dim() == 3 and xyz_bcn.size(1) == 3
        xyz = xyz_bcn.transpose(1, 2).contiguous()  # (B,N,3)
        B, N, _ = xyz.shape

        feat_in = None
        if feat0 is not None:
            assert feat0.dim() == 3 and feat0.size(2) == N
            feat_in = feat0.transpose(1, 2).contiguous()  # (B,N,C0)

        # SA
        l1_xyz, l1_feat = self.sa1(xyz, feat_in)         # (B,N1,C1)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)      # (B,N2,C2)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)      # (B,N3,C3)

        # FP to original N (xyz as skip)
        l2_feat_up = self.fp3(l3_xyz, l3_feat, l2_xyz, l2_feat)   # (B,N2,C2)
        l1_feat_up = self.fp2(l2_xyz, l2_feat_up, l1_xyz, l1_feat)# (B,N1,C1)
        feat_full  = self.fp1(l1_xyz, l1_feat_up, xyz, xyz)       # (B,N,128)

        # to conv1d layout
        x = feat_full.transpose(1, 2).contiguous()  # (B,128,N)

        # optional graph refine (EdgeConv on 128-d features)
        if self.use_graph_refine:
            x = self.refiner(x)                     # (B,128,N)

        # heads
        p   = self.p_head(x)    # (B,1,N)
        uvw = self.uvw_head(x)  # (B,3,N)
        out = torch.cat([p, uvw], dim=1)  # (B,4,N)
        return out
