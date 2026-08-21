"""
Clinical Cascade Mamba (CCM) for WSI Survival Analysis — v3.2 (LSMR-default).
Logits-Space Mean Residual (LSMR) as the default Stage-2 enhancement:
  - Keep original attention aggregation in feature space.
  - Add MeanMIL-style patch_logits.mean() residual in logits space.
  - AMALA (feature-space adaptive gate) removed based on ablation results.
Supports dict input: {'patch_features': (N,D), 'patch_coords': (N,2)}
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    raise ImportError("mamba_ssm.Mamba2 not found. Please install mamba-ssm.")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# ===================== Helper functions =====================
def _scatter_to_grid(features, coords, grid_size=None, grid_mode='square_norm'):
    """
    Scatter patch features onto a 2D pseudo-grid using normalized coordinates.
    grid_mode:
      - 'square_norm' (default, v3 legacy): isotropic square grid H=W.
      - 'aspect' (v5 backport): aspect-preserving rectangular grid.
    Returns:
        feature_map: (1, D, H, W)
        mask:        (1, 1, H, W)
        grid_shape:  (H, W) tuple
        (grid_h, grid_w): tuple of (N,) indices
        coords_norm: (N, 2) in [0, 1]
    """
    N, D = features.shape
    c_min = coords.min(dim=0, keepdim=True).values
    c_max = coords.max(dim=0, keepdim=True).values
    c_range = (c_max - c_min).clamp(min=1.0)
    c_norm = (coords - c_min) / c_range  # [0, 1]

    if grid_mode == 'square_norm':
        if grid_size is None:
            grid_size = int(math.ceil(math.sqrt(N * 1.2)))
            grid_size = max(grid_size, 8)
        H = W = grid_size
        grid_h = (c_norm[:, 1] * (H - 1)).long().clamp(0, H - 1)
        grid_w = (c_norm[:, 0] * (W - 1)).long().clamp(0, W - 1)

    elif grid_mode == 'aspect':
        # Backport from v5 MODULE M1: aspect-preserving rectangular grid.
        # Target cell count M = ceil(1.2 * N), min 64.
        M = max(64, int(math.ceil(N * 1.2)))
        Rx = float(c_range[0, 0].item())
        Ry = float(c_range[0, 1].item())
        ratio = Rx / max(Ry, 1e-6)
        W = int(math.ceil(math.sqrt(M * ratio)))
        W = max(8, W)
        H = int(math.ceil(M / W))
        H = max(8, H)
        # Cap cells by analytic isotropic scaling (v5 exact logic).
        if H * W > 65536:
            s = math.sqrt((H * W) / 65536)
            H = max(8, int(math.floor(H / s)))
            W = max(8, int(math.floor(W / s)))
            while H * W > 65536:
                if H >= W and H > 8:
                    H -= 1
                elif W > 8:
                    W -= 1
                else:
                    H, W = 8, 8
                    break
        grid_h = (c_norm[:, 1] * (H - 1)).long().clamp(0, H - 1)
        grid_w = (c_norm[:, 0] * (W - 1)).long().clamp(0, W - 1)

    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode}")

    feature_map = torch.zeros(1, D, H, W,
                               device=features.device, dtype=features.dtype)
    mask = torch.zeros(1, 1, H, W, device=features.device)

    batch_idx = torch.zeros(N, D, dtype=torch.long, device=features.device)
    channel_idx = torch.arange(D, device=features.device).unsqueeze(0).expand(N, -1)
    h_idx = grid_h.unsqueeze(1).expand(-1, D)
    w_idx = grid_w.unsqueeze(1).expand(-1, D)
    feature_map.index_put_((batch_idx, channel_idx, h_idx, w_idx), features, accumulate=True)
    mask.index_put_((batch_idx[:, 0].unsqueeze(1), torch.zeros(N, 1, dtype=torch.long, device=features.device),
                     grid_h.unsqueeze(1), grid_w.unsqueeze(1)),
                    torch.ones(N, 1, device=features.device), accumulate=True)
    mask = mask.clamp(0, 1)

    return feature_map, mask, (H, W), (grid_h, grid_w), c_norm


# ---- Exact diagonal flatten / unflatten (v3) ----
def _diagonal_flatten(fmap, direction):
    """
    Flatten a (D,H,W) feature map along the specified diagonal direction.
    Returns:
        padded: (num_diags, max_len, D)
        pad_mask: (num_diags, max_len)  # 1=valid, 0=padding
        coords:   list of list of (i,j) tuples for each diagonal
    """
    D, H, W = fmap.shape
    fmap_hwc = fmap.permute(1, 2, 0)  # (H, W, D)
    lines, coords, pad_mask = [], [], []

    if direction in [45, 225]:
        for k in range(H + W - 1):
            i_start = max(0, k - W + 1)
            i_end = min(H, k + 1)
            diag, coord = [], []
            for i in range(i_start, i_end):
                j = k - i
                diag.append(fmap_hwc[i, j])
                coord.append((i, j))
            if diag:
                lines.append(torch.stack(diag, dim=0))
                coords.append(coord)
    else:  # 135, 315
        for k in range(H + W - 1):
            i_start = max(0, k - W + 1)
            i_end = min(H, k + 1)
            diag, coord = [], []
            for i in range(i_start, i_end):
                j = W - 1 - (k - i)
                if 0 <= j < W:
                    diag.append(fmap_hwc[i, j])
                    coord.append((i, j))
            if diag:
                lines.append(torch.stack(diag, dim=0))
                coords.append(coord)

    if len(lines) == 0:
        empty = fmap.reshape(-1, D)
        return empty.unsqueeze(0), torch.ones(1, empty.shape[0], device=fmap.device), [[(0, 0)]]

    max_len = max(l.shape[0] for l in lines)
    padded = []
    for l in lines:
        pad_len = max_len - l.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, D, device=l.device, dtype=l.dtype)
            l = torch.cat([l, pad], dim=0)
        padded.append(l)
    padded = torch.stack(padded, dim=0)  # (num_diags, max_len, D)

    pad_mask = torch.zeros(len(lines), max_len, device=fmap.device)
    for idx, l in enumerate(lines):
        pad_mask[idx, :l.shape[0]] = 1.0

    return padded, pad_mask, coords


def _diagonal_unflatten(seq, coords, H, W):
    """
    Reconstruct (1, D, H, W) from a diagonal-flattened sequence using stored coords.
    """
    num_diags, max_len, D = seq.shape
    result = torch.zeros(D, H, W, device=seq.device, dtype=seq.dtype)
    count = torch.zeros(H, W, device=seq.device, dtype=seq.dtype)

    for d in range(num_diags):
        for t, (i, j) in enumerate(coords[d]):
            result[:, i, j] += seq[d, t]
            count[i, j] += 1.0

    count = count.clamp(min=1.0)
    result = result / count.unsqueeze(0)
    return result.unsqueeze(0)  # (1, D, H, W)


def _scan_direction(feature_map, direction, ssm_layers, drop_path_rate=0.0):
    """
    Flatten feature_map along `direction`, pass through Mamba2, and restore spatial shape.
    """
    _, D, H, W = feature_map.shape
    diag_coords = None

    if direction == 0:          # left -> right (row-major)
        seq = feature_map[0].permute(1, 2, 0)  # (H, W, D)
    elif direction == 180:      # right -> left
        seq = feature_map[0].permute(1, 2, 0).flip(1)
    elif direction == 90:       # top -> bottom (col-major)
        seq = feature_map[0].permute(2, 1, 0)  # (W, H, D)
    elif direction == 270:      # bottom -> top
        seq = feature_map[0].permute(2, 1, 0).flip(1)
    elif direction in [45, 135, 225, 315]:
        seq, _, diag_coords = _diagonal_flatten(feature_map[0], direction=direction)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    out = seq
    for ssm in ssm_layers:
        if drop_path_rate > 0.0 and out.requires_grad:
            residual = ssm(out)
            keep_prob = 1.0 - drop_path_rate
            mask = torch.rand(out.shape[0], 1, 1, device=out.device, dtype=out.dtype) < keep_prob
            out = out + residual * mask / keep_prob
        else:
            out = out + ssm(out)

    # Restore direction flips
    if direction == 180:
        out = out.flip(1)
    elif direction == 270:
        out = out.flip(1)

    # Restore spatial dimensions (1, D, H, W)
    if direction in [0, 180]:
        result = out.permute(2, 0, 1).unsqueeze(0)
    elif direction in [90, 270]:
        result = out.permute(2, 0, 1).unsqueeze(0)
        result = result.permute(0, 1, 3, 2)
    else:
        result = _diagonal_unflatten(out, diag_coords, H, W)

    return result


# ===================== Stage 2 reordering =====================
def _reorder_center_out(features, coords_norm, importance):
    center = (coords_norm * importance.unsqueeze(1)).sum(dim=0) / importance.sum().clamp(min=1e-6)
    dist = torch.norm(coords_norm - center.unsqueeze(0), dim=1)
    sorted_idx = torch.argsort(dist, descending=False)
    return features[sorted_idx]


def _reorder_risk_gradient(features, coords_norm, importance):
    sorted_idx = torch.argsort(importance, descending=True)
    return features[sorted_idx]


def _reorder_structure_guided(features, coords, grid_h, grid_w, grid_size, importance):
    if features.shape[0] < 2:
        return features
    coords_norm = coords.float()
    c_mean = coords_norm.mean(dim=0)
    centered = coords_norm - c_mean
    cov = centered.t() @ centered / centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    main_dir = eigvecs[:, -1]
    proj = centered @ main_dir
    sorted_idx = torch.argsort(proj, descending=False)
    return features[sorted_idx]


# ===================== CCM main model =====================
class CCM_MIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False,
                 stage1_dir=4, stage2_mode='center_out', soft_topk_ratio=0.3,
                 stage2_layers=1, drop_path_rate=0.0, ablation_mode='none',
                 tau_init=1.0, tau_min=0.05, selection_mode='soft',
                 diagonal_only=False, grid_mode='square_norm'):
        super().__init__()
        self.feat_dim = 512
        self.n_classes = n_classes
        self.survival = survival
        self.dropout_rate = dropout if dropout else 0.0
        self.stage1_dir = stage1_dir
        self.stage2_mode = stage2_mode
        self.soft_topk_ratio = soft_topk_ratio
        self.drop_path_rate = drop_path_rate
        self.ablation_mode = ablation_mode
        self.diagonal_only = diagonal_only
        # 'soft' 保留所有 patch(仅重排); 'hard' 截断到 top-K patch。
        self.selection_mode = selection_mode
        self.grid_mode = grid_mode

        # Annealable tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.register_buffer('tau', torch.tensor(tau_init))

        # Base directions
        # single-direction ablation collapses Stage 1 to one row-major scan.
        if ablation_mode == 'single_direction' or stage1_dir == 1:
            base_dirs = [0]
        elif stage1_dir == 2:
            base_dirs = [0, 90]
        elif stage1_dir == 8:
            base_dirs = [0, 90, 180, 270, 45, 135, 225, 315]
        elif stage1_dir == 4 and diagonal_only:
            base_dirs = [45, 135, 225, 315]
        else:  # default 4-direction
            base_dirs = [0, 90, 180, 270]
        self.directions = base_dirs
        self.num_dirs = len(self.directions)

        # --- Embedding ---
        fc_layers = [nn.Linear(in_dim, self.feat_dim)]
        if act.lower() == 'relu':
            fc_layers += [nn.ReLU()]
        elif act.lower() == 'gelu':
            fc_layers += [nn.GELU()]
        if dropout:
            fc_layers += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*fc_layers)
        self.embed_norm = nn.LayerNorm(self.feat_dim)

        # --- Stage 1: Multi-directional Mamba2 ---
        if ablation_mode != 'no_stage1':
            self.ssm_layers = nn.ModuleList([
                nn.ModuleList([Mamba2(d_model=self.feat_dim, d_state=64, d_conv=4, expand=2)
                               for _ in range(1)])
                for _ in range(self.num_dirs)
            ])
            self.dir_weights = nn.Parameter(torch.ones(self.num_dirs))
            self.importance_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 4),
                nn.GELU(),
                nn.Linear(self.feat_dim // 4, 1)
            )
        else:
            self.ssm_layers = None
            self.dir_weights = None
            self.importance_head = None

        # --- Stage 2: Semantic Reordering Mamba2 ---
        self.stage2_mamba = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                Mamba2(d_model=self.feat_dim, d_state=64, d_conv=4, expand=2)
            )
            for _ in range(stage2_layers)
        ])
        self.stage2_norm = nn.LayerNorm(self.feat_dim)

        # --- Fusion & Head ---
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.norm = nn.LayerNorm(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)

        # ===== v3.2: LSMR Logits 融合门控 (default, AMALA removed) =====
        self.logits_gate = nn.Sequential(
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid()
        )
        # 初始偏向 CCM (mean_weight ≈ 0.1)，保护其他癌种
        self.logits_gate[0].bias.data.fill_(-2.2)

        self.apply(initialize_weights)

    def update_tau(self, epoch, total_epochs):
        """Cosine annealing for Gumbel-Softmax temperature."""
        progress = epoch / max(total_epochs, 1)
        tau = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (1 + math.cos(progress * math.pi))
        self.tau.fill_(tau)

    def _scatter_and_scan(self, features, coords):
        """Scatter patches to grid, multi-direction scan, return global features & importance."""
        feature_map, mask, grid_shape, (grid_h, grid_w), coords_norm = _scatter_to_grid(
            features, coords, grid_size=None, grid_mode=self.grid_mode
        )
        H, W = grid_shape

        if self.ablation_mode == 'grid_only':
            # grid-only ablation: keep the 2D grid embedding but skip directional Mamba scanning.
            coarse_feat = feature_map  # (1, D, H, W)
        else:
            scans = []
            for i, direction in enumerate(self.directions):
                s = _scan_direction(feature_map, direction, self.ssm_layers[i],
                                    drop_path_rate=self.drop_path_rate)
                scans.append(s)

            weights = F.softmax(self.dir_weights, dim=0)
            coarse_feat = sum(w * s for w, s in zip(weights, scans))  # (1, D, H, W)
        coarse_feat = coarse_feat * mask

        feat_flat = coarse_feat[0].permute(1, 2, 0).reshape(-1, self.feat_dim)  # (H*W, D)
        scores_flat = self.importance_head(feat_flat).squeeze(-1)  # (H*W,)
        scores = scores_flat.reshape(H, W)
        scores = scores * mask[0, 0]

        return coarse_feat, scores, mask, (grid_h, grid_w), coords_norm, feat_flat

    def _soft_topk_mask(self, scores, mask, ratio=0.3, tau=None):
        """Soft Top-K mask with annealable temperature."""
        valid_mask = mask[0, 0] > 0.5
        valid_scores = scores[valid_mask]
        N_valid = valid_scores.shape[0]
        if N_valid == 0:
            return torch.zeros_like(scores), torch.zeros_like(scores)

        K = max(1, int(ratio * N_valid))

        if tau is None:
            tau = max(self.tau.item(), 0.05)

        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(valid_scores) + 1e-8) + 1e-8)
            perturbed = (valid_scores + gumbel) / tau
        else:
            perturbed = valid_scores / tau

        probs = F.softmax(perturbed, dim=0)
        soft_mask = torch.zeros_like(scores)
        soft_mask[valid_mask] = probs * K
        return soft_mask, probs

    def _select_and_reorder(self, feats, coords, coords_norm, grid_h, grid_w,
                            patch_importance, H):
        """Build the Stage-2 input sequence from Stage-1 importance.
        no_selection: all patches, original order. hard: top-K only. soft(default): reorder."""
        if self.ablation_mode == 'no_selection':
            return feats
        if self.selection_mode == 'hard':
            N_patch = feats.shape[0]
            K = max(1, int(self.soft_topk_ratio * N_patch))
            K = min(K, N_patch)
            topk_idx = torch.topk(patch_importance, K).indices
            return feats[topk_idx]
        if self.stage2_mode == 'center_out':
            return _reorder_center_out(feats, coords_norm, patch_importance)
        elif self.stage2_mode == 'risk_gradient':
            return _reorder_risk_gradient(feats, coords_norm, patch_importance)
        elif self.stage2_mode == 'structure_guided':
            return _reorder_structure_guided(feats, coords, grid_h, grid_w, H, patch_importance)
        return feats

    def forward(self, x):
        """
        x: dict with 'patch_features' (N, D) and 'patch_coords' (N, 2)
           or tensor (B, N, D) / (N, D) for backward compatibility.
        """
        if isinstance(x, dict):
            patch_features = x['patch_features']
            patch_coords = x['patch_coords']
        else:
            patch_features = x
            patch_coords = None

        if len(patch_features.shape) == 2:
            patch_features = patch_features.unsqueeze(0)  # (1, N, D)

        B, N, D = patch_features.shape
        h = patch_features.float()  # (1, N, D)
        h = self._fc1(h)
        h = self.embed_norm(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        feats = h[0]  # (N, 512)
        coords = patch_coords
        if coords is not None and coords.dim() == 3:
            coords = coords[0]  # (N, 2)

        # ================== Stage 1 ==================
        if self.ablation_mode == 'no_stage1':
            N = feats.shape[0]
            if coords is None:
                coords_norm = torch.linspace(0, 1, N, device=feats.device).unsqueeze(1).expand(N, 2)
            else:
                coords_norm = (coords.float() - coords.float().min(dim=0, keepdim=True).values) / \
                              (coords.float().max(dim=0, keepdim=True).values - coords.float().min(dim=0, keepdim=True).values + 1e-8)
            reordered = feats
            global_token = feats.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
            patch_importance = torch.ones(N, device=feats.device) / N
            scores = torch.zeros(N, device=feats.device)
            soft_mask = torch.ones(N, device=feats.device)
            probs = patch_importance
        else:
            coarse_feat, scores, mask, (grid_h, grid_w), coords_norm, feat_flat = self._scatter_and_scan(feats, coords)
            H, W = scores.shape

            if self.ablation_mode == 'random_mask':
                valid_mask = mask[0, 0] > 0.5
                N_valid = valid_mask.sum().item()
                random_scores = torch.rand(N_valid, device=feats.device)
                scores_random = torch.zeros_like(scores)
                scores_random[valid_mask] = random_scores
                soft_mask, probs = self._soft_topk_mask(scores_random, mask, ratio=self.soft_topk_ratio)
            else:
                soft_mask, probs = self._soft_topk_mask(scores, mask, ratio=self.soft_topk_ratio)

            patch_importance = soft_mask[grid_h, grid_w]
            patch_importance = patch_importance / patch_importance.sum().clamp(min=1e-6)

            reordered = self._select_and_reorder(
                feats, coords, coords_norm, grid_h, grid_w, patch_importance, H)

            global_token = coarse_feat.mean(dim=(2, 3)).unsqueeze(1)  # (1, 1, D)

        # ================== Ablation: no_stage2 ==================
        if self.ablation_mode == 'no_stage2':
            h_global = global_token.squeeze(1)  # (1, D)
            h_fused = self.norm(h_global)
            logits = self.classifier(h_fused)
            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            if self.survival:
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                return hazards, S, Y_hat, None, None
            return logits, Y_prob, Y_hat, None, None

        # ================== Stage 2 (normal or no_stage1) ==================
        seq = torch.cat([global_token, reordered.unsqueeze(0)], dim=1)  # (1, 1+N, D)

        for mamba_block in self.stage2_mamba:
            residual = seq
            seq = mamba_block[0](seq)  # LayerNorm
            seq = mamba_block[1](seq)  # Mamba2
            seq = seq + residual

        seq = self.stage2_norm(seq)

        # ============================================================
        # ===== Stage 2 Output: LSMR Logits Fusion (default) =========
        # ============================================================

        h_global = seq[:, 0, :]   # (1, D) — Stage-2 输出的全局 token
        h_seq = seq[:, 1:, :]     # (1, N, D) — Stage-2 输出的局部 patches

        # attention 聚合
        attn_scores = self.attention(h_seq)           # (1, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        h_local = (h_seq * attn_weights).sum(dim=1)   # (1, D)

        # 特征融合
        h_fused = self.fusion_proj(torch.cat([h_global, h_local], dim=-1))
        h_fused = self.norm(h_fused)

        if self.ablation_mode == 'no_lsmr':
            # ---------- Standard CCM path (LSMR disabled) ----------
            logits = self.classifier(h_fused)         # (1, n_classes)
            mean_weight = None
        else:
            # ---------- LSMR: logits-space residual fusion ----------
            # Path 1: CCM feature-space path
            logits_ccm = self.classifier(h_fused)     # (1, n_classes)

            # Path 2: MeanMIL logits-space path
            patch_logits = self.classifier(h_seq)     # (1, N, n_classes)
            logits_mean = patch_logits.mean(dim=1)    # (1, n_classes)

            # Adaptive fusion
            mean_weight = self.logits_gate(h_global)  # (1, 1), [0,1]
            logits = (1 - mean_weight) * logits_ccm + mean_weight * logits_mean

        # ============================================================
        # ===== Stage 2 Output End ===================================
        # ============================================================

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        return logits, Y_prob, Y_hat, None, None

    def forward_with_attention(self, x):
        """
        Forward with intermediate variables returned for visualization.
        """
        if isinstance(x, dict):
            patch_features = x['patch_features']
            patch_coords = x['patch_coords']
        else:
            patch_features = x
            patch_coords = None

        if len(patch_features.shape) == 2:
            patch_features = patch_features.unsqueeze(0)

        h = patch_features.float()
        h = self._fc1(h)
        h = self.embed_norm(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        feats = h[0]
        coords = patch_coords[0] if patch_coords is not None and patch_coords.dim() == 3 else patch_coords

        # ---------------- no_stage1 branch ----------------
        if self.ablation_mode == 'no_stage1':
            N = feats.shape[0]
            if coords is None:
                coords_norm = torch.linspace(0, 1, N, device=feats.device).unsqueeze(1).expand(N, 2)
            else:
                c_min = coords.float().min(dim=0, keepdim=True).values
                c_max = coords.float().max(dim=0, keepdim=True).values
                coords_norm = (coords.float() - c_min) / (c_max - c_min + 1e-8)

            reordered = feats
            global_token = feats.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)

            seq = torch.cat([global_token, reordered.unsqueeze(0)], dim=1)
            for mamba_block in self.stage2_mamba:
                residual = seq
                seq = mamba_block[0](seq)
                seq = mamba_block[1](seq)
                seq = seq + residual

            seq = self.stage2_norm(seq)
            h_global = seq[:, 0, :]
            h_seq = seq[:, 1:, :]

            # ===== Stage 2 Output (no_stage1 branch) =====
            attn_scores = self.attention(h_seq)
            attn_weights = F.softmax(attn_scores, dim=1)
            h_local = (h_seq * attn_weights).sum(dim=1)

            h_fused = self.fusion_proj(torch.cat([h_global, h_local], dim=-1))
            h_fused = self.norm(h_fused)

            if self.ablation_mode == 'no_lsmr':
                logits = self.classifier(h_fused)
                mean_weight = torch.tensor([[0.0]], device=feats.device)
            else:
                logits_ccm = self.classifier(h_fused)
                patch_logits = self.classifier(h_seq)
                logits_mean = patch_logits.mean(dim=1)
                mean_weight = self.logits_gate(h_global)
                logits = (1 - mean_weight) * logits_ccm + mean_weight * logits_mean

            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.topk(logits, 1, dim=1)[1]

            vis_dict = {
                'stage1_scores': torch.zeros(N, device=feats.device),
                'stage1_softmask': torch.ones(N, device=feats.device) / N,
                'stage1_probs': torch.ones(N, device=feats.device) / N,
                'stage2_attn': attn_weights[0, :, 0].detach().cpu(),
                'mean_weight': mean_weight.item(),
                'coords': coords.detach().cpu() if coords is not None else torch.zeros(N, 2),
                'coords_norm': coords_norm.detach().cpu(),
                'coarse_feat': None,
                'grid_h': torch.arange(N, device=feats.device),
                'grid_w': torch.zeros(N, device=feats.device),
                'dir_weights': None,
                'tau': self.tau.item(),
            }

            if self.survival:
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                return (hazards, S, Y_hat), vis_dict

            return (logits, Y_prob, Y_hat), vis_dict

        # ---------------- Normal branch ----------------
        coarse_feat, scores, mask, (grid_h, grid_w), coords_norm, feat_flat = self._scatter_and_scan(feats, coords)
        H, W = scores.shape

        soft_mask, probs = self._soft_topk_mask(scores, mask, ratio=self.soft_topk_ratio)
        patch_importance = soft_mask[grid_h, grid_w]
        patch_importance = patch_importance / patch_importance.sum().clamp(min=1e-6)

        reordered = self._select_and_reorder(
            feats, coords, coords_norm, grid_h, grid_w, patch_importance, H)

        global_token = coarse_feat.mean(dim=(2, 3)).unsqueeze(1)
        seq = torch.cat([global_token, reordered.unsqueeze(0)], dim=1)

        for mamba_block in self.stage2_mamba:
            residual = seq
            seq = mamba_block[0](seq)
            seq = mamba_block[1](seq)
            seq = seq + residual

        seq = self.stage2_norm(seq)
        h_global = seq[:, 0, :]
        h_seq = seq[:, 1:, :]

        # ===== Stage 2 Output (normal branch) =====
        attn_scores = self.attention(h_seq)
        attn_weights = F.softmax(attn_scores, dim=1)
        h_local = (h_seq * attn_weights).sum(dim=1)

        h_fused = self.fusion_proj(torch.cat([h_global, h_local], dim=-1))
        h_fused = self.norm(h_fused)

        if self.ablation_mode == 'no_lsmr':
            logits = self.classifier(h_fused)
            mean_weight = torch.tensor([[0.0]], device=feats.device)
        else:
            logits_ccm = self.classifier(h_fused)
            patch_logits = self.classifier(h_seq)
            logits_mean = patch_logits.mean(dim=1)
            mean_weight = self.logits_gate(h_global)
            logits = (1 - mean_weight) * logits_ccm + mean_weight * logits_mean

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        dir_weights = F.softmax(self.dir_weights, dim=0).detach().cpu() if self.dir_weights is not None else None
        vis_dict = {
            'stage1_scores': scores[grid_h, grid_w].detach().cpu(),
            'stage1_softmask': soft_mask[grid_h, grid_w].detach().cpu(),
            'stage1_probs': probs.detach().cpu(),
            'stage2_attn': attn_weights[0, :, 0].detach().cpu(),
            'mean_weight': mean_weight.item(),
            'coords': coords.detach().cpu(),
            'coords_norm': coords_norm.detach().cpu(),
            'coarse_feat': coarse_feat.detach().cpu(),
            'grid_h': grid_h.detach().cpu(),
            'grid_w': grid_w.detach().cpu(),
            'dir_weights': dir_weights,
            'tau': self.tau.item(),
        }

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return (hazards, S, Y_hat), vis_dict

        return (logits, Y_prob, Y_hat), vis_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.embed_norm = self.embed_norm.to(device)
        if self.ssm_layers is not None:
            self.ssm_layers = self.ssm_layers.to(device)
            self.importance_head = self.importance_head.to(device)
        self.stage2_mamba = self.stage2_mamba.to(device)
        self.stage2_norm = self.stage2_norm.to(device)
        self.fusion_proj = self.fusion_proj.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
        self.logits_gate = self.logits_gate.to(device)
        self.to(device)
        if hasattr(self, 'tau'):
            self.tau = self.tau.to(device)
