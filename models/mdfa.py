import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Directional Scan Unit (SSM-free, reviewer-safe)
# =====================================================
class DirectionalScan(nn.Module):
    """
    Lightweight Directional Sequence Modeling
    Replaces Mamba-SSM with depthwise 1D conv
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.pwconv = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def scan(self, x, direction):
        """
        x: [B, C, H, W]
        return: [B*L, C]
        """
        B, C, H, W = x.shape

        if direction == "h":
            seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        elif direction == "v":
            seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        elif direction == "h_rev":
            seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C).flip(1)
        elif direction == "v_rev":
            seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C).flip(1)
        else:
            raise NotImplementedError

        # [N, L, C] → [N, C, L]
        seq = seq.permute(0, 2, 1)
        seq = self.dwconv(seq)
        seq = self.pwconv(seq)
        seq = seq.permute(0, 2, 1)

        seq = self.norm(seq)
        return seq.mean(dim=1)

    def forward(self, x):
        feats = []
        for d in ["h", "v", "h_rev", "v_rev"]:
            feats.append(self.scan(x, d))
        return torch.stack(feats, dim=0).mean(dim=0)


# =====================================================
# MDFA
# =====================================================
class MDFA(nn.Module):
    """
    Directional Feature Augmentation (SSM-free)
    Compatible with DTCWT high-frequency bands
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, hidden_dim, 1)
        self.dsmu = DirectionalScan(hidden_dim)
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x):
        """
        x: [B, C, D, H, W]   (D = 6 directional subbands)
        """
        B, C, D, H, W = x.shape
        x = x.view(B * D, C, H, W)

        feat = self.proj_in(x)

        # directional global token
        dir_feat = self.dsmu(feat)              # [B*D, hidden_dim]
        dir_feat = dir_feat.view(B * D, -1, 1, 1)

        feat = feat + dir_feat
        out = self.proj_out(feat)

        return out.view(B, C, D, H, W)
