import torch
import torch.nn as nn
import torch.nn.functional as F

class ACFG(nn.Module):
    def __init__(self, channels, gamma_init=0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, low_freq, high_freq_list):
        # high_freq_list: list of [B,C,6,H,W]
        hf = torch.stack(high_freq_list, dim=0).mean(dim=(0, 2))
        hf = F.interpolate(hf, size=low_freq.shape[-2:], mode="bilinear")

        fused = self.conv(torch.cat([low_freq, hf], dim=1))
        return self.gamma * fused + (1 - self.gamma) * low_freq
