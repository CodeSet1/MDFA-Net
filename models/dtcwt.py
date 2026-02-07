import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class ComplexMagnitude(nn.Module):
    """
    Convert complex DTCWT coefficients to magnitude (real-valued)
    """
    def forward(self, yh):
        # yh: (B, C, 6, H, W, 2)
        real = yh[..., 0]
        imag = yh[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        return mag


class DTCWT(nn.Module):
    """
    Dual-Tree Complex Wavelet Transform (2D)
    """
    def __init__(self, J=1):
        super().__init__()
        self.J = J
        self.xfm = DTCWTForward(
            J=J,
            biort='near_sym_a',
            qshift='qshift_a'
        )
        self.ifm = DTCWTInverse(
            biort='near_sym_a',
            qshift='qshift_a'
        )
        self.mag = ComplexMagnitude()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            yl: (B, C, H/2, W/2)
            yh_mag: (B, C, 6, H/2, W/2)
        """
        yl, yh = self.xfm(x)
        yh_mag = self.mag(yh[0])
        return yl, yh_mag

    def inverse(self, yl, yh_mag):
        """
        Inverse using magnitude-only coefficients (phase-free approximation)
        """
        real = yh_mag
        imag = torch.zeros_like(real)
        yh = torch.stack([real, imag], dim=-1)
        return self.ifm((yl, [yh]))
