import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

from .modules import Normalize, nonlinearity

class SCSA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SCSA, self).__init__()
        self.ca_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.sa_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x_ca = self.ca_branch(x)
        x_sa = self.sa_branch(x)

        attention = x_ca * x_sa

        x_attended = x * attention
        return x_attended + self.fusion_conv(x_attended)

class LowFrequencyProcessor(nn.Module):
    def __init__(self, in_channels):
        super(LowFrequencyProcessor, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        )

        self.dil_conv_7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=9, dilation=3)
        self.dil_conv_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=6, dilation=3)
        self.dil_conv_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)

        self.fusion_conv_1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, padding=0)
        self.gelu = nn.GELU()
        self.fusion_conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        self.scsa = SCSA(in_channels)

        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x_init = self.initial_conv(x)

        x_dil_7 = self.dil_conv_7(x_init)
        x_dil_5 = self.dil_conv_5(x_init)
        x_dil_3 = self.dil_conv_3(x_init)

        x_combined = torch.cat([x_dil_7, x_dil_5, x_dil_3], dim=1)
        x_fused = self.fusion_conv_1(x_combined)
        x_fused = self.fusion_conv_2(self.gelu(x_fused))

        x_attn = self.scsa(x_fused)

        return self.residual_conv(x) + x_attn

class HighFrequencyProcessor(nn.Module):
    def __init__(self, in_channels):
        super(HighFrequencyProcessor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.conv1(x) + x)
        x2 = self.gelu(self.conv2(x1) + x1 + x)
        x3 = self.gelu(self.conv3(x2) + x2 + x1 + x)
        x4 = self.gelu(self.conv4(x3) + x3 + x2 + x1 + x)

        return x4

class WaveletMSDown(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet_type='haar'):
        super(WaveletMSDown, self).__init__()

        assert in_channels == out_channels, "WaveletMSDown 必须保持通道数不变以适配当前架构"

        self.dwt = DWTForward(J=1, wave=wavelet_type, mode='zero')

        self.low_freq_processor = LowFrequencyProcessor(in_channels)

        self.high_freq_processor_lh = HighFrequencyProcessor(in_channels)
        self.high_freq_processor_hl = HighFrequencyProcessor(in_channels)
        self.high_freq_processor_hh = HighFrequencyProcessor(in_channels)

        self.fusion_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, padding=0)
        self.norm_out = Normalize(out_channels)
        self.gelu_out = nn.GELU()

    def forward(self, x):
        ll, yh_list = self.dwt(x)
        yh = yh_list[0]

        lh = yh[:, :, 0, :, :]
        hl = yh[:, :, 1, :, :]
        hh = yh[:, :, 2, :, :]

        ll_out = self.low_freq_processor(ll)
        lh_out = self.high_freq_processor_lh(lh)
        hl_out = self.high_freq_processor_hl(hl)
        hh_out = self.high_freq_processor_hh(hh)

        combined = torch.cat([ll_out, lh_out, hl_out, hh_out], dim=1)

        fused_output = self.fusion_conv(combined)
        return self.gelu_out(self.norm_out(fused_output))
