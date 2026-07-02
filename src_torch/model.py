"""Clean, declarative PyTorch-native HRRRCast inference model (NCHW).

This module contains no dependency on the exported Keras architecture metadata
(`static_architecture.py`). All shapes, block counts, and channel widths are
declared explicitly in `HRRRCast.__init__`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


HRRR_CHANNELS = 138
GFS_CHANNELS = 42
NOISED_CHANNELS = 138
OTHER_CHANNELS = 10
TIME_DIM = 256
LN_EPS = 1.0e-6
REFLECT_PAD_H = (3, 2)
REFLECT_PAD_W = (1, 0)


class ChannelLayerNorm(nn.Module):
    """LayerNorm along the channel axis for NCHW (or feature axis for NC).

    Equivalent to Keras `LayerNormalization(axis=[C])`.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        affine_scale: bool = True,
        affine_center: bool = True,
        eps: float = LN_EPS,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine_scale:
            self.gamma = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter("gamma", None)
        if affine_center:
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if x.ndim == 2:
            if self.gamma is not None:
                y = y * self.gamma
            if self.beta is not None:
                y = y + self.beta
            return y
        if self.gamma is not None:
            y = y * self.gamma[None, :, None, None]
        if self.beta is not None:
            y = y + self.beta[None, :, None, None]
        return y


class ResidualBlock(nn.Module):
    """One HRRRCast residual block: conv-LN-relu-conv-LN -> FiLM -> CBAM -> shortcut."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cbam_reduce: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = ChannelLayerNorm(out_ch, affine_scale=False, affine_center=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(out_ch, affine_scale=True, affine_center=True)
        self.film_gamma = nn.Linear(time_dim, out_ch)
        self.film_beta = nn.Linear(time_dim, out_ch)
        self.cbam_ch_mlp_1 = nn.Linear(out_ch, cbam_reduce)
        self.cbam_ch_mlp_2 = nn.Linear(cbam_reduce, out_ch)
        self.cbam_sp_conv7 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = y * self.film_gamma(time)[:, :, None, None] + self.film_beta(time)[:, :, None, None]
        ch_pool = y.mean(dim=(2, 3))
        ch_gate = torch.sigmoid(self.cbam_ch_mlp_2(F.relu(self.cbam_ch_mlp_1(ch_pool))))
        y = y * ch_gate[:, :, None, None]
        sp_gate = torch.sigmoid(self.cbam_sp_conv7(y.mean(dim=1, keepdim=True)))
        y = y * sp_gate
        skip = self.shortcut(x) if self.shortcut is not None else x
        return F.relu(y + skip)


class ResidualStack(nn.Module):
    """A stack of `ResidualBlock`s with the first block doing any channel projection."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, time_dim: int, cbam_reduce: int):
        super().__init__()
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            block_in = in_ch if i == 0 else out_ch
            blocks.append(ResidualBlock(block_in, out_ch, time_dim, cbam_reduce))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time)
        return x


class HRRRCast(nn.Module):
    """PyTorch-native HRRRCast diffusion denoiser in NCHW layout.

    Accepts NCHW tensors of shape `(B, 328, H, W)` and returns `(B, 138, H, W)`.
    """

    def __init__(self) -> None:
        super().__init__()
        processed_ch = 136 + 40 + 136 + OTHER_CHANNELS  # 322

        self.time_dense = nn.Linear(2, TIME_DIM)
        self.time_norm = ChannelLayerNorm(TIME_DIM)
        self.skip1 = nn.Linear(TIME_DIM, TIME_DIM)
        self.skip2 = nn.Linear(TIME_DIM, 1)

        self.hrrr_pre = ResidualStack(HRRR_CHANNELS + OTHER_CHANNELS, 136, n_blocks=2, time_dim=TIME_DIM, cbam_reduce=17)
        self.gfs_pre = ResidualStack(GFS_CHANNELS + OTHER_CHANNELS, 40, n_blocks=2, time_dim=TIME_DIM, cbam_reduce=20)
        self.noised_pre = ResidualStack(NOISED_CHANNELS + OTHER_CHANNELS, 136, n_blocks=2, time_dim=TIME_DIM, cbam_reduce=17)

        self.enc0 = ResidualStack(processed_ch, 288, n_blocks=2, time_dim=TIME_DIM, cbam_reduce=16)
        self.enc1 = ResidualStack(288, 256, n_blocks=3, time_dim=TIME_DIM, cbam_reduce=16)
        self.enc2 = ResidualStack(256, 256, n_blocks=4, time_dim=TIME_DIM, cbam_reduce=16)

        self.processor = ResidualStack(256, 256, n_blocks=14, time_dim=TIME_DIM, cbam_reduce=16)

        self.dec0 = ResidualStack(256 + 256, 256, n_blocks=4, time_dim=TIME_DIM, cbam_reduce=16)
        self.dec1 = ResidualStack(256 + 288, 256, n_blocks=3, time_dim=TIME_DIM, cbam_reduce=16)
        self.dec2 = ResidualStack(256 + processed_ch, 288, n_blocks=2, time_dim=TIME_DIM, cbam_reduce=16)

        self.output_refine = ResidualStack(288, 288, n_blocks=1, time_dim=TIME_DIM, cbam_reduce=16)
        self.output_conv = nn.Conv2d(288, HRRR_CHANNELS, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The saved Keras model uses tf.gather with negative indices for the
        # time-condition input, which yields zeros rather than wrap-around.
        time_input = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
        time = F.relu(self.time_norm(self.time_dense(time_input)))

        padded = F.pad(x, (REFLECT_PAD_W[0], REFLECT_PAD_W[1], REFLECT_PAD_H[0], REFLECT_PAD_H[1]), mode="reflect")
        # Enter channels_last for the whole conv graph; cuDNN then uses NHWC
        # tensor-core kernels directly (no per-conv layout transposes).
        padded = padded.contiguous(memory_format=torch.channels_last)
        hrrr = padded[:, :HRRR_CHANNELS]
        gfs = padded[:, HRRR_CHANNELS:HRRR_CHANNELS + GFS_CHANNELS]
        noised = padded[:, HRRR_CHANNELS + GFS_CHANNELS:HRRR_CHANNELS + GFS_CHANNELS + NOISED_CHANNELS]
        other = padded[:, HRRR_CHANNELS + GFS_CHANNELS + NOISED_CHANNELS:]

        hrrr_p = self.hrrr_pre(torch.cat([hrrr, other], dim=1), time)
        gfs_p = self.gfs_pre(torch.cat([gfs, other], dim=1), time)
        noised_p = self.noised_pre(torch.cat([noised, other], dim=1), time)
        processed = torch.cat([hrrr_p, gfs_p, noised_p, other], dim=1)

        e0 = self.enc0(processed, time)
        p0 = F.max_pool2d(e0, kernel_size=2, stride=2)
        e1 = self.enc1(p0, time)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e2 = self.enc2(p1, time)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2)

        bottleneck = self.processor(p2, time)

        u0 = F.interpolate(bottleneck, scale_factor=2, mode="nearest")
        d0 = self.dec0(torch.cat([u0, p1], dim=1), time)
        u1 = F.interpolate(d0, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([u1, p0], dim=1), time)
        u2 = F.interpolate(d1, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([u2, processed], dim=1), time)

        refined = self.output_conv(self.output_refine(d2, time)).float()
        skip_scale = torch.sigmoid(self.skip2(F.relu(self.skip1(time))))[:, :, None, None]
        out = refined + (hrrr - noised) * skip_scale + noised
        return out[:, :, REFLECT_PAD_H[0]:-REFLECT_PAD_H[1], REFLECT_PAD_W[0]:]
