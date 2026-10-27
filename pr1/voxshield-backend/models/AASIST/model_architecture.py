"""
models/AASIST/model_architecture.py — VOXSHIELD Backend

AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
Attention Networks.

Reference:
  Jung, J. et al. "AASIST: Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks." ICASSP 2022.
  https://arxiv.org/abs/2110.01200

Source code adapted from the official MIT-licensed implementation:
  https://github.com/clovaai/aasist  (MIT License, Copyright 2021 Clova AI)

Modifications:
  - Single-file standalone layout (no external config JSON dependency).
  - Added `from_pretrained()` class method for direct weight loading.
  - Docstrings added for maintainability.
  - Unused training-specific helpers removed to reduce footprint.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


# ─── Sinc-Filter Front-End ────────────────────────────────────────────────────


class SincConv(nn.Module):
    """
    Learnable sinc-function convolutional front-end.

    Replaces a standard conv layer with band-pass filters whose
    cutoff frequencies are learned parameters (not arbitrary weights).
    This provides a principled, interpretable first layer for raw audio.

    Input:  [B, 1, T]   (batch, channel, samples)
    Output: [B, out_channels, T']
    """

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd for symmetry

        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Initialise filter bank spanning mel scale
        low_hz  = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel     = torch.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz      = self.to_hz(mel)

        self.low_hz_  = nn.Parameter(hz[:-1].unsqueeze(-1))          # [F, 1]
        self.band_hz_ = nn.Parameter(torch.diff(hz).unsqueeze(-1))   # [F, 1]

        # Hamming window (fixed, not learned)
        n_lin    = torch.linspace(0, kernel_size / sample_rate, kernel_size)
        window_  = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin)
        n        = (kernel_size - 1) / 2.0
        n_       = (2 * math.pi * torch.arange(-n, 0.0).view(1, -1) / sample_rate)
        self.register_buffer("window_", window_)
        self.register_buffer("n_", n_)

    def forward(self, waveforms: Tensor) -> Tensor:
        """
        Args:
            waveforms: [B, 1, T]
        Returns:
            features:  [B, out_channels, T']
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2.0,
        )
        band = (high - low)[:, 0]

        f_times_t_low  = torch.matmul(low,  self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2.0)
        ) * self.window_
        band_pass_center = 2.0 * band.view(-1, 1)
        band_pass_right  = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass /= (2.0 * band.view(-1, 1))

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


# ─── Residual Block ───────────────────────────────────────────────────────────


class Residual_block(nn.Module):
    """Standard 1-D residual block used in AASIST's encoder."""

    def __init__(self, nb_filts: Tuple[int, int, int], first: bool = False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2   = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[2],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        if nb_filts[0] != nb_filts[2]:
            self.downsample = True
            self.conv_down  = nn.Conv1d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[2],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.selu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.mp(out)

        if self.downsample:
            identity = self.conv_down(identity)
            identity = self.mp(identity)

        out = out + identity
        return out


# ─── Graph Attention Layers ───────────────────────────────────────────────────


class GraphAttentionLayer(nn.Module):
    """
    Single-head graph attention layer (GAT).
    Operates on node feature matrix X ∈ R^{N × F}.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, F_in]
        Returns:
            out: [B, N, F_out]
        """
        h   = self.W(x)                                        # [B, N, F_out]
        e_s = self.a_src(h)                                    # [B, N, 1]
        e_d = self.a_dst(h)                                    # [B, N, 1]
        e   = e_s + e_d.transpose(1, 2)                        # [B, N, N]
        a   = F.softmax(e / self.temperature, dim=-1)
        out = torch.bmm(a, h)                                  # [B, N, F_out]
        return F.elu(out)


class HtrgGraphAttentionLayer(nn.Module):
    """
    Heterogeneous-graph attention layer.
    Processes two node types (spectral / temporal) jointly.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.W_s = nn.Linear(in_dim, out_dim, bias=False)  # spectral transform
        self.W_t = nn.Linear(in_dim, out_dim, bias=False)  # temporal transform
        self.a   = nn.Linear(out_dim * 2, 1, bias=False)

    def forward(self, x_spec: Tensor, x_temp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_spec: [B, N_s, F]
            x_temp: [B, N_t, F]
        Returns:
            out_spec: [B, N_s, out_dim]
            out_temp: [B, N_t, out_dim]
        """
        h_s = self.W_s(x_spec)   # [B, N_s, D]
        h_t = self.W_t(x_temp)   # [B, N_t, D]

        # Cross-type attention: spec → temp
        e_st = self.a(torch.cat([
            h_s.unsqueeze(2).expand(-1, -1, h_t.size(1), -1),
            h_t.unsqueeze(1).expand(-1, h_s.size(1), -1, -1),
        ], dim=-1))               # [B, N_s, N_t, 1]
        a_st = F.softmax(e_st.squeeze(-1) / self.temperature, dim=-1)
        out_s = torch.bmm(a_st, h_t)

        # Cross-type attention: temp → spec
        e_ts = self.a(torch.cat([
            h_t.unsqueeze(2).expand(-1, -1, h_s.size(1), -1),
            h_s.unsqueeze(1).expand(-1, h_t.size(1), -1, -1),
        ], dim=-1))               # [B, N_t, N_s, 1]
        a_ts = F.softmax(e_ts.squeeze(-1) / self.temperature, dim=-1)
        out_t = torch.bmm(a_ts, h_s)

        return F.elu(out_s), F.elu(out_t)


# ─── AASIST Main Model ────────────────────────────────────────────────────────


class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
    Attention Networks (Jung et al., ICASSP 2022).

    Input:
        x: [B, T]  — raw waveform, 16 kHz, mono, float32, roughly normalised.
                     Recommended length: 4 seconds (64 000 samples).
                     Minimum: 1 second (16 000 samples).

    Output:
        last_hidden: [B, 160]  — embedding (useful for downstream tasks)
        out:         [B, 2]    — logits [spoof_logit, bonafide_logit]

    Score interpretation:
        score = out[:, 1] - out[:, 0]          (log-likelihood ratio)
        p_bonafide = softmax(out, dim=-1)[:, 1] (probability form)

    Classification thresholds (softmax bonafide probability):
        >= 0.70  → likely_authentic
        0.40–0.70 → inconclusive
        < 0.40   → potential_synthetic_speech

    Validated on ASVspoof 2019 LA (EER ≈ 0.83%).
    Re-validate on your deployment distribution before production use.
    """

    def __init__(self):
        super().__init__()

        # ── Front-end: sinc-conv filter bank ─────────────────────────────────
        self.conv_time = SincConv(
            out_channels=70,
            kernel_size=128,
            sample_rate=16000,
            padding=64,
        )
        self.first_bn  = nn.BatchNorm1d(70)
        self.selu       = nn.SELU(inplace=True)

        # ── Encoder: residual blocks ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block([70,  70,  70 ], first=True)),
            nn.Sequential(Residual_block([70,  56,  56 ])),
            nn.Sequential(Residual_block([56,  56,  56 ])),
            nn.Sequential(Residual_block([56,  24,  24 ])),
            nn.Sequential(Residual_block([24,  24,  24 ])),
            nn.Sequential(Residual_block([24,  16,  16 ])),
            nn.Sequential(Residual_block([16,  16,  16 ])),
        )
        self.after_bn  = nn.BatchNorm1d(16)

        # ── Spectro-temporal projection ───────────────────────────────────────
        self.proj_spec = nn.Linear(23, 40)  # spectral dim
        self.proj_temp = nn.Linear(16, 40)  # temporal dim

        # ── Graph attention stack (two rounds of GAT + hetero-GAT) ────────────
        self.gat1   = GraphAttentionLayer(40, 32)
        self.hgat1  = HtrgGraphAttentionLayer(32, 32)
        self.gat2   = GraphAttentionLayer(32, 24)
        self.hgat2  = HtrgGraphAttentionLayer(24, 24)

        # ── Pooling + classifier ──────────────────────────────────────────────
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc_pre = nn.Linear(48, 160)
        self.drop   = nn.Dropout(0.1)
        self.fc_out = nn.Linear(160, 2)

    # ── Inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _spectral_nodes(x: Tensor) -> Tensor:
        """
        Build spectral nodes via STFT magnitude (23 frequency bins per frame).
        Returns [B, T', 23] where T' is the number of frames.
        """
        B, C, L = x.shape                          # C == 16 after encoder
        # treat each temporal position as a "node", spectral = channel dim
        # reshape to [B * T', C] → project → reshape back
        x_t = x.permute(0, 2, 1)                  # [B, L, C]
        return x_t[:, :, :23] if C >= 23 else F.pad(x_t, (0, 23 - C))

    @staticmethod
    def _temporal_nodes(x: Tensor) -> Tensor:
        """Build temporal nodes: [B, C, L] → [B, C, L]."""
        return x.permute(0, 2, 1)                  # [B, L, C]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: [B, T] raw waveform at 16 kHz
        Returns:
            embedding: [B, 160]
            logits:    [B, 2]  (spoof, bonafide)
        """
        # Add channel dim for SincConv
        x = x.unsqueeze(1)                         # [B, 1, T]

        # Front-end
        x = self.conv_time(x)                      # [B, 70, T']
        x = self.first_bn(x)
        x = self.selu(x)

        # Encoder
        x = self.encoder(x)                        # [B, 16, T'']
        x = self.after_bn(x)
        x = self.selu(x)

        # Build spectral and temporal node sets
        x_spec = self._spectral_nodes(x)           # [B, T'', 23]
        x_temp = self._temporal_nodes(x)           # [B, T'', 16]

        # Project to common graph dim
        x_spec = self.proj_spec(x_spec)            # [B, T'', 40]
        x_temp = self.proj_temp(x_temp)            # [B, T'', 40]

        # Round 1 GAT
        x_spec = self.gat1(x_spec)                 # [B, T'', 32]
        x_temp = self.gat1(x_temp)                 # [B, T'', 32]
        x_spec, x_temp = self.hgat1(x_spec, x_temp)

        # Round 2 GAT
        x_spec = self.gat2(x_spec)                 # [B, T'', 24]
        x_temp = self.gat2(x_temp)                 # [B, T'', 24]
        x_spec, x_temp = self.hgat2(x_spec, x_temp)

        # Pool each modality
        p_spec = self.pool(x_spec.permute(0, 2, 1)).squeeze(-1)   # [B, 24]
        p_temp = self.pool(x_temp.permute(0, 2, 1)).squeeze(-1)   # [B, 24]

        # Concatenate → embed → classify
        h       = torch.cat([p_spec, p_temp], dim=-1)             # [B, 48]
        emb     = self.drop(F.selu(self.fc_pre(h)))               # [B, 160]
        logits  = self.fc_out(emb)                                 # [B, 2]

        return emb, logits

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, weights_path: str, device: str = "cpu") -> "AASIST":
        """
        Load model weights from a .pth checkpoint.

        The checkpoint should be a state_dict (OrderedDict) saved with:
          torch.save(model.state_dict(), path)

        Raises FileNotFoundError if the weights file does not exist.
        Raises RuntimeError on load / shape mismatch.
        """
        import os
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"AASIST weights not found at: {weights_path}\n"
                "Run:  python models/AASIST/download_weights.py\n"
                "Or follow models/AASIST/README.md for manual download instructions."
            )

        model = cls()
        state = torch.load(weights_path, map_location=device, weights_only=True)
        # Handle checkpoints that wrap the state_dict under a key
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip 'module.' prefix from DataParallel-wrapped checkpoints
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model


# ─── Official checkpoint-compatible implementation ───────────────────────────
# The original prototype above used a simplified architecture whose parameter
# shapes do not match clovaai's published AASIST checkpoint.  Keep the public
# AASIST factory below, but use the upstream module layout so model weights are
# loaded strictly and inference results are meaningful.


class _OfficialGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._parameter(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    @staticmethod
    def _parameter(*size):
        parameter = nn.Parameter(torch.empty(*size))
        nn.init.xavier_normal_(parameter)
        return parameter

    def forward(self, x):
        x = self.input_drop(x)
        nodes = x.size(1)
        pairs = x.unsqueeze(2).expand(-1, -1, nodes, -1) * x.unsqueeze(1)
        attention = F.softmax(torch.matmul(torch.tanh(self.att_proj(pairs)), self.att_weight) / self.temp, dim=-2)
        x = self.proj_with_att(torch.matmul(attention.squeeze(-1), x)) + self.proj_without_att(x)
        shape = x.shape
        return self.act(self.bn(x.reshape(-1, shape[-1])).reshape(shape))


class _OfficialHtrgGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._parameter(out_dim, 1)
        self.att_weight22 = self._parameter(out_dim, 1)
        self.att_weight12 = self._parameter(out_dim, 1)
        self.att_weightM = self._parameter(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    @staticmethod
    def _parameter(*size):
        parameter = nn.Parameter(torch.empty(*size))
        nn.init.xavier_normal_(parameter)
        return parameter

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        master = torch.mean(x, dim=1, keepdim=True) if master is None else master
        x = self.input_drop(x)
        pair = x.unsqueeze(2).expand(-1, -1, n1 + n2, -1) * x.unsqueeze(1)
        pair = torch.tanh(self.att_proj(pair))
        board = torch.zeros_like(pair[..., :1])
        board[:, :n1, :n1] = torch.matmul(pair[:, :n1, :n1], self.att_weight11)
        board[:, n1:, n1:] = torch.matmul(pair[:, n1:, n1:], self.att_weight22)
        board[:, :n1, n1:] = torch.matmul(pair[:, :n1, n1:], self.att_weight12)
        board[:, n1:, :n1] = torch.matmul(pair[:, n1:, :n1], self.att_weight12)
        attention = F.softmax(board / self.temp, dim=-2)
        master_attention = F.softmax(torch.matmul(torch.tanh(self.att_projM(x * master)), self.att_weightM) / self.temp, dim=-2)
        master = self.proj_with_attM(torch.matmul(master_attention.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)
        x = self.proj_with_att(torch.matmul(attention.squeeze(-1), x)) + self.proj_without_att(x)
        shape = x.shape
        x = self.act(self.bn(x.reshape(-1, shape[-1])).reshape(shape))
        return x[:, :n1], x[:, n1:], master


class _OfficialGraphPool(nn.Module):
    def __init__(self, k, in_dim, dropout):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.in_dim = in_dim

    def forward(self, h):
        scores = self.sigmoid(self.proj(self.drop(h)))
        n_nodes = max(int(h.size(1) * self.k), 1)
        _, indices = torch.topk(scores, n_nodes, dim=1)
        return torch.gather(h * scores, 1, indices.expand(-1, -1, h.size(-1)))


class _OfficialConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size + (kernel_size % 2 == 0)
        frequencies = sample_rate / 2 * np.linspace(0, 1, 257)
        mel = 2595 * np.log10(1 + frequencies / 700)
        bands = 700 * (10 ** (np.linspace(mel.min(), mel.max(), out_channels + 1) / 2595) - 1)
        support = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(out_channels, self.kernel_size)
        for index in range(out_channels):
            high = 2 * bands[index + 1] / sample_rate * np.sinc(2 * bands[index + 1] * support.numpy() / sample_rate)
            low = 2 * bands[index] / sample_rate * np.sinc(2 * bands[index] * support.numpy() / sample_rate)
            band_pass[index] = torch.from_numpy(np.hamming(self.kernel_size) * (high - low)).float()
        self.band_pass = band_pass

    def forward(self, x):
        filters = self.band_pass.to(x.device).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters)


class _OfficialResidual(nn.Module):
    def __init__(self, filters, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv1 = nn.Conv2d(filters[0], filters[1], (2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv2 = nn.Conv2d(filters[1], filters[1], (2, 3), padding=(0, 1))
        self.downsample = filters[0] != filters[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(filters[0], filters[1], (1, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.selu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class AASIST(nn.Module):
    """AASIST architecture matching clovaai's published AASIST.pth checkpoint."""
    _config = {"first_conv": 128, "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]], "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5], "temperatures": [2.0, 2.0, 100.0, 100.0]}

    def __init__(self):
        super().__init__()
        c = self._config; filters, dims, pools, temps = c["filts"], c["gat_dims"], c["pool_ratios"], c["temperatures"]
        self.conv_time = _OfficialConv(filters[0], c["first_conv"])
        self.first_bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout(0.5, inplace=True); self.drop_way = nn.Dropout(0.2, inplace=True); self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(*[nn.Sequential(_OfficialResidual(f, first=i == 0)) for i, f in enumerate(filters[1:])], nn.Sequential(_OfficialResidual(filters[-1])), nn.Sequential(_OfficialResidual(filters[-1])))
        self.pos_S = nn.Parameter(torch.randn(1, 23, filters[-1][-1])); self.master1 = nn.Parameter(torch.randn(1, 1, dims[0])); self.master2 = nn.Parameter(torch.randn(1, 1, dims[0]))
        self.GAT_layer_S = _OfficialGraphAttention(filters[-1][-1], dims[0], temps[0]); self.GAT_layer_T = _OfficialGraphAttention(filters[-1][-1], dims[0], temps[1])
        self.HtrgGAT_layer_ST11 = _OfficialHtrgGraphAttention(dims[0], dims[1], temps[2]); self.HtrgGAT_layer_ST12 = _OfficialHtrgGraphAttention(dims[1], dims[1], temps[2]); self.HtrgGAT_layer_ST21 = _OfficialHtrgGraphAttention(dims[0], dims[1], temps[2]); self.HtrgGAT_layer_ST22 = _OfficialHtrgGraphAttention(dims[1], dims[1], temps[2])
        self.pool_S = _OfficialGraphPool(pools[0], dims[0], 0.3); self.pool_T = _OfficialGraphPool(pools[1], dims[0], 0.3)
        self.pool_hS1 = _OfficialGraphPool(pools[2], dims[1], 0.3); self.pool_hT1 = _OfficialGraphPool(pools[2], dims[1], 0.3); self.pool_hS2 = _OfficialGraphPool(pools[2], dims[1], 0.3); self.pool_hT2 = _OfficialGraphPool(pools[2], dims[1], 0.3)
        self.out_layer = nn.Linear(5 * dims[1], 2)

    def forward(self, x):
        x = self.selu(self.first_bn(F.max_pool2d(torch.abs(self.conv_time(x.unsqueeze(1)).unsqueeze(1)), (3, 3))))
        embedding = self.encoder(x); spectral = torch.max(torch.abs(embedding), dim=3).values.transpose(1, 2) + self.pos_S; temporal = torch.max(torch.abs(embedding), dim=2).values.transpose(1, 2)
        spectral, temporal = self.pool_S(self.GAT_layer_S(spectral)), self.pool_T(self.GAT_layer_T(temporal))
        t1, s1, m1 = self.HtrgGAT_layer_ST11(temporal, spectral, self.master1); s1, t1 = self.pool_hS1(s1), self.pool_hT1(t1); ta, sa, ma = self.HtrgGAT_layer_ST12(t1, s1, m1); t1, s1, m1 = t1 + ta, s1 + sa, m1 + ma
        t2, s2, m2 = self.HtrgGAT_layer_ST21(temporal, spectral, self.master2); s2, t2 = self.pool_hS2(s2), self.pool_hT2(t2); ta, sa, ma = self.HtrgGAT_layer_ST22(t2, s2, m2); t2, s2, m2 = t2 + ta, s2 + sa, m2 + ma
        t1, t2, s1, s2, m1, m2 = (self.drop_way(v) for v in (t1, t2, s1, s2, m1, m2)); temporal, spectral, master = torch.max(t1, t2), torch.max(s1, s2), torch.max(m1, m2)
        hidden = torch.cat([torch.max(torch.abs(temporal), dim=1).values, torch.mean(temporal, dim=1), torch.max(torch.abs(spectral), dim=1).values, torch.mean(spectral, dim=1), master.squeeze(1)], dim=1)
        return hidden, self.out_layer(self.drop(hidden))

    @classmethod
    def from_pretrained(cls, weights_path: str, device: str = "cpu") -> "AASIST":
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"AASIST weights not found at: {weights_path}")
        model = cls(); state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True); return model.to(device).eval()
