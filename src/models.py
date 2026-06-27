from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGCNN(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, 7),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.encoder(x).squeeze(-1)
        emb = torch.relu(self.embedding(z))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class TCN(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        channels = [1, 32, 64, 64, 128]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(ConvBlock(channels[i], channels[i + 1], kernel_size=5, dilation=2**i))
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.encoder(x).squeeze(-1)
        emb = torch.relu(self.embedding(z))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.encoder(self.stem(x)).squeeze(-1)
        emb = torch.relu(self.embedding(z))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32, bottleneck_channels: int = 32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck
            else nn.Identity()
        )
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels
        self.branch9 = nn.Conv1d(branch_in, out_channels, kernel_size=9, padding=4, bias=False)
        self.branch19 = nn.Conv1d(branch_in, out_channels, kernel_size=19, padding=9, bias=False)
        self.branch39 = nn.Conv1d(branch_in, out_channels, kernel_size=39, padding=19, bias=False)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        out = torch.cat([self.branch9(z), self.branch19(z), self.branch39(z), self.pool_branch(x)], dim=1)
        return self.relu(self.norm(out))


class InceptionTime1D(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.block1 = InceptionModule1D(1, out_channels=24, bottleneck_channels=16)
        self.block2 = InceptionModule1D(96, out_channels=32, bottleneck_channels=32)
        self.shortcut = nn.Sequential(nn.Conv1d(1, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.block1(x)
        z = self.block2(z)
        z = self.relu(z + self.shortcut(x))
        z = self.pool(z).squeeze(-1)
        emb = torch.relu(self.embedding(z))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class BiGRUClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, hidden_size: int = 48):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        sequence = self.downsample(x).transpose(1, 2)
        _, h = self.gru(sequence)
        z = torch.cat([h[-2], h[-1]], dim=1)
        emb = self.embedding(z)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, hidden_size: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, kernel_size=9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBlock(64, 64, kernel_size=5),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        sequence = self.cnn(x).transpose(1, 2)
        _, (h, _) = self.lstm(sequence)
        z = torch.cat([h[-2], h[-1]], dim=1)
        emb = self.embedding(z)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class FixedWaveletFilterBank1D(nn.Module):
    """Fixed multi-scale wavelet-like filters for explicit ECG time-frequency cues."""

    def __init__(self, scales: tuple[float, ...] = (2.0, 4.0, 8.0), kernel_size: int = 31):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length wavelet filtering.")
        half = kernel_size // 2
        t = torch.linspace(-half, half, steps=kernel_size)
        filters = []
        for scale in scales:
            ts = t / scale
            gaussian = torch.exp(-0.5 * ts.pow(2))
            atoms = [
                (1.0 - ts.pow(2)) * gaussian,
                -ts * gaussian,
                torch.sin(2.0 * math.pi * ts) * gaussian,
            ]
            for atom in atoms:
                atom = atom - atom.mean()
                atom = atom / atom.norm(p=2).clamp_min(1e-6)
                filters.append(atom)
        weight = torch.stack(filters).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.padding = half
        self.out_channels = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = F.conv1d(x, self.weight.to(dtype=x.dtype), padding=self.padding)
        return torch.log1p(response.abs())


class CNNTCNValidityBottleneck(nn.Module):
    """Morphology-rhythm model whose decision path is gated by a validity bottleneck."""

    def __init__(
        self,
        num_classes: int = 3,
        embedding_dim: int = 128,
        bottleneck_dim: int = 16,
        tcn_channels: int = 96,
    ):
        super().__init__()
        self.morphology = nn.Sequential(
            ConvBlock(1, 32, kernel_size=9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBlock(64, tcn_channels, kernel_size=5),
        )
        self.rhythm = nn.Sequential(
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=1),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=2),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=4),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Linear(tcn_channels * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dim),
        )
        self.validity_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Linear(bottleneck_dim, 1)
        self.gate_head = nn.Sequential(nn.Linear(bottleneck_dim, 1), nn.Sigmoid())
        self.main_classifier = nn.Linear(embedding_dim, num_classes)
        self.boundary_adapter = nn.Sequential(
            nn.Linear(embedding_dim + bottleneck_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_gate: bool = False,
        return_bottleneck: bool = False,
    ):
        morph_map = self.morphology(x)
        rhythm_map = self.rhythm(morph_map)
        morph_vec = self.pool(morph_map).squeeze(-1)
        rhythm_vec = self.pool(rhythm_map).squeeze(-1)
        emb = self.embedding(torch.cat([morph_vec, rhythm_vec], dim=1))
        bottleneck = self.validity_bottleneck(emb)
        gate = self.gate_head(bottleneck).squeeze(1)
        boundary_logit = self.boundary_head(bottleneck).squeeze(1)
        main_logits = self.main_classifier(emb)
        adapter_logits = self.boundary_adapter(torch.cat([emb, bottleneck], dim=1))
        logits = main_logits + gate.unsqueeze(1) * adapter_logits

        outputs: list[torch.Tensor] = [logits]
        if return_embedding:
            outputs.append(emb)
        if return_gate:
            outputs.extend([gate, boundary_logit])
        if return_bottleneck:
            outputs.append(bottleneck)
        if len(outputs) > 1:
            return tuple(outputs)
        return logits


class CNNTCNValidityMixtureOfExperts(nn.Module):
    """CNN+TCN validity model with a decision-time VT/VF specialist mixture."""

    def __init__(
        self,
        num_classes: int = 3,
        embedding_dim: int = 128,
        bottleneck_dim: int = 16,
        tcn_channels: int = 96,
    ):
        super().__init__()
        if num_classes != 3:
            raise ValueError("CNNTCNValidityMixtureOfExperts currently expects SR/VT/VF classes.")
        self.morphology = nn.Sequential(
            ConvBlock(1, 32, kernel_size=9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBlock(64, tcn_channels, kernel_size=5),
        )
        self.rhythm = nn.Sequential(
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=1),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=2),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=4),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Linear(tcn_channels * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dim),
        )
        self.validity_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Linear(bottleneck_dim, 1)
        self.gate_head = nn.Sequential(nn.Linear(bottleneck_dim, 1), nn.Sigmoid())
        self.main_classifier = nn.Linear(embedding_dim, num_classes)
        self.vtvf_specialist = nn.Sequential(
            nn.Linear(embedding_dim + bottleneck_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_gate: bool = False,
        return_aux: bool = False,
    ):
        morph_map = self.morphology(x)
        rhythm_map = self.rhythm(morph_map)
        morph_vec = self.pool(morph_map).squeeze(-1)
        rhythm_vec = self.pool(rhythm_map).squeeze(-1)
        emb = self.embedding(torch.cat([morph_vec, rhythm_vec], dim=1))
        bottleneck = self.validity_bottleneck(emb)
        gate = self.gate_head(bottleneck).squeeze(1)
        boundary_logit = self.boundary_head(bottleneck).squeeze(1)
        main_logits = self.main_classifier(emb)
        specialist_logits = self.vtvf_specialist(torch.cat([emb, bottleneck], dim=1))
        logits = main_logits.clone()
        logits[:, 1:3] = (1.0 - gate.unsqueeze(1)) * main_logits[:, 1:3] + gate.unsqueeze(1) * specialist_logits

        outputs: list[torch.Tensor | dict[str, torch.Tensor]] = [logits]
        if return_embedding:
            outputs.append(emb)
        if return_gate:
            outputs.extend([gate, boundary_logit])
        if return_aux:
            outputs.append(
                {
                    "main_logits": main_logits,
                    "vtvf_specialist_logits": specialist_logits,
                    "validity_bottleneck": bottleneck,
                }
            )
        if len(outputs) > 1:
            return tuple(outputs)
        return logits


class CNNWaveletTCNBoundaryAdapter(nn.Module):
    """CNN+Wavelet+TCN validity model with a VT/VF specialist boundary adapter."""

    def __init__(
        self,
        num_classes: int = 3,
        embedding_dim: int = 128,
        bottleneck_dim: int = 16,
        tcn_channels: int = 80,
    ):
        super().__init__()
        if num_classes != 3:
            raise ValueError("CNNWaveletTCNBoundaryAdapter currently expects SR/VT/VF classes.")
        raw_channels = 48
        wavelet_channels = 48
        self.morphology = nn.Sequential(
            ConvBlock(1, 32, kernel_size=9),
            nn.MaxPool1d(2),
            ConvBlock(32, 40, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBlock(40, raw_channels, kernel_size=5),
        )
        self.wavelet_bank = FixedWaveletFilterBank1D()
        self.wavelet_encoder = nn.Sequential(
            ConvBlock(self.wavelet_bank.out_channels, 24, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBlock(24, 40, kernel_size=5),
            nn.MaxPool1d(2),
            ConvBlock(40, wavelet_channels, kernel_size=5),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(raw_channels + wavelet_channels, tcn_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU(inplace=True),
        )
        self.rhythm = nn.Sequential(
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=1),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=2),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=4),
            TemporalResidualBlock(tcn_channels, kernel_size=5, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Linear(raw_channels + wavelet_channels + tcn_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dim),
        )
        self.validity_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Linear(bottleneck_dim, 1)
        self.gate_head = nn.Sequential(nn.Linear(bottleneck_dim, 1), nn.Sigmoid())
        self.main_classifier = nn.Linear(embedding_dim, num_classes)
        self.vtvf_specialist = nn.Sequential(
            nn.Linear(embedding_dim + bottleneck_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_gate: bool = False,
        return_aux: bool = False,
    ):
        morph_map = self.morphology(x)
        wavelet_map = self.wavelet_encoder(self.wavelet_bank(x))
        fused_map = self.fusion(torch.cat([morph_map, wavelet_map], dim=1))
        rhythm_map = self.rhythm(fused_map)
        morph_vec = self.pool(morph_map).squeeze(-1)
        wavelet_vec = self.pool(wavelet_map).squeeze(-1)
        rhythm_vec = self.pool(rhythm_map).squeeze(-1)
        emb = self.embedding(torch.cat([morph_vec, wavelet_vec, rhythm_vec], dim=1))
        bottleneck = self.validity_bottleneck(emb)
        gate = self.gate_head(bottleneck).squeeze(1)
        boundary_logit = self.boundary_head(bottleneck).squeeze(1)
        main_logits = self.main_classifier(emb)
        specialist_logits = self.vtvf_specialist(torch.cat([emb, bottleneck], dim=1))
        logits = main_logits.clone()
        logits[:, 1:3] = (1.0 - gate.unsqueeze(1)) * main_logits[:, 1:3] + gate.unsqueeze(1) * specialist_logits

        outputs: list[torch.Tensor | dict[str, torch.Tensor]] = [logits]
        if return_embedding:
            outputs.append(emb)
        if return_gate:
            outputs.extend([gate, boundary_logit])
        if return_aux:
            outputs.append(
                {
                    "main_logits": main_logits,
                    "vtvf_specialist_logits": specialist_logits,
                    "validity_bottleneck": bottleneck,
                }
            )
        if len(outputs) > 1:
            return tuple(outputs)
        return logits


class RegularityFusionResNet1D(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, feature_dim: int = 9):
        super().__init__()
        self.feature_dim = feature_dim
        self.waveform = ResNet1D(num_classes=num_classes, embedding_dim=embedding_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.fused_embedding = nn.Linear(embedding_dim + 32, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None, return_embedding: bool = False):
        _, wave_emb = self.waveform(x, return_embedding=True)
        if features is None:
            features = torch.zeros((x.shape[0], self.feature_dim), device=x.device, dtype=x.dtype)
        feature_emb = self.feature_encoder(features)
        emb = torch.relu(self.fused_embedding(torch.cat([wave_emb, feature_emb], dim=1)))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class ReliabilityGatedRegularityFusion(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, feature_dim: int = 9):
        super().__init__()
        self.feature_dim = feature_dim
        self.waveform = ResNet1D(num_classes=num_classes, embedding_dim=embedding_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.reliability_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.boundary_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        features: torch.Tensor | None = None,
        return_embedding: bool = False,
        return_gate: bool = False,
    ):
        _, wave_emb = self.waveform(x, return_embedding=True)
        if features is None:
            features = torch.zeros((x.shape[0], self.feature_dim), device=x.device, dtype=x.dtype)
        feature_emb = self.feature_encoder(features)
        gate = self.reliability_gate(torch.cat([wave_emb, feature_emb], dim=1))
        emb = self.norm(wave_emb + gate * feature_emb)
        logits = self.classifier(emb)
        if return_gate:
            boundary_logit = self.boundary_head(emb).squeeze(1)
            if return_embedding:
                return logits, emb, gate.squeeze(1), boundary_logit
            return logits, gate.squeeze(1), boundary_logit
        if return_embedding:
            return logits, emb
        return logits


def build_model(name: str, num_classes: int = 3, feature_dim: int = 9) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return ECGCNN(num_classes=num_classes)
    if name == "tcn":
        return TCN(num_classes=num_classes)
    if name in {"inception_time", "inceptiontime", "inception"}:
        return InceptionTime1D(num_classes=num_classes)
    if name in {"bigru", "gru"}:
        return BiGRUClassifier(num_classes=num_classes)
    if name in {"cnn_lstm", "cnnlstm", "cnn-lstm"}:
        return CNNLSTMClassifier(num_classes=num_classes)
    if name in {"cnn_tcn_validity", "cnn-tcn-validity", "ctv", "validity_tcn"}:
        return CNNTCNValidityBottleneck(num_classes=num_classes)
    if name in {"cnn_tcn_validity_v2", "cnn-tcn-validity-v2", "ctv2", "validity_tcn_v2"}:
        return CNNTCNValidityMixtureOfExperts(num_classes=num_classes)
    if name in {
        "cnn_wavelet_tcn_boundary",
        "cnn-wavelet-tcn-boundary",
        "wavelet_tcn_validity",
        "wavelet_boundary",
        "cwtb",
    }:
        return CNNWaveletTCNBoundaryAdapter(num_classes=num_classes)
    if name in {"resnet", "resnet1d"}:
        return ResNet1D(num_classes=num_classes)
    if name in {"regularity_fusion", "fusion", "regularity_resnet"}:
        return RegularityFusionResNet1D(num_classes=num_classes, feature_dim=feature_dim)
    if name in {"reliability_gated_fusion", "gated_fusion", "rgrf"}:
        return ReliabilityGatedRegularityFusion(num_classes=num_classes, feature_dim=feature_dim)
    raise ValueError(f"Unknown model: {name}")
