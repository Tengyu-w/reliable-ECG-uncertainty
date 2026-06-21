from __future__ import annotations

import torch
from torch import nn


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
    if name in {"resnet", "resnet1d"}:
        return ResNet1D(num_classes=num_classes)
    if name in {"regularity_fusion", "fusion", "regularity_resnet"}:
        return RegularityFusionResNet1D(num_classes=num_classes, feature_dim=feature_dim)
    if name in {"reliability_gated_fusion", "gated_fusion", "rgrf"}:
        return ReliabilityGatedRegularityFusion(num_classes=num_classes, feature_dim=feature_dim)
    raise ValueError(f"Unknown model: {name}")
