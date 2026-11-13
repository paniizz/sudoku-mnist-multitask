"""Compact multi-task CNN (<20k params) for Sudoku-MNIST grids."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pw(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class ResidualDSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip is not None:
            identity = self.skip(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class SudokuNet(nn.Module):
    """Shared backbone with three task heads.

    Outputs
    -------
    missing_digit_logits : (B, 10)
    sorted_logits        : (B, 6, 3)  — 3 rows + 3 cols, 3 sort classes
    sum_preds            : (B, 6)     — sigmoid-normalized row/col sums
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.backbone = nn.Sequential(
            ResidualDSBlock(16, 16, stride=1),
            ResidualDSBlock(16, 24, stride=2),
            ResidualDSBlock(24, 24, stride=1),
            ResidualDSBlock(24, 32, stride=2),
            ResidualDSBlock(32, 32, stride=1),
            ResidualDSBlock(32, 32, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 32

        self.head_missing = nn.Linear(feat_dim, 10)
        self.head_sorted = nn.Linear(feat_dim, 18)
        self.head_sums = nn.Linear(feat_dim, 6)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)

        missing_digit_logits = self.head_missing(x)
        sorted_logits = self.head_sorted(x).view(-1, 6, 3)
        sum_preds = torch.sigmoid(self.head_sums(x))

        return {
            "missing_digit_logits": missing_digit_logits,
            "sorted_logits": sorted_logits,
            "sum_preds": sum_preds,
        }
