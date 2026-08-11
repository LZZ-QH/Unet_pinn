#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact U-Net models for power-map to fatigue-variable field prediction."""

from __future__ import annotations

import torch
from torch import nn


class EMA(nn.Module):
    """Efficient multi-scale attention block used by the earlier U-Net baseline."""

    def __init__(self, channels: int, factor: int = 32) -> None:
        super().__init__()
        groups = min(factor, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        group_channels = channels // groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = EMA(out_ch) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.conv(x))


class UNetPowerToFatigue(nn.Module):
    """Four-level U-Net with skip connections and optional EMA attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        base_ch: int = 32,
        use_attention: bool = True,
        dropout: bool = True,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch, use_attention=use_attention)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, use_attention=use_attention)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, use_attention=use_attention)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, use_attention=use_attention)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, use_attention=False)

        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1) if dropout else nn.Identity()
        self.drop2 = nn.Dropout2d(0.1) if dropout else nn.Identity()
        self.drop3 = nn.Dropout2d(0.2) if dropout else nn.Identity()
        self.drop4 = nn.Dropout2d(0.2) if dropout else nn.Identity()

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8, use_attention=use_attention)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, use_attention=use_attention)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, use_attention=use_attention)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, use_attention=use_attention)
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.drop1(self.pool(e1)))
        e3 = self.enc3(self.drop2(self.pool(e2)))
        e4 = self.enc4(self.drop3(self.pool(e3)))
        b = self.bottleneck(self.drop4(self.pool(e4)))

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out_conv(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
