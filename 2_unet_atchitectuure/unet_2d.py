"""
=============================================================================
功率→温度/应力预测 - UNet-V2 模型
=============================================================================
模型：
✅ UNet-V2 + EMA注意力

损失函数：
✅ 标准MSE (Mean Squared Error)

特点：
- 简洁高效的U-Net架构
- EMA多尺度注意力机制
- 4层编码器-解码器
- 完整跳跃连接

使用方法：
    from unet_2d import UNetV2_PowerToField
    
    # 创建模型（默认使用EMA注意力）
    model = UNetV2_PowerToField(in_channels=1, out_channels=1)
    
    # 使用标准MSE损失
    criterion = nn.MSELoss()
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== EMA 注意力模块 ====================

class EMA(nn.Module):
    """
    EMA (Efficient Multi-scale Attention) 注意力模块
    
    特点：
    - 多尺度交叉注意力
    - 高效的特征提取
    - 适合图像预测任务
    
    参数:
        in_channels: 输入通道数
        factor: 分组因子，用于控制计算复杂度 (默认32)
    """
    def __init__(self, in_channels, factor=32):
        super().__init__()
        self.groups = min(factor, in_channels)
        # 确保通道数能被groups整除
        while in_channels % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        group_channels = in_channels // self.groups
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        
        # 水平和垂直池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        
        # 1x1卷积融合
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        # 分支1：归一化 + sigmoid加权
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # 分支2：3x3卷积
        x2 = self.conv3x3(group_x)
        
        # 交叉注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        
        # 计算注意力权重
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        
        # 应用权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# ==================== 网络模块 ====================

class ConvBlock(nn.Module):
    """
    双卷积块 + EMA注意力
    
    结构：Conv-BN-ReLU-Conv-BN-ReLU-EMA
    
    参数:
        in_ch: 输入通道数
        out_ch: 输出通道数
        kernel_size: 卷积核大小 (默认3)
        padding: 填充大小 (默认1)
        use_attention: 是否使用EMA注意力 (默认True)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # EMA注意力
        if use_attention:
            self.attention = EMA(out_ch)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


class UNetV2_PowerToField(nn.Module):
    """
    U-Net v2 - 功率→温度/应力预测模型
    
    架构特点：
    ┌─────────────────────────────────────────────────────────┐
    │ 编码器 (4层下采样)                                        │
    │  ├─ Enc1: 1 → 32   + EMA  → Pool + Dropout(0.1)         │
    │  ├─ Enc2: 32 → 64  + EMA  → Pool + Dropout(0.1)         │
    │  ├─ Enc3: 64 → 128 + EMA  → Pool + Dropout(0.2)         │
    │  └─ Enc4: 128 → 256 + EMA → Pool + Dropout(0.2)         │
    │                                                           │
    │ 瓶颈层                                                    │
    │  └─ Bottleneck: 256 → 512 (无注意力)                     │
    │                                                           │
    │ 解码器 (4层上采样 + 跳跃连接)                             │
    │  ├─ Dec4: 512+256 → 256 + EMA                            │
    │  ├─ Dec3: 256+128 → 128 + EMA                            │
    │  ├─ Dec2: 128+64  → 64  + EMA                            │
    │  └─ Dec1: 64+32   → 32  + EMA                            │
    │                                                           │
    │ 输出层                                                    │
    │  └─ Out: 32 → 1                                          │
    └─────────────────────────────────────────────────────────┘
    
    参数:
        in_channels: 输入通道数 (默认 1 - 单通道功率图)
        out_channels: 输出通道数 (默认 1 - 单通道温度/应力图)
        base_ch: 基础通道数 (默认 32)
        use_attention: 是否使用EMA注意力 (默认 True)
    
    输入输出:
        输入: [B, 1, H, W] - 功率分布图
        输出: [B, 1, H, W] - 温度/应力分布图
    
    示例:
        >>> model = UNetV2_PowerToField()
        >>> x = torch.randn(2, 1, 256, 256)  # 2个样本, 256x256分辨率
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([2, 1, 256, 256])
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, use_attention=True):
        super().__init__()
        
        print(f"\n{'='*60}")
        print(f"创建 UNet-V2 模型")
        print(f"{'='*60}")
        print(f"  输入通道: {in_channels}")
        print(f"  输出通道: {out_channels}")
        print(f"  基础通道: {base_ch}")
        print(f"  EMA注意力: {'启用' if use_attention else '禁用'}")
        
        # ========== 编码器 (Encoder) ==========
        self.enc1 = ConvBlock(in_channels, base_ch, use_attention=use_attention)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.enc2 = ConvBlock(base_ch, base_ch * 2, use_attention=use_attention)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.1)
        
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, use_attention=use_attention)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, use_attention=use_attention)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout2d(0.2)
        
        # ========== 瓶颈层 (Bottleneck) ==========
        # 最深层不使用注意力，减少计算量
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, use_attention=False)
        
        # ========== 解码器 (Decoder) ==========
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8, use_attention=use_attention)
        
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, use_attention=use_attention)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, use_attention=use_attention)
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, use_attention=use_attention)
        
        # ========== 输出层 ==========
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        
        # 计算参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  总参数量: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入功率图 [B, 1, H, W]
        
        返回:
            out: 预测温度/应力图 [B, 1, H, W]
        """
        # ========== 编码路径 (Downsampling) ==========
        # 逐层提取特征并降低分辨率
        e1 = self.enc1(x)                    # [B, 32, H, W]
        p1 = self.dropout1(self.pool1(e1))   # [B, 32, H/2, W/2]
        
        e2 = self.enc2(p1)                   # [B, 64, H/2, W/2]
        p2 = self.dropout2(self.pool2(e2))   # [B, 64, H/4, W/4]
        
        e3 = self.enc3(p2)                   # [B, 128, H/4, W/4]
        p3 = self.dropout3(self.pool3(e3))   # [B, 128, H/8, W/8]
        
        e4 = self.enc4(p3)                   # [B, 256, H/8, W/8]
        p4 = self.dropout4(self.pool4(e4))   # [B, 256, H/16, W/16]
        
        # ========== 瓶颈层 ==========
        b = self.bottleneck(p4)              # [B, 512, H/16, W/16]
        
        # ========== 解码路径 (Upsampling + Skip Connections) ==========
        # 逐层恢复分辨率并融合编码器特征
        
        # Level 4
        u4 = self.up4(b)                     # [B, 256, H/8, W/8]
        if u4.shape[-2:] != e4.shape[-2:]:
            u4 = F.interpolate(u4, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))  # [B, 256, H/8, W/8]
        
        # Level 3
        u3 = self.up3(d4)                    # [B, 128, H/4, W/4]
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # [B, 128, H/4, W/4]
        
        # Level 2
        u2 = self.up2(d3)                    # [B, 64, H/2, W/2]
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # [B, 64, H/2, W/2]
        
        # Level 1
        u1 = self.up1(d2)                    # [B, 32, H, W]
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # [B, 32, H, W]
        
        # ========== 输出层 ==========
        out = self.out_conv(d1)              # [B, 1, H, W]
        
        return out


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("="*60)
    print("测试 UNet-V2 模型")
    print("="*60)
    
    # 创建模型
    model = UNetV2_PowerToField(in_channels=1, out_channels=1, base_ch=32)
    
    # 测试前向传播
    x = torch.randn(2, 1, 256, 256)
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    print(f"✓ 前向传播测试通过！")
    
    # 测试损失计算
    criterion = nn.MSELoss()
    target = torch.randn(2, 1, 256, 256)
    loss = criterion(y, target)
    print(f"\nMSE损失: {loss.item():.6f}")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)