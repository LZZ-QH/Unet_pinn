#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
功率→温度+应力 联合评估脚本 (v13 - 最大值位置误差)
=============================================================================
修复内容：
✅ 【v13】改用方法1：在真实最大值位置比较预测误差
✅ 【v13】去掉P99标注，只标注真实最大值位置
✅ 【v13】显示：全场MAE + 最大值位置的预测误差
✅ 【v12】适配新的统一模型架构 UNetV2_PowerToField
✅ 【v11】统一所有子图坐标范围为芯片边界 X[-40,40] Y[-15,15]
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['axes.unicode_minus'] = False

# 彩色输出
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


class ColorPrinter:
    @staticmethod
    def header(text):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*70}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{text}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*70}{Style.RESET_ALL}")
    
    @staticmethod
    def success(text):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def info(text):
        print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(text):
        print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

cp = ColorPrinter()


# =============================================================================
# 配置
# =============================================================================

class Config:
    """评估配置"""
    
    # 样本ID
    sample_id = 2002
    
    # 模型架构路径
    model_dir = "/home/txr/code/final_version/2_unet_architecture"
    
    # 温度模型配置
    temp_model_path = "/home/txr/code/final_version/2_unet_architecture/training_logs/power_to_temperature_prediction/unet_power_to_temperature_final_20251225_094913.pth"
    temp_cache_dir = "/home/txr/code/final_version/2_unet_architecture/cache/power_to_temperature"
    temp_cache_pattern = "power_temp_{:04d}.pkl"
    
    # 应力模型配置
    stress_model_path = "/home/txr/code/final_version/2_unet_architecture/training_logs/power_to_stress_prediction/cache/unet_mask_weighted_power2stress_final_20251224_113026.pth"
    stress_cache_dir = "/home/txr/code/final_version/2_unet_architecture/cache/power_to_stress"
    stress_cache_pattern = "power_stress_{:04d}.pkl"
    
    # 输出配置
    output_dir = "/home/txr/code/final_version/2_unet_architecture/evaluation_result"
    dpi = 600
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 统一物理坐标配置
    unified_x_min = -40.0
    unified_x_max = 40.0
    unified_y_min = -15.0
    unified_y_max = 15.0


# =============================================================================
# 数据加载
# =============================================================================

def load_cache_data(cache_path: str):
    """加载缓存数据"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple) and len(data) == 2:
        return data[0], data[1]
    else:
        raise ValueError(f"未知的缓存格式: {type(data)}")


def load_unet_model(model_path: str, model_dir: str, device: str):
    """加载UNet模型（兼容新旧架构）"""
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    from unet_2d import UNetV2_PowerToField
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = UNetV2_PowerToField(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        base_ch=config.get('base_ch', 32),
        use_attention=config.get('use_attention', True)
    )
    
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 兼容旧版架构
    if any('.attention.attention.' in k for k in state_dict.keys()):
        print(f"  检测到旧版架构权重，自动转换键名...")
        state_dict = {k.replace('.attention.attention.', '.attention.'): v 
                      for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    p_stats = checkpoint.get('p_stats', {})
    
    if 't_stats' in checkpoint:
        output_stats = checkpoint['t_stats']
        model_type = '温度'
    elif 's_stats' in checkpoint:
        output_stats = checkpoint['s_stats']
        model_type = '应力'
    else:
        raise ValueError("Checkpoint中缺少输出统计信息")
    
    return model, p_stats, output_stats, model_type


def normalize_power(power: np.ndarray, p_stats: dict) -> np.ndarray:
    """功率归一化"""
    median = p_stats.get('median', 0)
    iqr = p_stats.get('iqr', 1)
    if iqr < 1e-8:
        iqr = 1.0
    
    power_norm = np.zeros_like(power, dtype=np.float32)
    nonzero_mask = power > 1e-6
    if nonzero_mask.any():
        power_norm[nonzero_mask] = (power[nonzero_mask] - median) / iqr
    
    return power_norm


def denormalize_output(output_norm: np.ndarray, stats: dict) -> np.ndarray:
    """输出反归一化"""
    mean = stats['mean']
    std = stats['std']
    return output_norm * std + mean


def predict(model, power: np.ndarray, p_stats: dict, output_stats: dict, device: str):
    """执行预测"""
    power_norm = normalize_power(power, p_stats)
    power_t = torch.from_numpy(power_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_norm = model(power_t)
    
    pred = denormalize_output(pred_norm.cpu().squeeze().numpy(), output_stats)
    return pred


# =============================================================================
# 辅助函数
# =============================================================================

def create_error_colormap():
    """使用matplotlib内置的Blues色图 - 最简洁清晰"""
    return plt.cm.Blues


def pixel_to_physical(pixel_x, pixel_y, img_size, x_min, x_max, y_min, y_max):
    """将像素坐标转换为物理坐标"""
    phys_x = x_min + (pixel_x / (img_size - 1)) * (x_max - x_min)
    phys_y = y_min + (pixel_y / (img_size - 1)) * (y_max - y_min)
    return phys_x, phys_y


def get_text_position(point_x, point_y, x_min, x_max, y_min, y_max, offset_ratio=0.08):
    """智能计算文本位置"""
    x_range = x_max - x_min
    y_range = y_max - y_min
    offset_x = x_range * offset_ratio
    offset_y = y_range * offset_ratio
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    if point_x > center_x:
        text_x = point_x - offset_x
        ha = 'right'
    else:
        text_x = point_x + offset_x
        ha = 'left'
    
    if point_y > center_y:
        text_y = point_y - offset_y
        va = 'top'
    else:
        text_y = point_y + offset_y
        va = 'bottom'
    
    # 边界检查
    margin = 0.08
    if text_x < x_min + x_range * margin:
        text_x = x_min + x_range * margin
        ha = 'left'
    elif text_x > x_max - x_range * margin:
        text_x = x_max - x_range * margin
        ha = 'right'
    
    if text_y < y_min + y_range * margin:
        text_y = y_min + y_range * margin
        va = 'bottom'
    elif text_y > y_max - y_range * margin:
        text_y = y_max - y_range * margin
        va = 'top'
    
    return text_x, text_y, ha, va


# =============================================================================
# 可视化 - v13: 最大值位置误差
# =============================================================================

def create_joint_visualization_horizontal(
    power_display: np.ndarray,
    true_temp: np.ndarray,
    pred_temp: np.ndarray,
    true_stress: np.ndarray,
    pred_stress: np.ndarray,
    sample_id: int,
    output_path: str,
    dpi: int = 600
):
    """
    创建横排3×3联合可视化（v13: 最大值位置误差）
    
    【v13 关键改动】
    - 找到真实最大值的位置
    - 在该位置比较预测值和真实值
    - 误差 = |pred[max_loc] - true[max_loc]|
    """
    
    cp.header(f"生成联合可视化（v13 - 最大值位置误差）- Sample {sample_id}")
    
    config = Config()
    
    # 计算逐像素误差（用于MAE和误差图显示）
    temp_error = np.abs(pred_temp - true_temp)
    stress_error_pa = np.abs(pred_stress - true_stress)
    stress_error_mpa = stress_error_pa / 1e6
    
    # 转换应力为MPa
    true_stress_mpa = true_stress / 1e6
    pred_stress_mpa = pred_stress / 1e6
    
    # ============================================================
    # 【v13】方法1：在真实最大值位置计算误差
    # ============================================================
    
    # 温度：找到真实最大温度的位置
    temp_true_max_idx = np.unravel_index(np.argmax(true_temp), true_temp.shape)
    temp_true_max = true_temp[temp_true_max_idx]  # 真实最大温度
    temp_pred_at_max = pred_temp[temp_true_max_idx]  # 该位置的预测温度
    temp_max_loc_error = abs(temp_pred_at_max - temp_true_max)  # 最大值位置误差
    
    # 应力：找到真实最大应力的位置
    stress_true_max_idx = np.unravel_index(np.argmax(true_stress_mpa), true_stress_mpa.shape)
    stress_true_max = true_stress_mpa[stress_true_max_idx]  # 真实最大应力
    stress_pred_at_max = pred_stress_mpa[stress_true_max_idx]  # 该位置的预测应力
    stress_max_loc_error = abs(stress_pred_at_max - stress_true_max)  # 最大值位置误差
    
    # 全场MAE
    temp_mae = temp_error.mean()
    stress_mae_mpa = stress_error_mpa.mean()
    
    # R²
    temp_r2 = r2_score(true_temp.flatten(), pred_temp.flatten())
    stress_r2 = r2_score(true_stress_mpa.flatten(), pred_stress_mpa.flatten())
    
    # 获取图片尺寸
    img_height, img_width = temp_error.shape
    
    # 统一坐标范围
    unified_extent = [config.unified_x_min, config.unified_x_max,
                      config.unified_y_min, config.unified_y_max]
    
    print(f"\n  【v13 方法1统计】")
    print(f"  温度:")
    print(f"    全场MAE: {temp_mae:.4f} K")
    print(f"    真实最大温度: {temp_true_max:.2f} K (位置: {temp_true_max_idx})")
    print(f"    该位置预测温度: {temp_pred_at_max:.2f} K")
    print(f"    最大值位置误差: {temp_max_loc_error:.4f} K")
    print(f"    R²: {temp_r2:.6f}")
    
    print(f"  应力:")
    print(f"    全场MAE: {stress_mae_mpa:.4f} MPa")
    print(f"    真实最大应力: {stress_true_max:.2f} MPa (位置: {stress_true_max_idx})")
    print(f"    该位置预测应力: {stress_pred_at_max:.2f} MPa")
    print(f"    最大值位置误差: {stress_max_loc_error:.4f} MPa")
    print(f"    R²: {stress_r2:.4f}")
    
    # ============================================================
    # 字体大小配置
    # ============================================================
    TITLE_FONTSIZE = 15
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    CBAR_LABEL_FONTSIZE = 12
    STATS_FONTSIZE = 12
    MARKER_SIZE = 22
    ANNOTATION_FONTSIZE = 12
    SUBLABEL_FONTSIZE = 18
    BOX_ALPHA = 0.9
    
    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), dpi=dpi)
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.95, 
                        hspace=0.45, wspace=0.35)
    
    error_cmap = create_error_colormap()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    # ==================== 第1行：功率 + 两个R²散点图 ====================
    
    # (0,0) 功率分布
    ax = axes[0, 0]
    im = ax.imshow(power_display, cmap='hot', origin='lower', interpolation='nearest',
                   extent=unified_extent, aspect='auto')
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Power Density (W/m³)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # (0,1) 温度R²散点图
    ax = axes[0, 1]
    true_flat = true_temp.flatten()
    pred_flat = pred_temp.flatten()
    n_points = min(5000, len(true_flat))
    indices = np.random.choice(len(true_flat), n_points, replace=False)
    
    ax.scatter(true_flat[indices], pred_flat[indices], alpha=0.3, s=2, c='steelblue')
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('True Temperature (K)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Predicted Temperature (K)', fontsize=LABEL_FONTSIZE)
    ax.set_title(f'R² = {temp_r2:.6f}', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(fontsize=STATS_FONTSIZE, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    
    # 显示全场MAE和最大值位置误差
    stats_text = f"Field MAE: {temp_mae:.4f} K\nMax Loc Error: {temp_max_loc_error:.4f} K"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=STATS_FONTSIZE,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # (0,2) 应力R²散点图
    ax = axes[0, 2]
    true_stress_flat = true_stress_mpa.flatten()
    pred_stress_flat = pred_stress_mpa.flatten()
    indices_s = np.random.choice(len(true_stress_flat), n_points, replace=False)
    
    ax.scatter(true_stress_flat[indices_s], pred_stress_flat[indices_s], alpha=0.3, s=2, c='mediumseagreen')
    min_val = min(true_stress_flat.min(), pred_stress_flat.min())
    max_val = max(true_stress_flat.max(), pred_stress_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('True Stress (MPa)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Predicted Stress (MPa)', fontsize=LABEL_FONTSIZE)
    ax.set_title(f'R² = {stress_r2:.4f}', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(fontsize=STATS_FONTSIZE, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    
    stats_text = f"Field MAE: {stress_mae_mpa:.4f} MPa\nMax Loc Error: {stress_max_loc_error:.4f} MPa"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=STATS_FONTSIZE,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ==================== 第2行：温度（真实、预测、误差）====================
    
    # (1,0) 真实温度
    ax = axes[1, 0]
    im = ax.imshow(true_temp, cmap='jet', origin='lower', interpolation='nearest',
                   extent=unified_extent, aspect='auto')
    ax.set_title(f'True: [{true_temp.min():.1f}, {true_temp.max():.1f}] K', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (K)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # (1,1) 预测温度
    ax = axes[1, 1]
    im = ax.imshow(pred_temp, cmap='jet', origin='lower', interpolation='nearest',
                   vmin=true_temp.min(), vmax=true_temp.max(),
                   extent=unified_extent, aspect='auto')
    ax.set_title(f'Predict: [{pred_temp.min():.1f}, {pred_temp.max():.1f}] K', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (K)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # (1,2) 温度误差图 + 标注真实最大值位置
    ax = axes[1, 2]
    temp_vmax = np.percentile(temp_error, 99)  # 用P99作为色标上限，避免极端值影响
    
    im = ax.imshow(temp_error, cmap=error_cmap, origin='lower', interpolation='nearest',
                   vmin=0, vmax=temp_vmax, extent=unified_extent, aspect='auto')
    
    # 将真实最大值位置的像素坐标转换为物理坐标
    temp_max_phys_x, temp_max_phys_y = pixel_to_physical(
        temp_true_max_idx[1], temp_true_max_idx[0], img_height,
        config.unified_x_min, config.unified_x_max, config.unified_y_min, config.unified_y_max
    )
    
    # 标注真实最大值位置（红色星号）
    ax.plot(temp_max_phys_x, temp_max_phys_y, 'r*', markersize=MARKER_SIZE, 
            markeredgecolor='darkred', markeredgewidth=1.5)
    
    text_x, text_y, ha, va = get_text_position(
        temp_max_phys_x, temp_max_phys_y, 
        config.unified_x_min, config.unified_x_max, config.unified_y_min, config.unified_y_max
    )
    
    # 标注信息：真实值、预测值、误差
    annotation = f'True: {temp_true_max:.2f} K\nPred: {temp_pred_at_max:.2f} K\nError: {temp_max_loc_error:.2f} K'
    ax.text(text_x, text_y, annotation, 
            fontsize=ANNOTATION_FONTSIZE, color='darkred',  ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred', alpha=BOX_ALPHA))
    
    ax.set_title(f'Field MAE: {temp_mae:.4f} K | Max Loc Error: {temp_max_loc_error:.2f} K', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error (K)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # ==================== 第3行：应力（真实、预测、误差）====================
    
    # (2,0) 真实应力
    ax = axes[2, 0]
    im = ax.imshow(true_stress_mpa, cmap='viridis', origin='lower', interpolation='nearest',
                   extent=unified_extent, aspect='auto')
    ax.set_title(f'True: [{true_stress_mpa.min():.1f}, {true_stress_mpa.max():.1f}] MPa', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Stress (MPa)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # (2,1) 预测应力
    ax = axes[2, 1]
    im = ax.imshow(pred_stress_mpa, cmap='viridis', origin='lower', interpolation='nearest',
                   vmin=true_stress_mpa.min(), vmax=true_stress_mpa.max(),
                   extent=unified_extent, aspect='auto')
    ax.set_title(f'Predict: [{pred_stress_mpa.min():.1f}, {pred_stress_mpa.max():.1f}] MPa', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Stress (MPa)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # (2,2) 应力误差图 + 标注真实最大值位置
    ax = axes[2, 2]
    stress_vmax = np.percentile(stress_error_mpa, 99)
    
    im = ax.imshow(stress_error_mpa, cmap=error_cmap, origin='lower', interpolation='nearest',
                   vmin=0, vmax=stress_vmax, extent=unified_extent, aspect='auto')
    
    # 将真实最大值位置的像素坐标转换为物理坐标
    stress_max_phys_x, stress_max_phys_y = pixel_to_physical(
        stress_true_max_idx[1], stress_true_max_idx[0], img_height,
        config.unified_x_min, config.unified_x_max, config.unified_y_min, config.unified_y_max
    )
    
    # 标注真实最大值位置
    ax.plot(stress_max_phys_x, stress_max_phys_y, 'r*', markersize=MARKER_SIZE, 
            markeredgecolor='darkred', markeredgewidth=1.5)
    
    text_x, text_y, ha, va = get_text_position(
        stress_max_phys_x, stress_max_phys_y,
        config.unified_x_min, config.unified_x_max, config.unified_y_min, config.unified_y_max
    )
    
    annotation = f'True: {stress_true_max:.2f} MPa\nPred: {stress_pred_at_max:.2f} MPa\nError: {stress_max_loc_error:.2f} MPa'
    ax.text(text_x, text_y, annotation, 
            fontsize=ANNOTATION_FONTSIZE, color='darkred', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred', alpha=BOX_ALPHA))
    
    ax.set_title(f'Field MAE: {stress_mae_mpa:.4f} MPa | Max Loc Error: {stress_max_loc_error:.2f} MPa', 
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('X (mm)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Y (mm)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error (MPa)', fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # 添加子图字母标签
    for idx, (ax, label) in enumerate(zip(axes.flat, subplot_labels)):
        bbox = ax.get_position()
        fig.text(bbox.x0 + bbox.width/2, bbox.y0 - 0.035, label,
                 ha='center', va='top', fontsize=SUBLABEL_FONTSIZE)
    
    # 保存
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    cp.success(f"可视化图已保存: {output_path}")
    
    return {
        'temp_mae': temp_mae,
        'temp_max_loc_error': temp_max_loc_error,
        'temp_true_max': temp_true_max,
        'temp_pred_at_max': temp_pred_at_max,
        'temp_r2': temp_r2,
        'stress_mae_mpa': stress_mae_mpa,
        'stress_max_loc_error': stress_max_loc_error,
        'stress_true_max': stress_true_max,
        'stress_pred_at_max': stress_pred_at_max,
        'stress_r2': stress_r2
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    
    cp.header("🔬 功率→温度+应力 联合评估 (v13 - 最大值位置误差)")
    
    config = Config()
    
    print(f"\n{Fore.CYAN}配置信息:{Style.RESET_ALL}")
    print(f"  样本ID: {Fore.YELLOW}{config.sample_id}{Style.RESET_ALL}")
    print(f"  设备: {Fore.YELLOW}{config.device}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}v13版本改进:{Style.RESET_ALL}")
    print(f"  ✅ 方法1：在真实最大值位置计算预测误差")
    print(f"  ✅ 去掉P99标注，只标注真实最大值位置")
    print(f"  ✅ 显示：全场MAE + 最大值位置误差")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载缓存数据
    cp.header("【1】加载缓存数据")
    
    temp_cache_path = os.path.join(config.temp_cache_dir, config.temp_cache_pattern.format(config.sample_id))
    cp.info(f"温度缓存: {temp_cache_path}")
    power_temp, true_temp = load_cache_data(temp_cache_path)
    
    stress_cache_path = os.path.join(config.stress_cache_dir, config.stress_cache_pattern.format(config.sample_id))
    cp.info(f"应力缓存: {stress_cache_path}")
    power_stress, true_stress = load_cache_data(stress_cache_path)
    
    # 加载模型
    cp.header("【2】加载模型")
    
    cp.info("加载温度预测模型...")
    temp_model, temp_p_stats, temp_output_stats, _ = load_unet_model(
        config.temp_model_path, config.model_dir, config.device)
    cp.success("温度模型加载成功")
    
    cp.info("加载应力预测模型...")
    stress_model, stress_p_stats, stress_output_stats, _ = load_unet_model(
        config.stress_model_path, config.model_dir, config.device)
    cp.success("应力模型加载成功")
    
    # 执行预测
    cp.header("【3】执行预测")
    
    pred_temp = predict(temp_model, power_temp, temp_p_stats, temp_output_stats, config.device)
    pred_stress = predict(stress_model, power_stress, stress_p_stats, stress_output_stats, config.device)
    
    # 生成可视化
    cp.header("【4】生成可视化（v13 最大值位置误差）")
    
    output_path = os.path.join(config.output_dir, f"joint_evaluation_v13_sample_{config.sample_id:06d}.png")
    
    metrics = create_joint_visualization_horizontal(
        power_display=power_temp,
        true_temp=true_temp,
        pred_temp=pred_temp,
        true_stress=true_stress,
        pred_stress=pred_stress,
        sample_id=config.sample_id,
        output_path=output_path,
        dpi=config.dpi
    )
    
    # 打印结果摘要
    cp.header("✅ 评估完成")
    
    print(f"\n{Fore.CYAN}结果摘要 (v13 方法1):{Style.RESET_ALL}")
    print(f"\n  {Fore.YELLOW}【温度预测】{Style.RESET_ALL}")
    print(f"    全场MAE: {Fore.GREEN}{metrics['temp_mae']:.4f} K{Style.RESET_ALL}")
    print(f"    真实最大温度: {metrics['temp_true_max']:.2f} K")
    print(f"    该位置预测温度: {metrics['temp_pred_at_max']:.2f} K")
    print(f"    最大值位置误差: {Fore.RED}{metrics['temp_max_loc_error']:.4f} K{Style.RESET_ALL}")
    print(f"    R² Score: {Fore.GREEN}{metrics['temp_r2']:.6f}{Style.RESET_ALL}")
    
    print(f"\n  {Fore.YELLOW}【应力预测】{Style.RESET_ALL}")
    print(f"    全场MAE: {Fore.GREEN}{metrics['stress_mae_mpa']:.4f} MPa{Style.RESET_ALL}")
    print(f"    真实最大应力: {metrics['stress_true_max']:.2f} MPa")
    print(f"    该位置预测应力: {metrics['stress_pred_at_max']:.2f} MPa")
    print(f"    最大值位置误差: {Fore.RED}{metrics['stress_max_loc_error']:.4f} MPa{Style.RESET_ALL}")
    print(f"    R² Score: {Fore.GREEN}{metrics['stress_r2']:.4f}{Style.RESET_ALL}")
    
    print(f"\n  输出文件: {output_path}")


if __name__ == "__main__":
    main()