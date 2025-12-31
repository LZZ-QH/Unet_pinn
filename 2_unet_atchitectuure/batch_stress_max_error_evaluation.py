#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
批量应力预测评估脚本
=============================================================================
功能：
✅ 批量评估2001-3000号样本的应力预测
✅ 统计每个样本的最大应力预测值与真实值误差
✅ 生成详细的统计分析报告
✅ 输出CSV文件和可视化图表
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================

class Config:
    """评估配置"""
    
    # 样本范围
    sample_start = 2001
    sample_end = 3000
    
    # 模型架构路径
    model_dir = "/home/txr/code/final_version/2_unet_architecture"
    
    # 应力模型配置
    stress_model_path = "/home/txr/code/final_version/2_unet_architecture/training_logs/power_to_stress_prediction/cache/unet_mask_weighted_power2stress_final_20251224_113026.pth"
    stress_cache_dir = "/home/txr/code/final_version/2_unet_architecture/cache/power_to_stress"
    stress_cache_pattern = "power_stress_{:04d}.pkl"
    
    # 输出配置
    output_dir = "/home/txr/code/final_version/2_unet_architecture/evaluation_result/batch_stress_analysis_final"
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 数据加载和模型函数
# ============================================================

def load_cache_data(cache_path: str):
    """加载缓存数据"""
    if not os.path.exists(cache_path):
        return None, None
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple) and len(data) == 2:
        return data[0], data[1]
    else:
        return None, None


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
        print("  检测到旧版架构权重，自动转换键名...")
        state_dict = {k.replace('.attention.attention.', '.attention.'): v 
                      for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    p_stats = checkpoint.get('p_stats', {})

    # 兼容：训练代码将输出统计统一保存在 t_stats 中
    if checkpoint.get('t_stats'):
        s_stats = checkpoint['t_stats']
    elif checkpoint.get('s_stats'):
        s_stats = checkpoint['s_stats']
    else:
        raise ValueError("Checkpoint中缺少输出统计信息")

    return model, p_stats, s_stats
    
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


def predict(model, power: np.ndarray, p_stats: dict, s_stats: dict, device: str):
    """执行预测"""
    power_norm = normalize_power(power, p_stats)
    power_t = torch.from_numpy(power_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_norm = model(power_t)
    
    pred = denormalize_output(pred_norm.cpu().squeeze().numpy(), s_stats)
    return pred


# ============================================================
# 批量评估
# ============================================================

def batch_evaluate(config: Config):
    """批量评估所有样本"""
    
    print("=" * 70)
    print("批量应力预测评估")
    print("=" * 70)
    print(f"\n样本范围: {config.sample_start} - {config.sample_end}")
    print(f"设备: {config.device}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n加载应力预测模型...")
    model, p_stats, s_stats = load_unet_model(
        config.stress_model_path,
        config.model_dir,
        config.device
    )
    print("✓ 模型加载成功")
    
    # 存储结果
    results = []
    failed_samples = []
    
    # 批量评估
    print(f"\n开始评估 {config.sample_end - config.sample_start + 1} 个样本...")
    
    for sample_id in tqdm(range(config.sample_start, config.sample_end + 1), 
                          desc="评估进度"):
        cache_path = os.path.join(
            config.stress_cache_dir,
            config.stress_cache_pattern.format(sample_id)
        )
        
        power, true_stress = load_cache_data(cache_path)
        
        if power is None or true_stress is None:
            failed_samples.append(sample_id)
            continue
        
        # 预测
        pred_stress = predict(model, power, p_stats, s_stats, config.device)
        
        # 转换为MPa
        true_stress_mpa = true_stress / 1e6
        pred_stress_mpa = pred_stress / 1e6
        
        # 计算统计信息
        # 最大应力
        true_max = true_stress_mpa.max()
        pred_max = pred_stress_mpa.max()
        max_error = abs(pred_max - true_max)
        max_relative_error = (max_error / true_max) * 100 if true_max > 0 else 0
        
        # 全场MAE
        mae = np.abs(pred_stress_mpa - true_stress_mpa).mean()
        
        # 全场最大误差
        field_max_error = np.abs(pred_stress_mpa - true_stress_mpa).max()
        
        # R²
        r2 = r2_score(true_stress_mpa.flatten(), pred_stress_mpa.flatten())
        
        results.append({
            'sample_id': sample_id,
            'true_max_mpa': true_max,
            'pred_max_mpa': pred_max,
            'max_stress_error_mpa': max_error,
            'max_stress_relative_error_%': max_relative_error,
            'field_mae_mpa': mae,
            'field_max_error_mpa': field_max_error,
            'r2': r2
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n✓ 评估完成")
    print(f"  成功样本: {len(df)}")
    print(f"  失败样本: {len(failed_samples)}")
    
    return df, failed_samples


# ============================================================
# 统计分析
# ============================================================

def analyze_results(df: pd.DataFrame, output_dir: str):
    """分析评估结果"""
    
    print("\n" + "=" * 70)
    print("统计分析结果")
    print("=" * 70)
    
    # 1. 最大应力误差统计
    print("\n【最大应力预测误差统计】")
    print(f"  样本数: {len(df)}")
    print(f"  平均误差 (MAE): {df['max_stress_error_mpa'].mean():.4f} MPa")
    print(f"  中位数误差: {df['max_stress_error_mpa'].median():.4f} MPa")
    print(f"  最小误差: {df['max_stress_error_mpa'].min():.4f} MPa")
    print(f"  最大误差: {df['max_stress_error_mpa'].max():.4f} MPa")
    print(f"  标准差: {df['max_stress_error_mpa'].std():.4f} MPa")
    
    # 分位数
    print(f"\n  分位数分布:")
    for p in [50, 75, 90, 95, 99]:
        val = df['max_stress_error_mpa'].quantile(p/100)
        print(f"    P{p}: {val:.4f} MPa")
    
    # 2. 相对误差统计
    print("\n【最大应力相对误差统计】")
    print(f"  平均相对误差: {df['max_stress_relative_error_%'].mean():.2f}%")
    print(f"  中位数相对误差: {df['max_stress_relative_error_%'].median():.2f}%")
    print(f"  最大相对误差: {df['max_stress_relative_error_%'].max():.2f}%")
    
    # 3. 全场误差统计
    print("\n【全场误差统计】")
    print(f"  平均MAE: {df['field_mae_mpa'].mean():.4f} MPa")
    print(f"  平均最大误差: {df['field_max_error_mpa'].mean():.4f} MPa")
    print(f"  平均R²: {df['r2'].mean():.6f}")
    
    # 4. 误差分布
    print("\n【误差分布统计】")
    bins = [0, 1, 2, 3, 5, 10, 20, float('inf')]
    labels = ['0-1', '1-2', '2-3', '3-5', '5-10', '10-20', '>20']
    df['error_bin'] = pd.cut(df['max_stress_error_mpa'], bins=bins, labels=labels)
    dist = df['error_bin'].value_counts().sort_index()
    print("  最大应力误差分布 (MPa):")
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {label} MPa: {count} 样本 ({pct:.1f}%)")
    
    # 5. 找出误差最大和最小的样本
    print("\n【误差最大的10个样本】")
    top10_error = df.nlargest(10, 'max_stress_error_mpa')
    for _, row in top10_error.iterrows():
        print(f"  Sample {int(row['sample_id'])}: 真实={row['true_max_mpa']:.2f}, "
              f"预测={row['pred_max_mpa']:.2f}, 误差={row['max_stress_error_mpa']:.2f} MPa "
              f"({row['max_stress_relative_error_%']:.1f}%)")
    
    print("\n【误差最小的10个样本】")
    bottom10_error = df.nsmallest(10, 'max_stress_error_mpa')
    for _, row in bottom10_error.iterrows():
        print(f"  Sample {int(row['sample_id'])}: 真实={row['true_max_mpa']:.2f}, "
              f"预测={row['pred_max_mpa']:.2f}, 误差={row['max_stress_error_mpa']:.4f} MPa "
              f"({row['max_stress_relative_error_%']:.2f}%)")
    
    return df


# ============================================================
# 可视化
# ============================================================

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """创建可视化图表 - 仅生成最大应力预测vs真实值散点图"""
    
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 最大应力预测 vs 真实值 散点图
    ax.scatter(df['true_max_mpa'], df['pred_max_mpa'], alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    min_val = min(df['true_max_mpa'].min(), df['pred_max_mpa'].min())
    max_val = max(df['true_max_mpa'].max(), df['pred_max_mpa'].max())
    padding = (max_val - min_val) * 0.02
    ax.plot([min_val - padding, max_val + padding], 
            [min_val - padding, max_val + padding], 
            'r--', linewidth=2, label='Perfect')
    
    # 计算R²
    r2 = r2_score(df['true_max_mpa'], df['pred_max_mpa'])
    
    ax.set_xlabel('True Max Stress (MPa)', fontsize=18)
    ax.set_ylabel('Predicted Max Stress (MPa)', fontsize=18)
    ax.set_title(f'Max Stress: Prediction vs True (R²={r2:.4f})', fontsize=20)
    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=16)
    
    # 添加统计信息（右下角，避免遮挡）
    mae = df['max_stress_error_mpa'].mean()
    stats_text = f"MAE: {mae:.3f} MPa\nSamples: {len(df)}"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'max_stress_prediction_scatter.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 图表已保存: {output_path}")
    
    return output_path


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    
    config = Config()
    
    # 批量评估
    df, failed_samples = batch_evaluate(config)
    
    if len(df) == 0:
        print("没有成功评估的样本！")
        return
    
    # 统计分析
    df = analyze_results(df, config.output_dir)
    
    # 保存CSV
    csv_path = os.path.join(config.output_dir, 'batch_stress_evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 详细结果已保存: {csv_path}")
    
    # 保存失败样本列表
    if failed_samples:
        failed_path = os.path.join(config.output_dir, 'failed_samples.txt')
        with open(failed_path, 'w') as f:
            for sid in failed_samples:
                f.write(f"{sid}\n")
        print(f"✓ 失败样本列表已保存: {failed_path}")
    
    # 生成可视化
    create_visualizations(df, config.output_dir)
    
    # 保存统计摘要
    summary_path = os.path.join(config.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("批量应力预测评估统计摘要\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"样本范围: {config.sample_start} - {config.sample_end}\n")
        f.write(f"成功评估样本数: {len(df)}\n")
        f.write(f"失败样本数: {len(failed_samples)}\n\n")
        
        f.write("【最大应力预测误差统计】\n")
        f.write(f"  平均误差 (MAE): {df['max_stress_error_mpa'].mean():.4f} MPa\n")
        f.write(f"  中位数误差: {df['max_stress_error_mpa'].median():.4f} MPa\n")
        f.write(f"  最小误差: {df['max_stress_error_mpa'].min():.4f} MPa\n")
        f.write(f"  最大误差: {df['max_stress_error_mpa'].max():.4f} MPa\n")
        f.write(f"  标准差: {df['max_stress_error_mpa'].std():.4f} MPa\n\n")
        
        f.write("  分位数:\n")
        for p in [50, 75, 90, 95, 99]:
            val = df['max_stress_error_mpa'].quantile(p/100)
            f.write(f"    P{p}: {val:.4f} MPa\n")
        
        f.write("\n【相对误差统计】\n")
        f.write(f"  平均相对误差: {df['max_stress_relative_error_%'].mean():.2f}%\n")
        f.write(f"  中位数相对误差: {df['max_stress_relative_error_%'].median():.2f}%\n")
        f.write(f"  最大相对误差: {df['max_stress_relative_error_%'].max():.2f}%\n\n")
        
        f.write("【全场统计】\n")
        f.write(f"  平均MAE: {df['field_mae_mpa'].mean():.4f} MPa\n")
        f.write(f"  平均R²: {df['r2'].mean():.6f}\n")
        
        # 最大应力的R²
        r2_max = r2_score(df['true_max_mpa'], df['pred_max_mpa'])
        f.write(f"  最大应力R²: {r2_max:.6f}\n")
    
    print(f"✓ 统计摘要已保存: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✅ 批量评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()