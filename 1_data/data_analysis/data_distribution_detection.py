#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
统一数据分析脚本 - 功率、温度、应力
=============================================================================
功能：
✅ 分析功率数据 (power_a_*.txt)
✅ 分析温度数据 (temp*.txt)
✅ 分析应力数据 (strength*.txt)
✅ 每种数据生成2张图：分布热力图 + CDF累积分布图
✅ 按固定间隔选择文件进行分析

输出目录：
- 功率: /home/txr/code/second_version/final_version/1_data_generate/data_analysis/power
- 温度: /home/txr/code/second_version/final_version/1_data_generate/data_analysis/temperature
- 应力: /home/txr/code/second_version/final_version/1_data_generate/data_analysis/stress
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# =============================================================================
# 通用函数
# =============================================================================

def create_2d_histogram(x, y, values, bins=100):
    """创建2D直方图(热力图)"""
    x_edges = np.linspace(x.min(), x.max(), bins)
    y_edges = np.linspace(y.min(), y.max(), bins)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where(counts > 0, H / counts, 0)
    
    return H.T, xedges, yedges


# =============================================================================
# 功率数据分析
# =============================================================================

def load_power_data(file_path):
    """加载功率数据文件 (CSV格式: x, y, power)"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=0)
    x = data[:, 0]
    y = data[:, 1]
    power = data[:, 2]
    return x, y, power


def plot_power_distribution(file_path, bins=100, cmap='jet', save_path=None):
    """绘制功率分布图"""
    x, y, power = load_power_data(file_path)
    
    filename = os.path.basename(file_path)
    file_number = filename.replace('power_a_', '').replace('.txt', '')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # (a) 原始分布 - 全部数据
    total_points = len(x)
    H1, xedges1, yedges1 = create_2d_histogram(x, y, power, bins=bins)
    
    im1 = ax1.pcolormesh(xedges1, yedges1, H1, cmap=cmap, shading='auto')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title(f'(a) Power Distribution - All Data\nFile: {filename} ({total_points:,} points)', fontsize=11)
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Power (W/m³)')
    
    # (b) 非零值分布
    nonzero_mask = power > 0
    x_nonzero = x[nonzero_mask]
    y_nonzero = y[nonzero_mask]
    power_nonzero = power[nonzero_mask]
    
    nonzero_count = len(x_nonzero)
    nonzero_ratio = (nonzero_count / total_points) * 100 if total_points > 0 else 0
    
    if nonzero_count > 0:
        H2, xedges2, yedges2 = create_2d_histogram(x_nonzero, y_nonzero, power_nonzero, bins=bins)
        im2 = ax2.pcolormesh(xedges2, yedges2, H2, cmap=cmap, shading='auto')
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title(f'(b) Non-zero Power Distribution\n({nonzero_count:,} points, {nonzero_ratio:.1f}%)', fontsize=11)
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Power (W/m³)')
    else:
        ax2.text(0.5, 0.5, 'No Non-zero Values', ha='center', va='center', fontsize=16)
        ax2.set_title(f'(b) Non-zero Distribution\n(0 points, 0.0%)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Distribution saved: {save_path}")
    
    plt.close()
    
    return {
        'filename': filename,
        'file_number': file_number,
        'total_points': total_points,
        'nonzero_count': nonzero_count,
        'nonzero_ratio': nonzero_ratio,
        'value_min': power.min(),
        'value_max': power.max(),
        'value_mean': power.mean(),
        'value_median': np.median(power_nonzero) if nonzero_count > 0 else 0
    }


def plot_power_cdf(file_path, save_path=None):
    """绘制功率CDF图（包含零值统计）"""
    x, y, power = load_power_data(file_path)
    filename = os.path.basename(file_path)
    
    total_points = len(power)
    
    # 统计零值和非零值
    zero_count = np.sum(power == 0)
    zero_ratio = (zero_count / total_points) * 100
    
    power_nonzero = power[power > 0]
    nonzero_count = len(power_nonzero)
    nonzero_ratio = (nonzero_count / total_points) * 100
    
    if nonzero_count == 0:
        print(f"  Warning: {filename} has no non-zero values")
        return None, None, None
    
    # 排序非零值
    sorted_power = np.sort(power_nonzero)
    
    # CDF从零值比例开始
    cdf_increments = np.arange(1, len(sorted_power) + 1) / len(sorted_power) * nonzero_ratio
    cdf = zero_ratio + cdf_increments
    
    # 统计量
    median_value = np.median(sorted_power)
    percentile_90_value = np.percentile(sorted_power, 90)
    
    median_position = zero_ratio + 50 * nonzero_ratio / 100
    percentile_90_position = zero_ratio + 90 * nonzero_ratio / 100
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 零值水平线
    if zero_ratio > 0:
        ax.hlines(y=zero_ratio, xmin=0, xmax=sorted_power[0], 
                 colors='blue', linewidth=2, linestyles='solid', 
                 label=f'Zero values ({zero_ratio:.1f}%)')
        ax.axhspan(0, zero_ratio, alpha=0.1, color='blue')
        ax.text(sorted_power[0] * 0.5, zero_ratio/2, 
               f'{zero_count:,} zero points\n({zero_ratio:.1f}%)',
               fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # CDF曲线
    ax.plot(sorted_power, cdf, 'g-', linewidth=2, label='Non-zero CDF')
    
    # 百分位线
    ax.axhline(y=median_position, color='red', linestyle='--', linewidth=1.5, label='Median (50th)')
    ax.axhline(y=percentile_90_position, color='orange', linestyle='--', linewidth=1.5, label='90th percentile')
    ax.axvline(x=median_value, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=percentile_90_value, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Power (W/m³)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title(f'Power CDF (Including Zero Values)\nFile: {filename}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim([0, 105])
    
    # 统计文本
    stats_text = (f'Total: {total_points:,}\n'
                 f'Zero: {zero_count:,} ({zero_ratio:.1f}%)\n'
                 f'Non-zero: {nonzero_count:,} ({nonzero_ratio:.1f}%)\n'
                 f'---\n'
                 f'Median: {median_value:.2e}\n'
                 f'90th: {percentile_90_value:.2e}')
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ CDF saved: {save_path}")
    
    plt.close()
    
    return zero_ratio, median_value, percentile_90_value


# =============================================================================
# 温度/应力数据分析
# =============================================================================

def load_comsol_data(file_path):
    """加载COMSOL格式数据文件 (温度或应力)"""
    valid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            if 'X,Y,Z' in line or 'x,y,z' in line.lower():
                continue
            valid_lines.append(line)
    
    if len(valid_lines) == 0:
        raise ValueError(f"No valid data found in {file_path}")
    
    data = []
    for line in valid_lines:
        try:
            values = [float(v.strip()) for v in line.split(',')]
            if len(values) >= 4:
                data.append([values[0], values[1], values[2], values[3]])
        except (ValueError, IndexError):
            continue
    
    if len(data) == 0:
        raise ValueError(f"Could not parse data from {file_path}")
    
    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 3]  # X, Y, Value


def plot_temp_stress_distribution(file_path, data_type='temp', bins=100, cmap='hot', save_path=None):
    """绘制温度/应力分布图"""
    x, y, values = load_comsol_data(file_path)
    
    filename = os.path.basename(file_path)
    
    if data_type == 'temp':
        pattern = r'temp(\d+)\.txt'
        value_label = 'Temperature (K)'
        title_prefix = 'Temperature'
    else:
        pattern = r'strength(\d+)\.txt'
        value_label = 'Stress (Pa)'
        title_prefix = 'Stress'
    
    match = re.match(pattern, filename)
    file_number = match.group(1) if match else 'unknown'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # (a) 完整分布
    total_points = len(x)
    H1, xedges1, yedges1 = create_2d_histogram(x, y, values, bins=bins)
    
    im1 = ax1.pcolormesh(xedges1, yedges1, H1, cmap=cmap, shading='auto')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'(a) {title_prefix} Distribution - All Data\n'
                  f'File: {filename} ({total_points:,} points)', fontsize=11)
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label=value_label)
    
    # (b) 中位数±1σ范围内的分布
    median = np.median(values)
    std = np.std(values)
    mask_median = np.abs(values - median) <= std
    
    x_median = x[mask_median]
    y_median = y[mask_median]
    values_median = values[mask_median]
    
    median_count = len(x_median)
    median_ratio = (median_count / total_points) * 100 if total_points > 0 else 0
    
    if median_count > 0:
        H2, xedges2, yedges2 = create_2d_histogram(x_median, y_median, values_median, bins=bins)
        im2 = ax2.pcolormesh(xedges2, yedges2, H2, cmap=cmap, shading='auto')
        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Y (m)', fontsize=12)
        ax2.set_title(f'(b) Values Within ±1σ of Median\n'
                      f'({median_count:,} points, {median_ratio:.1f}%)', fontsize=11)
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label=value_label)
    else:
        ax2.text(0.5, 0.5, 'No Data in Range', ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Distribution saved: {save_path}")
    
    plt.close()
    
    return {
        'filename': filename,
        'file_number': file_number,
        'data_type': data_type,
        'total_points': total_points,
        'value_min': values.min(),
        'value_max': values.max(),
        'value_mean': values.mean(),
        'value_median': median,
        'value_std': std
    }


def plot_temp_stress_cdf(file_path, data_type='temp', save_path=None):
    """绘制温度/应力CDF图"""
    x, y, values = load_comsol_data(file_path)
    filename = os.path.basename(file_path)
    
    if data_type == 'temp':
        value_label = 'Temperature (K)'
        title_prefix = 'Temperature'
    else:
        value_label = 'Stress (Pa)'
        title_prefix = 'Stress'
    
    total_points = len(values)
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    
    # 统计量
    median_value = np.median(sorted_values)
    percentile_10 = np.percentile(sorted_values, 10)
    percentile_25 = np.percentile(sorted_values, 25)
    percentile_75 = np.percentile(sorted_values, 75)
    percentile_90 = np.percentile(sorted_values, 90)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(sorted_values, cdf, 'b-', linewidth=2, label='CDF')
    
    # 百分位线
    ax.axhline(y=10, color='lightblue', linestyle='--', linewidth=1, alpha=0.7, label='10th')
    ax.axhline(y=25, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='25th')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Median')
    ax.axhline(y=75, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='75th')
    ax.axhline(y=90, color='gold', linestyle='--', linewidth=1, alpha=0.7, label='90th')
    
    ax.axvline(x=percentile_10, color='lightblue', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=percentile_25, color='cyan', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=median_value, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=percentile_75, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=percentile_90, color='gold', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(value_label, fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title(f'{title_prefix} CDF\nFile: {filename}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim([0, 105])
    
    # 统计文本
    stats_text = (f'Total: {total_points:,}\n'
                 f'---\n'
                 f'Min: {sorted_values[0]:.3e}\n'
                 f'10th: {percentile_10:.3e}\n'
                 f'25th: {percentile_25:.3e}\n'
                 f'Median: {median_value:.3e}\n'
                 f'75th: {percentile_75:.3e}\n'
                 f'90th: {percentile_90:.3e}\n'
                 f'Max: {sorted_values[-1]:.3e}\n'
                 f'---\n'
                 f'Mean: {values.mean():.3e}\n'
                 f'Std: {values.std():.3e}')
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ CDF saved: {save_path}")
    
    plt.close()
    
    return median_value, percentile_10, percentile_90


# =============================================================================
# 批量分析函数
# =============================================================================

def analyze_power_batch(data_dir, start_id, end_id, step, bins, output_dir):
    """批量分析功率数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_ids = list(range(start_id, end_id + 1, step))
    
    print("="*70)
    print("Power Data Analysis")
    print("="*70)
    print(f"Directory: {data_dir}")
    print(f"ID Range: {start_id} - {end_id}, Step: {step}")
    print(f"Selected IDs: {selected_ids}")
    print("="*70)
    
    all_stats = []
    
    for file_id in selected_ids:
        file_path = os.path.join(data_dir, f'power_a_{file_id}.txt')
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: power_a_{file_id}.txt")
            continue
        
        print(f"\n[Power] Analyzing: power_a_{file_id}.txt")
        
        dist_path = os.path.join(output_dir, f'power_distribution_{file_id}.png')
        cdf_path = os.path.join(output_dir, f'power_cdf_{file_id}.png')
        
        stats = plot_power_distribution(file_path, bins=bins, cmap='jet', save_path=dist_path)
        zero_ratio, median, p90 = plot_power_cdf(file_path, save_path=cdf_path)
        
        if zero_ratio is not None:
            stats['zero_ratio'] = zero_ratio
            stats['median'] = median
            stats['percentile_90'] = p90
        
        all_stats.append(stats)
        print(f"  Points: {stats['total_points']:,}, Range: [{stats['value_min']:.2e}, {stats['value_max']:.2e}]")
    
    # 保存摘要
    save_summary(output_dir, 'power', all_stats, start_id, end_id, step, bins)
    
    return all_stats


def analyze_temp_batch(data_dir, start_id, end_id, step, bins, output_dir):
    """批量分析温度数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_ids = list(range(start_id, end_id + 1, step))
    
    print("\n" + "="*70)
    print("Temperature Data Analysis")
    print("="*70)
    print(f"Directory: {data_dir}")
    print(f"ID Range: {start_id} - {end_id}, Step: {step}")
    print(f"Selected IDs: {selected_ids}")
    print("="*70)
    
    all_stats = []
    
    for file_id in selected_ids:
        file_path = os.path.join(data_dir, f'temp{file_id}.txt')
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: temp{file_id}.txt")
            continue
        
        print(f"\n[Temp] Analyzing: temp{file_id}.txt")
        
        dist_path = os.path.join(output_dir, f'temp_distribution_{file_id}.png')
        cdf_path = os.path.join(output_dir, f'temp_cdf_{file_id}.png')
        
        try:
            stats = plot_temp_stress_distribution(file_path, data_type='temp', bins=bins, cmap='hot', save_path=dist_path)
            median, p10, p90 = plot_temp_stress_cdf(file_path, data_type='temp', save_path=cdf_path)
            
            stats['cdf_median'] = median
            stats['percentile_10'] = p10
            stats['percentile_90'] = p90
            
            all_stats.append(stats)
            print(f"  Points: {stats['total_points']:,}, Range: [{stats['value_min']:.2e}, {stats['value_max']:.2e}]")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    save_summary(output_dir, 'temperature', all_stats, start_id, end_id, step, bins)
    
    return all_stats


def analyze_stress_batch(data_dir, start_id, end_id, step, bins, output_dir):
    """批量分析应力数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_ids = list(range(start_id, end_id + 1, step))
    
    print("\n" + "="*70)
    print("Stress Data Analysis")
    print("="*70)
    print(f"Directory: {data_dir}")
    print(f"ID Range: {start_id} - {end_id}, Step: {step}")
    print(f"Selected IDs: {selected_ids}")
    print("="*70)
    
    all_stats = []
    
    for file_id in selected_ids:
        file_path = os.path.join(data_dir, f'strength{file_id}.txt')
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: strength{file_id}.txt")
            continue
        
        print(f"\n[Stress] Analyzing: strength{file_id}.txt")
        
        dist_path = os.path.join(output_dir, f'stress_distribution_{file_id}.png')
        cdf_path = os.path.join(output_dir, f'stress_cdf_{file_id}.png')
        
        try:
            stats = plot_temp_stress_distribution(file_path, data_type='strength', bins=bins, cmap='viridis', save_path=dist_path)
            median, p10, p90 = plot_temp_stress_cdf(file_path, data_type='strength', save_path=cdf_path)
            
            stats['cdf_median'] = median
            stats['percentile_10'] = p10
            stats['percentile_90'] = p90
            
            all_stats.append(stats)
            print(f"  Points: {stats['total_points']:,}, Range: [{stats['value_min']:.2e}, {stats['value_max']:.2e}]")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    save_summary(output_dir, 'stress', all_stats, start_id, end_id, step, bins)
    
    return all_stats


def save_summary(output_dir, data_type, all_stats, start_id, end_id, step, bins):
    """保存分析摘要"""
    summary_path = os.path.join(output_dir, f'{data_type}_analysis_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{data_type.upper()} Data Analysis Summary\n")
        f.write("="*80 + "\n")
        f.write(f"ID Range: {start_id} - {end_id}\n")
        f.write(f"Step: {step}\n")
        f.write(f"Bins: {bins}\n")
        f.write(f"Files Analyzed: {len(all_stats)}\n")
        f.write("="*80 + "\n\n")
        
        for stat in all_stats:
            f.write(f"File: {stat['filename']}\n")
            f.write(f"  Total Points: {stat['total_points']:,}\n")
            f.write(f"  Value Range: [{stat['value_min']:.3e}, {stat['value_max']:.3e}]\n")
            f.write(f"  Mean: {stat['value_mean']:.3e}\n")
            f.write(f"  Median: {stat['value_median']:.3e}\n")
            if 'value_std' in stat:
                f.write(f"  Std: {stat['value_std']:.3e}\n")
            if 'zero_ratio' in stat:
                f.write(f"  Zero Ratio: {stat['zero_ratio']:.1f}%\n")
            f.write("\n")
    
    print(f"\n✓ Summary saved: {summary_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 分析功率、温度、应力数据"""
    
    # 配置参数
    power_input_dir = "/home/txr/power_a_input/"
    output_dir = "/home/txr/power_a_output/"
    
    start_id = 1
    end_id = 3000
    step = 500
    bins = 250
    
    # 输出目录
    power_output = "/home/txr/code/second_version/final_version/1_data_generate/data_analysis/power"
    temp_output = "/home/txr/code/second_version/final_version/1_data_generate/data_analysis/temperature"
    stress_output = "/home/txr/code/second_version/final_version/1_data_generate/data_analysis/stress"
    
    print("\n" + "🔬"*35)
    print("Unified Data Analysis: Power, Temperature, Stress")
    print("🔬"*35 + "\n")
    
    # 1. 分析功率数据
    print("\n📊 Part 1: Power Data")
    print("-"*70)
    analyze_power_batch(power_input_dir, start_id, end_id, step, bins, power_output)
    
    # 2. 分析温度数据
    print("\n📊 Part 2: Temperature Data")
    print("-"*70)
    analyze_temp_batch(output_dir, start_id, end_id, step, bins, temp_output)
    
    # 3. 分析应力数据
    print("\n📊 Part 3: Stress Data")
    print("-"*70)
    analyze_stress_batch(output_dir, start_id, end_id, step, bins, stress_output)
    
    # 完成
    print("\n" + "="*70)
    print("✅ All Analysis Complete!")
    print("="*70)
    print(f"Power results:       {power_output}")
    print(f"Temperature results: {temp_output}")
    print(f"Stress results:      {stress_output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()