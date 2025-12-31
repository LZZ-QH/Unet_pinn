#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
UNet应力预测 + PINN疲劳寿命预测 整合Pipeline
================================================================================
功能：
  1. 读取UNet批量应力预测结果 (CSV文件)
  2. 使用训练好的PINN模型预测疲劳寿命Nf
  3. 对比使用真实应力 vs UNet预测应力 计算的疲劳寿命差异
  4. 生成论文用图表:
     - Fig 11: Nf预测 vs 真实值散点图
     - Fig 12: 误差传递分析 (2子图)

完整预测链路：
  功率密度 → UNet → 最大应力σm → PINN → 疲劳寿命Nf

================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import time
import argparse
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置matplotlib - 论文级别配置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


# ==============================================================================
# 配置类
# ==============================================================================

@dataclass
class Config:
    """配置参数"""
    
    # UNet评估结果路径
    unet_results_csv: str = "/home/txr/code/final_version/2_unet_architecture/evaluation_result/batch_stress_analysis_final/batch_stress_evaluation_results.csv"
    
    # 输出目录
    output_dir: str = "/home/txr/code/final_version/3_pinn_architecture/integrated_fatigue_analysis"
    
    # 材料参数 (焊料层 Sn-Ag-Cu 或 60Sn-40Pb 典型值)
    epsilon_f_prime: float = 0.325      # 疲劳延性系数
    sigma_f_prime: float = 1200.0       # 疲劳强度系数 (MPa)
    c: float = -0.57                    # 疲劳延性指数
    E: float = 45000.0                  # 弹性模量 (MPa)
    b: float = -0.12                    # 疲劳强度指数
    
    # PINN训练参数
    pinn_epochs: int = 10000
    pinn_lr: float = 0.001
    lambda_phys: float = 0.1
    n_train_samples: int = 3000
    
    # 训练数据参数范围
    epsilon_f_range: Tuple[float, float] = (0.25, 0.55)
    sigma_f_range: Tuple[float, float] = (800, 2000)
    c_range: Tuple[float, float] = (-0.8, -0.4)
    E_range: Tuple[float, float] = (30000, 80000)
    sigma_m_range: Tuple[float, float] = (50, 500)
    
    # 可视化参数 - 论文级别
    dpi: int = 600
    
    # PINN模型保存路径
    pinn_model_path: str = ""
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.pinn_model_path:
            self.pinn_model_path = os.path.join(self.output_dir, "pinn_model.pth")


# ==============================================================================
# 物理方程和求解器
# ==============================================================================

def coffin_manson_residual(Nf: float, epsilon_f_prime: float, sigma_f_prime: float,
                           c: float, E: float, sigma_m: float, b: float = -0.12) -> float:
    """Coffin-Manson 方程残差"""
    lhs = sigma_m / (2 * E)
    term1 = epsilon_f_prime * (2 * Nf) ** c
    term2 = ((sigma_f_prime - sigma_m) / E) * (2 * Nf) ** b
    return lhs - (term1 + term2)


def solve_bisection(epsilon_f_prime: float, sigma_f_prime: float, c: float,
                    E: float, sigma_m: float, b: float = -0.12) -> Optional[float]:
    """二分法求解Nf"""
    low, high = 1.0, 1e15
    tol = 1e-9
    
    f_low = coffin_manson_residual(low, epsilon_f_prime, sigma_f_prime, c, E, sigma_m, b)
    f_high = coffin_manson_residual(high, epsilon_f_prime, sigma_f_prime, c, E, sigma_m, b)
    
    if f_low * f_high > 0:
        return np.nan
    
    for _ in range(100):
        mid = (low + high) / 2
        f_mid = coffin_manson_residual(mid, epsilon_f_prime, sigma_f_prime, c, E, sigma_m, b)
        
        if abs(f_mid) < tol:
            break
        
        if f_low * f_mid < 0:
            high = mid
        else:
            low = mid
            f_low = f_mid
    
    return (low + high) / 2


# ==============================================================================
# PINN 模型
# ==============================================================================

class CoffinMansonPINN(nn.Module):
    """PINN模型：预测疲劳寿命"""
    
    def __init__(self, hidden_dims: List[int] = [64, 64, 64, 32]):
        super().__init__()
        
        layers = []
        in_dim = 5
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNFatiguePredictor:
    """PINN疲劳寿命预测器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = CoffinMansonPINN()
        self.X_mean = None
        self.X_std = None
        self.is_trained = False
    
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        n = self.config.n_train_samples
        
        epsilon_f = np.random.uniform(*self.config.epsilon_f_range, n)
        sigma_f = np.random.uniform(*self.config.sigma_f_range, n)
        c = np.random.uniform(*self.config.c_range, n)
        E = np.random.uniform(*self.config.E_range, n)
        sigma_m = np.random.uniform(*self.config.sigma_m_range, n)
        
        X = np.stack([epsilon_f, sigma_f, c, E, sigma_m], axis=1)
        
        Y = []
        valid_idx = []
        
        for i in range(n):
            nf = solve_bisection(epsilon_f[i], sigma_f[i], c[i], E[i], sigma_m[i], self.config.b)
            if not np.isnan(nf) and nf > 0:
                Y.append(np.log10(nf))
                valid_idx.append(i)
        
        X = X[valid_idx]
        Y = np.array(Y).reshape(-1, 1)
        
        return X, Y
    
    def _physics_loss(self, x_norm: torch.Tensor, x_phys: torch.Tensor) -> torch.Tensor:
        """计算物理损失"""
        log_Nf_pred = self.model(x_norm)
        Nf_pred = torch.pow(10, log_Nf_pred)
        
        eps_f = x_phys[:, 0:1]
        sig_f = x_phys[:, 1:2]
        c = x_phys[:, 2:3]
        E = x_phys[:, 3:4]
        sig_m = x_phys[:, 4:5]
        b = self.config.b
        
        term1 = eps_f * torch.pow(2 * Nf_pred, c)
        term2 = ((sig_f - sig_m) / E) * torch.pow(2 * Nf_pred, b)
        lhs = sig_m / (2 * E)
        rhs = term1 + term2
        
        return torch.mean((lhs - rhs) ** 2)
    
    def train(self, verbose: bool = True) -> float:
        """训练PINN模型"""
        print("\n" + "=" * 60)
        print("训练 PINN 疲劳寿命预测模型")
        print("=" * 60)
        
        print("\n[1/3] 生成训练数据...")
        X, Y = self._generate_training_data()
        print(f"  有效样本数: {len(X)}")
        
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        X_norm = (X - self.X_mean) / self.X_std
        
        train_size = int(0.8 * len(X))
        X_train = torch.tensor(X_norm[:train_size], dtype=torch.float32)
        Y_train = torch.tensor(Y[:train_size], dtype=torch.float32)
        X_phys = torch.tensor(X[:train_size], dtype=torch.float32)
        
        print(f"\n[2/3] 训练模型 ({self.config.pinn_epochs} epochs)...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.pinn_lr)
        
        start_time = time.time()
        loss_history = []
        
        for epoch in range(self.config.pinn_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            y_pred = self.model(X_train)
            loss_data = nn.MSELoss()(y_pred, Y_train)
            loss_phy = self._physics_loss(X_train, X_phys)
            
            total_loss = loss_data + self.config.lambda_phys * loss_phy
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            
            if verbose and (epoch + 1) % 2000 == 0:
                print(f"  Epoch {epoch+1}: Loss = {total_loss.item():.6f}")
        
        train_time = time.time() - start_time
        print(f"\n[3/3] 训练完成! 耗时: {train_time:.1f}s")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'config': {
                'epsilon_f_prime': self.config.epsilon_f_prime,
                'sigma_f_prime': self.config.sigma_f_prime,
                'c': self.config.c,
                'E': self.config.E,
                'b': self.config.b
            },
            'loss_history': loss_history
        }, self.config.pinn_model_path)
        print(f"  模型已保存: {self.config.pinn_model_path}")
        
        self.is_trained = True
        self.loss_history = loss_history
        
        return train_time
    
    def load(self, model_path: str = None):
        """加载预训练模型"""
        path = model_path or self.config.pinn_model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含numpy数组的checkpoint
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.X_mean = checkpoint['X_mean']
        self.X_std = checkpoint['X_std']
        self.is_trained = True
        print(f"✓ 已加载PINN模型: {path}")
    
    def predict_batch(self, sigma_m_array: np.ndarray) -> np.ndarray:
        """批量预测疲劳寿命"""
        if not self.is_trained:
            raise RuntimeError("模型未训练或加载")
        
        n = len(sigma_m_array)
        
        X = np.zeros((n, 5))
        X[:, 0] = self.config.epsilon_f_prime
        X[:, 1] = self.config.sigma_f_prime
        X[:, 2] = self.config.c
        X[:, 3] = self.config.E
        X[:, 4] = sigma_m_array
        
        X_norm = (X - self.X_mean) / self.X_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            log_Nf = self.model(X_tensor).numpy().flatten()
        
        return 10 ** log_Nf


# ==============================================================================
# 整合分析
# ==============================================================================

class IntegratedFatigueAnalyzer:
    """整合分析器：UNet应力 → PINN疲劳寿命"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pinn = PINNFatiguePredictor(config)
        self.results = None
    
    def load_unet_results(self) -> pd.DataFrame:
        """加载UNet评估结果"""
        if not os.path.exists(self.config.unet_results_csv):
            raise FileNotFoundError(f"UNet结果文件不存在: {self.config.unet_results_csv}")
        
        df = pd.read_csv(self.config.unet_results_csv)
        print(f"✓ 已加载UNet评估结果: {len(df)} 个样本")
        
        return df
    
    def run_analysis(self, retrain_pinn: bool = False) -> pd.DataFrame:
        """运行完整分析"""
        print("\n" + "=" * 70)
        print("UNet应力 + PINN疲劳寿命 整合分析")
        print("=" * 70)
        
        print("\n[1/4] 加载UNet应力预测结果...")
        df = self.load_unet_results()
        
        print("\n[2/4] 准备PINN模型...")
        if retrain_pinn or not os.path.exists(self.config.pinn_model_path):
            self.pinn.train(verbose=True)
        else:
            self.pinn.load()
        
        print("\n[3/4] 计算疲劳寿命...")
        
        true_stress = df['true_max_mpa'].values
        Nf_true_pinn = self.pinn.predict_batch(true_stress)
        
        pred_stress = df['pred_max_mpa'].values
        Nf_pred_pinn = self.pinn.predict_batch(pred_stress)
        
        print("  计算二分法参考值...")
        Nf_true_bisec = []
        Nf_pred_bisec = []
        
        for i in tqdm(range(len(df)), desc="  二分法计算"):
            nf_true = solve_bisection(
                self.config.epsilon_f_prime, self.config.sigma_f_prime,
                self.config.c, self.config.E, true_stress[i], self.config.b
            )
            nf_pred = solve_bisection(
                self.config.epsilon_f_prime, self.config.sigma_f_prime,
                self.config.c, self.config.E, pred_stress[i], self.config.b
            )
            Nf_true_bisec.append(nf_true)
            Nf_pred_bisec.append(nf_pred)
        
        Nf_true_bisec = np.array(Nf_true_bisec)
        Nf_pred_bisec = np.array(Nf_pred_bisec)
        
        print("\n[4/4] 汇总分析结果...")
        
        df['Nf_true_pinn'] = Nf_true_pinn
        df['Nf_pred_pinn'] = Nf_pred_pinn
        df['Nf_true_bisec'] = Nf_true_bisec
        df['Nf_pred_bisec'] = Nf_pred_bisec
        
        df['Nf_error_pinn'] = np.abs(Nf_pred_pinn - Nf_true_pinn)
        df['Nf_relative_error_pinn_%'] = np.abs(Nf_pred_pinn - Nf_true_pinn) / Nf_true_pinn * 100
        
        df['Nf_error_bisec'] = np.abs(Nf_pred_bisec - Nf_true_bisec)
        df['Nf_relative_error_bisec_%'] = np.abs(Nf_pred_bisec - Nf_true_bisec) / Nf_true_bisec * 100
        
        # 使用Bisection作为真值计算log_Nf
        df['log_Nf_true'] = np.log10(Nf_true_bisec)
        df['log_Nf_pred'] = np.log10(Nf_pred_pinn)
        df['log_Nf_error'] = np.abs(df['log_Nf_pred'] - df['log_Nf_true'])
        
        self.results = df
        
        return df
    
    def print_summary(self):
        """打印分析摘要"""
        if self.results is None:
            raise RuntimeError("请先运行 run_analysis()")
        
        df = self.results
        
        print("\n" + "=" * 70)
        print("分析结果摘要")
        print("=" * 70)
        
        print("\n【应力预测误差 (UNet)】")
        print(f"  样本数: {len(df)}")
        print(f"  最大应力MAE: {df['max_stress_error_mpa'].mean():.3f} MPa")
        print(f"  最大应力相对误差: {df['max_stress_relative_error_%'].mean():.2f}%")
        
        print("\n【疲劳寿命误差 (PINN, 由应力误差传递)】")
        print(f"  Nf相对误差 (PINN): {df['Nf_relative_error_pinn_%'].mean():.2f}%")
        print(f"  Nf相对误差 (二分法): {df['Nf_relative_error_bisec_%'].mean():.2f}%")
        print(f"  log10(Nf)绝对误差: {df['log_Nf_error'].mean():.4f}")
        
        print("\n【疲劳寿命分布】")
        print(f"  真实Nf范围: {df['Nf_true_bisec'].min():.2e} ~ {df['Nf_true_bisec'].max():.2e}")
        print(f"  预测Nf范围: {df['Nf_pred_pinn'].min():.2e} ~ {df['Nf_pred_pinn'].max():.2e}")
        
        corr = df['max_stress_error_mpa'].corr(df['Nf_relative_error_pinn_%'])
        print(f"\n【误差传递分析】")
        print(f"  应力误差 vs Nf误差 相关系数: {corr:.4f}")
        
        print("\n【Nf相对误差分位数 (PINN)】")
        for p in [50, 75, 90, 95, 99]:
            val = df['Nf_relative_error_pinn_%'].quantile(p/100)
            print(f"    P{p}: {val:.2f}%")
    
    # ==========================================================================
    # 论文图表生成 - Fig 11 和 Fig 12
    # ==========================================================================
    
    def create_fig11(self) -> str:
        """
        生成 Fig 11: Fatigue Life Prediction vs True (散点图)
        
        使用Bisection作为真值，PINN作为预测值
        对应论文 Section V-E
        """
        if self.results is None:
            raise RuntimeError("请先运行 run_analysis()")
        
        df = self.results
        
        # 计算R²
        from sklearn.metrics import r2_score
        r2 = r2_score(df['log_Nf_true'], df['log_Nf_pred'])
        mae = df['log_Nf_error'].mean()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 散点图 - 使用浅蓝色
        ax.scatter(df['log_Nf_true'], df['log_Nf_pred'], 
                   alpha=0.5, s=25, c='#6BAED6', edgecolors='none')
        
        # 完美预测线
        min_val = min(df['log_Nf_true'].min(), df['log_Nf_pred'].min()) - 0.05
        max_val = max(df['log_Nf_true'].max(), df['log_Nf_pred'].max()) + 0.05
        ax.plot([min_val, max_val], [min_val, max_val], 
                color='#C0392B', linestyle='--', linewidth=2, label='Perfect')
        
        # 标签 - 使用粗体
        ax.set_xlabel(r'$True$ $\log_{10}(N_f)$ $[Bisection]$', fontsize=20)
        ax.set_ylabel(r'$Predicted$ $\log_{10}(N_f)$ $[PINN]$', fontsize=20)
        
        # 标题 - 包含R²
        ax.set_title(f'End-to-End Fatigue Life Prediction (R²={r2:.4f})', 
                     fontsize=22)
        
        # 图例 - 左上角
        ax.legend(loc='upper left', fontsize=18)
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 设置坐标轴范围和比例
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', labelsize=18)
        
        # 统计信息框 - 右下角，浅黄色背景
        stats_text = f'MAE: {mae:.4f}\nSamples: {len(df)}'
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', 
                         edgecolor='gray', alpha=0.9))
        
        plt.tight_layout()
        
        # 保存PNG
        output_path = os.path.join(self.config.output_dir, 'fig11_fatigue_life_prediction.png')
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Fig 11 已保存: {output_path}")
        
        return output_path
    
    def create_fig12(self) -> str:
        """
        生成 Fig 12: Error Propagation Analysis (2子图)
        
        左图: 应力误差 vs Nf误差 散点图
        右图: 误差传递链柱状图 (4个指标，调整顺序)
        
        对应论文 Section V-E
        """
        if self.results is None:
            raise RuntimeError("请先运行 run_analysis()")
        
        df = self.results
        
        # 创建1x2子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ==================== 左图：误差传递散点图 ====================
        ax1 = axes[0]
        
        ax1.scatter(df['max_stress_error_mpa'], df['Nf_relative_error_pinn_%'], 
                   alpha=0.5, s=25, c='coral', edgecolors='none')
        
        # 添加线性趋势线
        z = np.polyfit(df['max_stress_error_mpa'], df['Nf_relative_error_pinn_%'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['max_stress_error_mpa'].min(), 
                            df['max_stress_error_mpa'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2.5, 
                label=f'Trend (slope={z[0]:.2f})')
        
        ax1.set_xlabel('Stress Error (MPa)', fontsize=18)
        ax1.set_ylabel(r'$N_f$ Relative Error (%)', fontsize=20)
        ax1.set_title('Error Propagation: Stress → Fatigue Life', 
                     fontsize=22)
        ax1.legend(loc='upper left', fontsize=18)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.tick_params(axis='both', labelsize=18)
        
        # 设置坐标轴范围
        ax1.set_xlim(0, df['max_stress_error_mpa'].max() * 1.05)
        ax1.set_ylim(0, df['Nf_relative_error_pinn_%'].max() * 1.05)
        
        # ==================== 右图：误差链柱状图 ====================
        ax2 = axes[1]
        
        # 计算误差值
        stress_mae = df['max_stress_error_mpa'].mean()
        stress_rel_err = df['max_stress_relative_error_%'].mean()
        nf_rel_err = df['Nf_relative_error_pinn_%'].mean()
        log_nf_mae = df['log_Nf_error'].mean()
        
        # 4个指标，调整顺序：第3和第4交换位置
        categories = [
            'UNet Stress\nMAE', 
            'UNet Stress\nRel.Err', 
            r'$\log_{10}(N_f)$' + '\nMAE',      # 原第4位 -> 现第3位
            r'PINN $N_f$' + '\nRel.Err'         # 原第3位 -> 现第4位
        ]
        
        values = [
            stress_mae,        # 1.67 MPa
            stress_rel_err,    # 0.56%
            log_nf_mae,        # 0.0252 (不放大)
            nf_rel_err         # 5.69%
        ]
        
        # 颜色：蓝色(UNet相关)，绿色(Nf相关)
        colors = ['#3498db', '#3498db', '#2ecc71', '#2ecc71']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.85, 
                      edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        labels = [
            f'{stress_mae:.2f} MPa', 
            f'{stress_rel_err:.2f}%', 
            f'{log_nf_mae:.4f}',
            f'{nf_rel_err:.2f}%'
        ]
        
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.02,
                    label, ha='center', va='bottom', fontsize=16)
        
        ax2.set_ylabel('Error Value', fontsize=18)
        ax2.set_title('Error Chain: Power → Stress → Fatigue Life', 
                     fontsize=20)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax2.tick_params(axis='both', labelsize=16)
        
        # 设置y轴范围
        ax2.set_ylim(0, max(values) * 1.25)
        
        plt.tight_layout()
        
        # 保存PNG
        output_path = os.path.join(self.config.output_dir, 'fig12_error_propagation_analysis.png')
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Fig 12 已保存: {output_path}")
        
        return output_path
    
    def create_paper_figures(self):
        """生成所有论文用图"""
        print("\n" + "=" * 70)
        print("生成论文图表")
        print("=" * 70)
        
        fig11_path = self.create_fig11()
        fig12_path = self.create_fig12()
        
        print("\n" + "-" * 70)
        print("论文图表生成完成!")
        print(f"  Fig 11: {fig11_path}")
        print(f"  Fig 12: {fig12_path}")
        print("-" * 70)
        
        return fig11_path, fig12_path
    
    def save_results(self):
        """保存结果"""
        if self.results is None:
            raise RuntimeError("请先运行 run_analysis()")
        
        # 保存CSV
        csv_path = os.path.join(self.config.output_dir, 'integrated_fatigue_results.csv')
        self.results.to_csv(csv_path, index=False)
        print(f"✓ 详细结果: {csv_path}")
        
        # 保存摘要
        summary_path = os.path.join(self.config.output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("UNet应力 + PINN疲劳寿命 整合分析摘要\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("【材料参数 (焊料层)】\n")
            f.write(f"  ε'f (疲劳延性系数): {self.config.epsilon_f_prime}\n")
            f.write(f"  σ'f (疲劳强度系数): {self.config.sigma_f_prime} MPa\n")
            f.write(f"  c (疲劳延性指数): {self.config.c}\n")
            f.write(f"  E (弹性模量): {self.config.E} MPa\n")
            f.write(f"  b (疲劳强度指数): {self.config.b}\n\n")
            
            df = self.results
            f.write("【应力预测误差 (UNet)】\n")
            f.write(f"  样本数: {len(df)}\n")
            f.write(f"  最大应力MAE: {df['max_stress_error_mpa'].mean():.4f} MPa\n")
            f.write(f"  最大应力相对误差: {df['max_stress_relative_error_%'].mean():.2f}%\n\n")
            
            f.write("【疲劳寿命预测结果 (PINN)】\n")
            from sklearn.metrics import r2_score
            r2 = r2_score(df['log_Nf_true'], df['log_Nf_pred'])
            f.write(f"  R²: {r2:.4f}\n")
            f.write(f"  Nf相对误差: {df['Nf_relative_error_pinn_%'].mean():.2f}%\n")
            f.write(f"  log10(Nf)绝对误差: {df['log_Nf_error'].mean():.4f}\n")
            f.write(f"  真实Nf范围: {df['Nf_true_bisec'].min():.2e} ~ {df['Nf_true_bisec'].max():.2e}\n\n")
            
            # 误差传递分析
            z = np.polyfit(df['max_stress_error_mpa'], df['Nf_relative_error_pinn_%'], 1)
            amplification = df['Nf_relative_error_pinn_%'].mean() / df['max_stress_relative_error_%'].mean()
            f.write("【误差传递分析】\n")
            f.write(f"  误差传递斜率: {z[0]:.2f} (%/MPa)\n")
            f.write(f"  误差放大倍数: {amplification:.1f}x\n\n")
            
            f.write("【误差分位数】\n")
            for p in [50, 75, 90, 95, 99]:
                val = df['Nf_relative_error_pinn_%'].quantile(p/100)
                f.write(f"  P{p}: {val:.2f}%\n")
        
        print(f"✓ 分析摘要: {summary_path}")


# ==============================================================================
# 主函数
# ==============================================================================

def main(config: Config = None, retrain_pinn: bool = False):
    """主函数"""
    
    if config is None:
        config = Config()
    
    analyzer = IntegratedFatigueAnalyzer(config)
    
    # 运行分析
    df = analyzer.run_analysis(retrain_pinn=retrain_pinn)
    
    # 打印摘要
    analyzer.print_summary()
    
    # 生成论文图表 (Fig 11 和 Fig 12)
    analyzer.create_paper_figures()
    
    # 保存结果
    analyzer.save_results()
    
    print("\n" + "=" * 70)
    print("✅ 整合分析完成!")
    print(f"   输出目录: {config.output_dir}")
    print("=" * 70)
    
    return analyzer


# ==============================================================================
# 命令行接口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet应力 + PINN疲劳寿命 整合分析')
    
    parser.add_argument('--unet-csv', type=str, 
                       default="/home/txr/code/final_version/2_unet_architecture/evaluation_result/batch_stress_analysis/batch_stress_evaluation_results.csv",
                       help='UNet评估结果CSV路径')
    
    parser.add_argument('--output-dir', '-o', type=str,
                       default='/home/txr/code/final_version/3_pinn_architecture/integrated_fatigue_analysis',
                       help='输出目录')
    
    parser.add_argument('--retrain', action='store_true',
                       help='重新训练PINN模型')
    
    parser.add_argument('--epochs', type=int, default=10000,
                       help='PINN训练轮次')
    
    # 材料参数
    parser.add_argument('--epsilon-f', type=float, default=0.325,
                       help='疲劳延性系数')
    parser.add_argument('--sigma-f', type=float, default=1200.0,
                       help='疲劳强度系数 (MPa)')
    parser.add_argument('--c', type=float, default=-0.57,
                       help='疲劳延性指数')
    parser.add_argument('--E', type=float, default=45000.0,
                       help='弹性模量 (MPa)')
    
    args = parser.parse_args()
    
    config = Config(
        unet_results_csv=args.unet_csv,
        output_dir=args.output_dir,
        pinn_epochs=args.epochs,
        epsilon_f_prime=args.epsilon_f,
        sigma_f_prime=args.sigma_f,
        c=args.c,
        E=args.E
    )
    
    main(config, retrain_pinn=args.retrain)