#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
统一训练脚本 - 功率→温度/应力预测（通用版）
=============================================================================
主要特性：
✅ 支持温度预测和应力预测（通过配置切换）
✅ 统一的UNet-V2 + EMA注意力架构
✅ 标准MSE损失函数
✅ 多GPU支持（DataParallel）
✅ ReduceLROnPlateau自适应学习率
✅ 完整的训练日志和可视化

使用方法：
    在main()函数中配置task_type即可：
    config.task_type = 'power_to_temperature'  # 温度预测
    config.task_type = 'power_to_stress'       # 应力预测
=============================================================================
"""

import os
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import random
import json
from datetime import datetime
import time
import warnings
from tqdm import tqdm
from pathlib import Path

# 彩色输出
from colorama import init, Fore, Back, Style

init(autoreset=True)
warnings.filterwarnings('ignore')

# 导入统一的模型
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from unet_2d import UNetV2_PowerToField


# ==================== 彩色输出工具 ====================

class ColorPrinter:
    """彩色输出工具"""
    
    @staticmethod
    def header(text):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*70}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{text}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*70}{Style.RESET_ALL}")
    
    @staticmethod
    def success(text):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(text):
        print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")
    
    @staticmethod
    def error(text):
        print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def info(text):
        print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")
    
    @staticmethod
    def best(text):
        print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} {text} {Style.RESET_ALL}")
    
    @staticmethod
    def lr_update(epoch, old_lr, new_lr, reason=""):
        """学习率更新提示"""
        reduction_ratio = (old_lr - new_lr) / old_lr * 100 if old_lr > 0 else 0
        print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Epoch {epoch}: 学习率衰减{Style.RESET_ALL}")
        print(f"   {old_lr:.2e} → {Fore.GREEN}{new_lr:.2e}{Style.RESET_ALL} "
              f"({Fore.RED}↓{reduction_ratio:.1f}%{Style.RESET_ALL})")
        if reason:
            print(f"   原因: {Fore.CYAN}{reason}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")


cp = ColorPrinter()


# ==================== 工具函数 ====================

def set_seed(seed=42):
    """设置所有随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def robust_normalize(data, percentile_low=1, percentile_high=99):
    """鲁棒归一化（用于功率数据）"""
    flat_data = data.flatten()
    non_zero = flat_data[flat_data > 0]
    
    if len(non_zero) > 0:
        p_low = np.percentile(non_zero, percentile_low)
        p_high = np.percentile(non_zero, percentile_high)
    else:
        p_low = 0
        p_high = np.max(flat_data)
    
    data_clipped = np.clip(data, p_low, p_high)
    
    if p_high > p_low:
        normalized = (data_clipped - p_low) / (p_high - p_low)
    else:
        normalized = np.zeros_like(data)
    
    median = np.median(flat_data)
    q75, q25 = np.percentile(flat_data, [75, 25])
    iqr = q75 - q25
    
    stats = {
        'method': 'robust',
        'p_low': float(p_low),
        'p_high': float(p_high),
        'median': float(median),
        'iqr': float(iqr) if iqr > 1e-8 else 1.0
    }
    
    return normalized, stats


def zscore_normalize(data):
    """Z-score归一化（用于温度/应力数据）"""
    mean = np.mean(data)
    std = np.std(data)
    
    if std > 1e-8:
        normalized = (data - mean) / std
    else:
        normalized = data - mean
    
    stats = {
        'method': 'zscore',
        'mean': float(mean),
        'std': float(std) if std > 1e-8 else 1.0
    }
    
    return normalized, stats


# ==================== GPU配置 ====================

def check_gpu_memory():
    """检查GPU内存状态"""
    if not torch.cuda.is_available():
        cp.warning("CUDA不可用")
        return
    
    cp.header("🖥️  GPU内存状态")
    
    available_gpus = torch.cuda.device_count()
    print(f"可用GPU总数: {Fore.YELLOW}{available_gpus}{Style.RESET_ALL}\n")
    
    for i in range(available_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        free_memory = total_memory - allocated_memory
        usage_percent = (allocated_memory / total_memory) * 100
        
        if usage_percent > 90:
            color = Fore.RED
            status = "⚠️  几乎满"
        elif usage_percent > 70:
            color = Fore.YELLOW
            status = "⚠️  较满"
        else:
            color = Fore.GREEN
            status = "✅ 空闲"
        
        print(f"{Fore.CYAN}GPU {i}: {gpu_name}{Style.RESET_ALL}")
        print(f"  总内存: {total_memory:.1f} GB")
        print(f"  已用: {allocated_memory:.1f} GB ({usage_percent:.1f}%) {color}{status}{Style.RESET_ALL}")
        print(f"  空闲: {free_memory:.1f} GB")
        print()


def setup_gpus(gpu_ids=None):
    """配置GPU设备"""
    if not torch.cuda.is_available():
        cp.warning("CUDA不可用，将使用CPU")
        return torch.device('cpu'), []
    
    check_gpu_memory()
    
    if gpu_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    elif isinstance(gpu_ids, int):
        device_ids = [gpu_ids]
    elif isinstance(gpu_ids, str):
        device_ids = [int(x.strip()) for x in gpu_ids.split(',')]
    elif isinstance(gpu_ids, (list, tuple)):
        device_ids = list(gpu_ids)
    else:
        raise ValueError(f"gpu_ids格式错误: {gpu_ids}")
    
    available_gpus = torch.cuda.device_count()
    for gpu_id in device_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id} 不存在（只有{available_gpus}个GPU）")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    device = torch.device('cuda:0')
    
    cp.header("🎯 选择的GPU设备")
    cp.success(f"使用GPU: {device_ids}")
    
    for gpu_id in device_ids:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"  {Fore.GREEN}GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB){Style.RESET_ALL}")
    
    print()
    return device, device_ids


# ==================== 学习率调度器 ====================

class ReduceLROnPlateauScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-7, verbose=True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = None
        self.wait_count = 0
        self.total_reductions = 0
    
    def step(self, val_loss, epoch=0):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"{Fore.BLUE}  初始验证损失: {val_loss:.6f}{Style.RESET_ALL}")
            return {'reduced': False, 'new_lr': self.optimizer.param_groups[0]['lr'],
                    'old_lr': self.optimizer.param_groups[0]['lr']}
        
        improvement_threshold = 1e-4
        
        if val_loss < self.best_loss - improvement_threshold:
            self.best_loss = val_loss
            self.wait_count = 0
            if self.verbose:
                print(f"{Fore.GREEN}  ✓ 验证损失改善: {val_loss:.6f}{Style.RESET_ALL}")
            return {'reduced': False, 'new_lr': self.optimizer.param_groups[0]['lr'],
                    'old_lr': self.optimizer.param_groups[0]['lr']}
        else:
            self.wait_count += 1
            if self.verbose:
                remaining = self.patience - self.wait_count
                if remaining > 0:
                    print(f"{Fore.YELLOW}  ⊘ 未改善 ({self.wait_count}/{self.patience}，"
                          f"还有{remaining}次机会){Style.RESET_ALL}")
            
            if self.wait_count >= self.patience:
                old_lr = self.optimizer.param_groups[0]['lr']
                self._reduce_lr(epoch)
                self.wait_count = 0
                new_lr = self.optimizer.param_groups[0]['lr']
                return {'reduced': True, 'new_lr': new_lr, 'old_lr': old_lr}
            
            return {'reduced': False, 'new_lr': self.optimizer.param_groups[0]['lr'],
                    'old_lr': self.optimizer.param_groups[0]['lr']}
    
    def _reduce_lr(self, epoch=0):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
        self.total_reductions += 1


# ==================== 数据加载器工厂 ====================

def create_dataloaders_for_task(task_type, config, train_ids, val_ids, test_ids,
                                power_map, target_map, common_ids):
    """
    根据任务类型创建DataLoader
    
    参数:
        task_type: 'power_to_temperature' 或 'power_to_stress'
        config: 训练配置
        train_ids, val_ids, test_ids: 数据集划分ID
        power_map: 功率文件映射
        target_map: 目标文件映射（温度或应力）
        common_ids: 公共ID列表
    """
    # 准备文件列表
    all_ids = sorted(common_ids)
    power_file_list = [power_map[i] for i in all_ids]
    target_file_list = [target_map[i] for i in all_ids]
    file_ids_list = all_ids
    
    id_to_idx = {file_id: idx for idx, file_id in enumerate(all_ids)}
    train_indices = [id_to_idx[i] for i in train_ids]
    val_indices = [id_to_idx[i] for i in val_ids]
    test_indices = [id_to_idx[i] for i in test_ids] if test_ids else None
    
    # 🔧 导入统一的数据准备模块
    from in_out_data_preparation import create_power_temp_dataloaders, create_power_stress_dataloaders
    
    # 根据任务类型调用对应的函数
    if task_type == 'power_to_temperature':
        cp.info("创建功率→温度数据加载器")
        train_loader, val_loader, test_loader, dataset = create_power_temp_dataloaders(
            power_files=power_file_list,
            temp_files=target_file_list,
            file_ids=file_ids_list,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=test_indices,
            flp_file_path=config.flp_file_path,
            grid_res=config.grid_res,
            batch_size=config.batch_size,
            cache_dir=config.cache_dir,
            precision=config.precision,
            num_workers=config.num_workers,
            padding_x=config.padding_x,
            padding_y=config.padding_y
        )
    elif task_type == 'power_to_stress':
        cp.info("创建功率→应力数据加载器")
        train_loader, val_loader, test_loader, dataset = create_power_stress_dataloaders(
            power_files=power_file_list,
            stress_files=target_file_list,
            file_ids=file_ids_list,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=test_indices,
            flp_file_path=config.flp_file_path,
            grid_res=config.grid_res,
            batch_size=config.batch_size,
            cache_dir=config.cache_dir,
            precision=config.precision,
            num_workers=config.num_workers,
            padding_x=config.padding_x,
            padding_y=config.padding_y
        )
    else:
        raise ValueError(f"未知任务类型: {task_type}")
    
    return train_loader, val_loader, test_loader, dataset


# ==================== 数据集配置 ====================

class DatasetConfig:
    """数据集划分配置"""
    
    def __init__(self, mode='id_range'):
        self.mode = mode
        self.train_id_range = None
        self.val_id_range = None
        self.test_id_range = None
    
    def split_data(self, common_ids):
        """根据配置划分数据集"""
        common_ids = sorted(common_ids)
        
        if self.mode == 'id_range':
            train_ids = []
            val_ids = []
            test_ids = []
            
            if self.train_id_range:
                train_start, train_end = self.train_id_range
                train_ids = [i for i in common_ids if train_start <= i <= train_end]
            
            if self.val_id_range:
                val_start, val_end = self.val_id_range
                val_ids = [i for i in common_ids if val_start <= i <= val_end]
            
            if self.test_id_range:
                test_start, test_end = self.test_id_range
                test_ids = [i for i in common_ids if test_start <= i <= test_end]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return train_ids, val_ids, test_ids


# ==================== 训练配置 ====================

class TrainingConfig:
    """统一训练配置类"""
    
    def __init__(self):
        # ============================================================
        # 🎯 【关键】任务类型选择
        # ============================================================
        self.task_type = 'power_to_temperature'  # 'power_to_temperature' 或 'power_to_stress'
        
        # ============================================================
        # 📁 数据路径配置（根据任务类型需要分别配置）
        # ============================================================
        # 功率数据目录（输入，两个任务共用）
        self.power_dir = "./power_data"
        
        # 温度预测相关路径
        self.temp_dir = "./temp_data"           # 温度数据目录（输出）
        
        # 应力预测相关路径
        self.stress_dir = "./stress_data"       # 应力数据目录（输出）
        
        # FLP文件路径（可选）
        self.flp_file_path = None
        
        # ============================================================
        # 🎮 GPU配置
        # ============================================================
        self.gpu_ids = None  # None=使用所有GPU, "0"=单GPU, "0,1,2,3"=多GPU
        self.device_ids = []
        
        # ============================================================
        # 📊 数据集配置
        # ============================================================
        self.dataset_config = DatasetConfig(mode='id_range')
        
        # ============================================================
        # 🔧 数据处理
        # ============================================================
        self.grid_res = (256, 256)
        self.use_cache = True
        self.num_workers = 4
        self.cache_dir = None
        self.precision = 'float64'
        
        # FLP padding参数
        self.padding_x = 3.6
        self.padding_y = 3.43
        
        # ============================================================
        # 🏗️ 模型配置
        # ============================================================
        self.in_channels = 1
        self.out_channels = 1
        self.base_ch = 32
        self.use_attention = True  # 是否使用EMA注意力
        
        # ============================================================
        # 🎓 训练参数
        # ============================================================
        self.epochs = 300
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.use_amp = False
        
        # ============================================================
        # 📉 学习率调度器
        # ============================================================
        self.use_scheduler = True
        self.plateau_patience = 10
        self.plateau_factor = 0.5
        self.plateau_min_lr = 1e-5
        
        # ============================================================
        # ⏹️ 早停
        # ============================================================
        self.early_stopping = True
        self.early_stopping_patience = 20
        
        # ============================================================
        # 💾 保存配置
        # ============================================================
        self.save_dir = "./training_logs"
        self.print_freq = 1
        self.save_freq = 50
        
        # ============================================================
        # 📈 误差监控
        # ============================================================
        self.monitor_errors = True
        # 温度误差阈值（K）
        self.temp_error_thresholds = [1.0, 5.0, 10.0, 20.0, 30.0]
        # 应力误差阈值（Pa）
        self.stress_error_thresholds = [1e7, 5e7, 10e7, 20e7, 25e7, 30e7]
    
    def get_target_dir(self):
        """根据任务类型获取目标数据目录"""
        if self.task_type == 'power_to_temperature':
            return self.temp_dir
        elif self.task_type == 'power_to_stress':
            return self.stress_dir
        else:
            raise ValueError(f"未知任务类型: {self.task_type}")
    
    def get_error_thresholds(self):
        """根据任务类型获取误差阈值 - 修复版"""
        if self.task_type == 'power_to_temperature':
            return self.temp_error_thresholds
        elif self.task_type == 'power_to_stress':
            return self.stress_error_thresholds
        else:
            raise ValueError(f"未知任务类型: {self.task_type}")
    
    def get_target_unit(self):
        """获取目标单位 - 修复版"""
        if self.task_type == 'power_to_temperature':
            return 'K'
        elif self.task_type == 'power_to_stress':
            return 'Pa'
        else:
            raise ValueError(f"未知任务类型: {self.task_type}")
    
    def to_dict(self):
        """转换为字典"""
        d = {}
        for key, value in self.__dict__.items():
            if key == 'dataset_config':
                continue
            if isinstance(value, (int, float, str, bool, list, tuple)) or value is None:
                d[key] = value
        return d
    
    def print_config(self):
        """打印配置"""
        cp.header("⚙️  配置信息")
        
        # 任务类型
        task_emoji = "🌡️" if self.task_type == 'power_to_temperature' else "💪"
        task_name = "温度预测" if self.task_type == 'power_to_temperature' else "应力预测"
        print(f"{Fore.CYAN}  任务类型:{Style.RESET_ALL} {task_emoji} {Fore.GREEN}{Style.BRIGHT}{task_name}{Style.RESET_ALL}")
        
        # 数据路径
        print(f"{Fore.CYAN}  数据目录:{Style.RESET_ALL}")
        print(f"    输入（功率）: {self.power_dir}")
        print(f"    输出（{self.get_target_unit()}）: {self.get_target_dir()}")
        if self.flp_file_path:
            print(f"    FLP: {self.flp_file_path}")
        if self.cache_dir:
            print(f"    缓存: {self.cache_dir}")
        
        # GPU配置
        print(f"{Fore.CYAN}  GPU配置:{Style.RESET_ALL} {self.gpu_ids if self.gpu_ids else '使用所有GPU'}")
        
        # 模型配置
        print(f"{Fore.CYAN}  模型配置:{Style.RESET_ALL}")
        print(f"    UNet-V2 + EMA注意力")
        print(f"    基础通道: {self.base_ch}")
        print(f"    EMA注意力: {Fore.GREEN if self.use_attention else Fore.RED}{'启用' if self.use_attention else '禁用'}{Style.RESET_ALL}")
        
        # 训练参数
        print(f"{Fore.CYAN}  训练参数:{Style.RESET_ALL}")
        print(f"    Epochs: {Fore.YELLOW}{self.epochs}{Style.RESET_ALL}")
        print(f"    Batch size: {Fore.YELLOW}{self.batch_size}{Style.RESET_ALL}")
        print(f"    Learning rate: {Fore.YELLOW}{self.learning_rate}{Style.RESET_ALL}")
        print(f"    损失函数: {Fore.YELLOW}MSE{Style.RESET_ALL}")
        
        # 学习率调度器
        print(f"{Fore.CYAN}  学习率调度器:{Style.RESET_ALL}")
        print(f"    类型: {Fore.YELLOW}ReduceLROnPlateau{Style.RESET_ALL}")
        print(f"    Patience: {Fore.YELLOW}{self.plateau_patience}{Style.RESET_ALL} epoch")
        print(f"    衰减因子: {Fore.YELLOW}{self.plateau_factor}{Style.RESET_ALL}")
        print(f"    最小LR: {Fore.YELLOW}{self.plateau_min_lr:.0e}{Style.RESET_ALL}")


# ==================== 文件匹配 ====================

def extract_file_id(filepath: str) -> int:
    """从文件路径中提取数字ID"""
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return None


def match_files(power_dir, target_dir, task_type):
    """
    扫描并匹配功率和目标文件
    
    参数:
        power_dir: 功率数据目录
        target_dir: 目标数据目录（温度或应力）
        task_type: 'power_to_temperature' 或 'power_to_stress'
    """
    all_power_files = sorted(glob.glob(os.path.join(power_dir, "*.txt")) + 
                            glob.glob(os.path.join(power_dir, "*.csv")))
    
    # 根据任务类型匹配不同的目标文件
    if task_type == 'power_to_temperature':
        all_target_files = sorted(glob.glob(os.path.join(target_dir, "temp*.txt")) + 
                                 glob.glob(os.path.join(target_dir, "temp*.csv")))
        target_name = "温度"
    elif task_type == 'power_to_stress':
        all_target_files = sorted(glob.glob(os.path.join(target_dir, "strength*.txt")) + 
                                 glob.glob(os.path.join(target_dir, "strength*.csv")))
        target_name = "应力"
    else:
        raise ValueError(f"未知任务类型: {task_type}")
    
    cp.header("📁 文件扫描")
    print(f"  功率文件: {Fore.YELLOW}{len(all_power_files)}{Style.RESET_ALL} 个")
    print(f"  {target_name}文件: {Fore.YELLOW}{len(all_target_files)}{Style.RESET_ALL} 个")
    
    power_map = {}
    for f in all_power_files:
        file_id = extract_file_id(f)
        if file_id is not None:
            power_map[file_id] = f
    
    target_map = {}
    for f in all_target_files:
        file_id = extract_file_id(f)
        if file_id is not None:
            target_map[file_id] = f
    
    common_ids = sorted(set(power_map.keys()) & set(target_map.keys()))
    
    print(f"\n{Fore.GREEN}  成功匹配: {len(common_ids)} 对{Style.RESET_ALL}")
    if len(common_ids) > 0:
        print(f"  ID范围: {min(common_ids)} - {max(common_ids)}")
    
    return power_map, target_map, common_ids


# ==================== 误差统计 ====================

def compute_error_statistics(pred, target, target_stats, thresholds):
    """计算详细的误差统计"""
    with torch.no_grad():
        errors_norm = torch.abs(pred - target)
        std = target_stats['std']
        errors_real = errors_norm * std
        
        max_error = errors_real.max().item()
        mean_error = errors_real.mean().item()
        median_error = errors_real.median().item()
        
        total_pixels = errors_real.numel()
        stats = {
            'max_error': max_error,
            'mean_error': mean_error,
            'median_error': median_error,
        }
        
        for threshold in thresholds:
            pixels_above = (errors_real > threshold).sum().item()
            percent_above = (pixels_above / total_pixels) * 100
            stats[f'pixels_above_{threshold}'] = pixels_above
            stats[f'percent_above_{threshold}'] = percent_above
        
        return stats


# ==================== 训练日志 ====================

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.log_file = os.path.join(save_dir, f"training_log_{self.timestamp}.csv")
        self.summary_file = os.path.join(save_dir, f"training_summary_{self.timestamp}.json")
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'time': [],
            'is_best': [],
            'lr_reduced': [],
        }
        
        if config.monitor_errors:
            self.history.update({
                'max_error': [],
                'mean_error': [],
                'median_error': [],
            })
        
        self.best_stats = {
            'val_loss': float('inf'),
            'val_loss_epoch': 0,
            'max_error': float('inf'),
            'max_error_epoch': 0,
            'mean_error': float('inf'),
            'mean_error_epoch': 0,
        }
        
        self.start_time = time.time()
        cp.success("日志记录器已初始化")
    
    def log_epoch(self, epoch, train_loss, val_loss, lr, error_stats=None, lr_reduced=False):
        """记录一个epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)
        self.history['time'].append(time.time() - self.start_time)
        self.history['lr_reduced'].append(1 if lr_reduced else 0)
        
        is_best = val_loss < self.best_stats['val_loss']
        self.history['is_best'].append(1 if is_best else 0)
        
        if is_best:
            self.best_stats['val_loss'] = val_loss
            self.best_stats['val_loss_epoch'] = epoch
        
        if error_stats and self.config.monitor_errors:
            self.history['max_error'].append(error_stats['max_error'])
            self.history['mean_error'].append(error_stats['mean_error'])
            self.history['median_error'].append(error_stats['median_error'])
            
            if error_stats['max_error'] < self.best_stats['max_error']:
                self.best_stats['max_error'] = error_stats['max_error']
                self.best_stats['max_error_epoch'] = epoch
            
            if error_stats['mean_error'] < self.best_stats['mean_error']:
                self.best_stats['mean_error'] = error_stats['mean_error']
                self.best_stats['mean_error_epoch'] = epoch
        
        return is_best
    
    def save(self):
        """保存日志"""
        try:
            df = pd.DataFrame(self.history)
            df.to_csv(self.log_file, index=False)
            cp.success(f"CSV日志已保存: {self.log_file}")
        except Exception as e:
            cp.error(f"CSV保存失败: {e}")
        
        total_time = time.time() - self.start_time
        
        summary = {
            'timestamp': self.timestamp,
            'task_type': self.config.task_type,
            'config': self.config.to_dict(),
            'best_stats': self.best_stats,
            'total_epochs': len(self.history['epoch']),
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        cp.success(f"摘要已保存: {self.summary_file}")
    
    def plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2 if self.config.monitor_errors else 1, figsize=(14, 10))
        if not self.config.monitor_errors:
            axes = axes.reshape(-1, 1)
        
        epochs = self.history['epoch']
        best_epoch = self.best_stats['val_loss_epoch']
        
        # 损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        
        for i, lr_reduced in enumerate(self.history['lr_reduced']):
            if lr_reduced:
                ax.axvline(x=epochs[i], color='red', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        task_name = '温度' if self.config.task_type == 'power_to_temperature' else '应力'
        ax.set_title(f'{task_name}预测训练损失 (红色虚线=学习率衰减)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax = axes[1, 0]
        ax.plot(epochs, self.history['lr'], color='green', linewidth=2)
        
        for i, lr_reduced in enumerate(self.history['lr_reduced']):
            if lr_reduced:
                ax.scatter(epochs[i], self.history['lr'][i], color='red', s=100, zorder=5, marker='v')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (红色三角=衰减点)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if self.config.monitor_errors:
            unit = self.config.get_target_unit()
            
            ax = axes[0, 1]
            ax.plot(epochs, self.history['max_error'], color='red', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'Max Error ({unit})')
            ax.set_title('Maximum Error')
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            ax.plot(epochs, self.history['mean_error'], color='purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'Mean Error ({unit})')
            ax.set_title('Mean Error')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'training_curves_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        cp.success(f"训练曲线已保存: {plot_path}")
        plt.close()


# ==================== 主训练函数 ====================

def train_model(config: TrainingConfig):
    """主训练函数 - 支持温度和应力预测"""
    
    task_emoji = "🌡️" if config.task_type == 'power_to_temperature' else "💪"
    task_name = "温度预测" if config.task_type == 'power_to_temperature' else "应力预测"
    
    cp.header(f"{task_emoji} 统一训练脚本 - {task_name}")
    config.print_config()
    
    # GPU配置
    device, device_ids = setup_gpus(config.gpu_ids)
    config.device_ids = device_ids
    use_multi_gpu = len(device_ids) > 1
    
    if use_multi_gpu:
        cp.success(f"🚀 多GPU模式: 使用 {len(device_ids)} 个GPU")
    else:
        cp.info(f"单GPU模式: GPU {device_ids[0] if device_ids else 'CPU'}")
    
    # 扫描和匹配文件
    cp.header("【1】扫描和匹配文件")
    power_map, target_map, common_ids = match_files(
        config.power_dir, 
        config.get_target_dir(), 
        config.task_type
    )
    
    if len(common_ids) == 0:
        cp.error("没有找到匹配的文件对!")
        raise ValueError()
    
    train_ids, val_ids, test_ids = config.dataset_config.split_data(common_ids)
    
    print(f"\n{Fore.CYAN}数据集划分:{Style.RESET_ALL}")
    print(f"  训练集: {Fore.YELLOW}{len(train_ids)}{Style.RESET_ALL} 样本 (ID: {min(train_ids)}-{max(train_ids)})")
    print(f"  验证集: {Fore.YELLOW}{len(val_ids)}{Style.RESET_ALL} 样本 (ID: {min(val_ids)}-{max(val_ids)})")
    if test_ids:
        print(f"  测试集: {Fore.YELLOW}{len(test_ids)}{Style.RESET_ALL} 样本 (ID: {min(test_ids)}-{max(test_ids)})")
    
    # 创建DataLoader
    cp.header("【2】创建DataLoader")
    train_loader, val_loader, test_loader, dataset = create_dataloaders_for_task(
        config.task_type, config, train_ids, val_ids, test_ids,
        power_map, target_map, common_ids
    )
    
    # 计算归一化统计
    cp.header("【3】计算归一化统计")
    
    sample_size = min(50, len(train_ids))
    sample_indices = np.random.choice(range(len(train_ids)), sample_size, replace=False)
    
    power_values = []
    target_values = []
    
    # 获取实际的数据集索引
    all_ids = sorted(common_ids)
    id_to_idx = {file_id: idx for idx, file_id in enumerate(all_ids)}
    train_dataset_indices = [id_to_idx[i] for i in train_ids]
    
    for i in tqdm(sample_indices, desc=f"{Fore.CYAN}采样并插值{Style.RESET_ALL}", ncols=100):
        try:
            idx = train_dataset_indices[i]
            power_grid, target_grid = dataset[idx]
            power_values.append(power_grid.flatten())
            target_values.append(target_grid.flatten())
        except Exception as e:
            cp.warning(f"样本{i}统计失败: {e}")
            continue
    
    if not power_values:
        cp.error("无法计算归一化统计!")
        raise ValueError()
    
    power_all = np.concatenate(power_values)
    target_all = np.concatenate(target_values)
    
    _, p_stats = robust_normalize(power_all)
    _, t_stats = zscore_normalize(target_all)
    
    unit = config.get_target_unit()
    print(f"  功率: median={Fore.YELLOW}{p_stats.get('median', 0):.4e}{Style.RESET_ALL}, "
          f"iqr={Fore.YELLOW}{p_stats.get('iqr', 1):.4e}{Style.RESET_ALL}")
    print(f"  目标: mean={Fore.YELLOW}{t_stats['mean']:.2e}{unit}{Style.RESET_ALL}, "
          f"std={Fore.YELLOW}{t_stats['std']:.4e}{unit}{Style.RESET_ALL}")
    
    def normalize_batch(power_batch, target_batch):
        """批量归一化"""
        power_norm = power_batch.copy()
        median = p_stats.get('median', 0)
        iqr = p_stats.get('iqr', 1)
        if iqr < 1e-8:
            iqr = 1.0
        nonzero_mask = power_norm > 1e-6
        if nonzero_mask.any():
            power_norm[nonzero_mask] = (power_norm[nonzero_mask] - median) / iqr
        
        target_norm = (target_batch - t_stats['mean']) / t_stats['std']
        
        return power_norm, target_norm
    
    # 创建模型
    cp.header("【4】创建模型")
    
    model = UNetV2_PowerToField(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_ch=config.base_ch,
        use_attention=config.use_attention
    )
    
    # 多GPU包装
    if use_multi_gpu:
        cp.info("使用DataParallel包装模型")
        model = nn.DataParallel(model, device_ids=list(range(len(device_ids))))
    
    model = model.to(device)
    
    # 损失函数（标准MSE）
    criterion = nn.MSELoss()
    cp.success("损失函数: 标准MSE")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateauScheduler(
        optimizer,
        patience=config.plateau_patience,
        factor=config.plateau_factor,
        min_lr=config.plateau_min_lr,
        verbose=True
    )
    
    # 训练日志
    logger = TrainingLogger(config, config.save_dir)
    
    # 混合精度
    scaler = None
    if config.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        cp.info("混合精度训练已启用（AMP）")
    
    # 预热
    cp.header("【5】开始训练")
    cp.info("预热阶段...")
    model.train()
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    cp.success("预热完成")
    
    # ==================== 训练循环 ====================
    
    best_model_state = None
    patience_counter = 0
    
    try:
        for epoch in range(1, config.epochs + 1):
            # === 训练阶段 ===
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, 
                       desc=f"{Fore.BLUE}Epoch {epoch}/{config.epochs} [Train]{Style.RESET_ALL}",
                       leave=False, ncols=100)
            
            for batch_idx, (power_grid, target_grid) in enumerate(pbar):
                power_norm, target_norm = normalize_batch(power_grid.numpy(), target_grid.numpy())
                
                power_norm_t = torch.from_numpy(power_norm).float().unsqueeze(1).to(device)
                target_norm_t = torch.from_numpy(target_norm).float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                if config.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = model(power_norm_t)
                        loss = criterion(pred, target_norm_t)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(power_norm_t)
                    loss = criterion(pred, target_norm_t)
                    loss.backward()
                    optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss * power_grid.shape[0]
                
                pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            
            pbar.close()
            train_loss /= len(train_ids)
            
            # === 验证阶段 ===
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            pbar = tqdm(val_loader,
                       desc=f"{Fore.GREEN}Epoch {epoch}/{config.epochs} [Val]{Style.RESET_ALL}",
                       leave=False, ncols=100)
            
            with torch.no_grad():
                for power_grid, target_grid in pbar:
                    power_norm, target_norm = normalize_batch(power_grid.numpy(), target_grid.numpy())
                    
                    power_norm_t = torch.from_numpy(power_norm).float().unsqueeze(1).to(device)
                    target_norm_t = torch.from_numpy(target_norm).float().unsqueeze(1).to(device)
                    
                    pred = model(power_norm_t)
                    loss = criterion(pred, target_norm_t)
                    val_loss += loss.item() * power_grid.shape[0]
                    
                    val_preds.append(pred.cpu())
                    val_targets.append(target_norm_t.cpu())
                    
                    pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
            
            pbar.close()
            val_loss /= len(val_ids)
            
            # === 计算误差统计 ===
            error_stats = None
            if config.monitor_errors:
                val_preds_all = torch.cat(val_preds, dim=0)
                val_targets_all = torch.cat(val_targets, dim=0)
                error_stats = compute_error_statistics(
                    val_preds_all, val_targets_all, t_stats, config.get_error_thresholds()
                )
            
            # === 学习率调度 ===
            current_lr = optimizer.param_groups[0]['lr']
            lr_reduced = False
            
            if scheduler is not None:
                scheduler_result = scheduler.step(val_loss, epoch=epoch)
                lr_reduced = scheduler_result['reduced']
                current_lr = scheduler_result['new_lr']
                
                if lr_reduced:
                    cp.lr_update(epoch, scheduler_result['old_lr'],
                               scheduler_result['new_lr'],
                               f"验证损失连续{config.plateau_patience}个epoch未改善")
            
            # === 记录日志 ===
            is_best = logger.log_epoch(epoch, train_loss, val_loss, current_lr, error_stats, lr_reduced)
            
            # === 打印进度 ===
            if epoch % config.print_freq == 0:
                unit = config.get_target_unit()
                print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Epoch {epoch}/{config.epochs} [{task_name}]{Style.RESET_ALL}")
                print(f"  Train Loss: {Fore.YELLOW}{train_loss:.6f}{Style.RESET_ALL}")
                print(f"  Val Loss:   {Fore.YELLOW}{val_loss:.6f}{Style.RESET_ALL}")
                print(f"  Current LR: {Fore.MAGENTA}{current_lr:.2e}{Style.RESET_ALL}", end="")
                
                if lr_reduced:
                    print(f" {Fore.RED}[✓ 已衰减]{Style.RESET_ALL}")
                else:
                    print()
                
                if error_stats:
                    if config.task_type == 'power_to_temperature':
                        print(f"  Max Error:  {Fore.YELLOW}{error_stats['max_error']:.3f}{unit}{Style.RESET_ALL}")
                        print(f"  Mean Error: {Fore.YELLOW}{error_stats['mean_error']:.3f}{unit}{Style.RESET_ALL}")
                    else:
                        print(f"  Max Error:  {Fore.YELLOW}{error_stats['max_error']:.3e}{unit}{Style.RESET_ALL}")
                        print(f"  Mean Error: {Fore.YELLOW}{error_stats['mean_error']:.3e}{unit}{Style.RESET_ALL}")
                
                if is_best:
                    cp.best("🌟 新的最佳模型!")
                
                print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
            
            # === 早停检查 ===
            if is_best:
                patience_counter = 0
                if use_multi_gpu:
                    best_model_state = model.module.state_dict().copy()
                else:
                    best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # === 定期保存检查点 ===
            if epoch % config.save_freq == 0:
                checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth')
                model_to_save = model.module if use_multi_gpu else model
                torch.save({
                    'epoch': epoch,
                    'task_type': config.task_type,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config.to_dict(),
                    'p_stats': p_stats,
                    't_stats': t_stats,
                    'train_ids': train_ids,
                    'val_ids': val_ids,
                    'test_ids': test_ids,
                }, checkpoint_path)
                cp.info(f"检查点已保存: epoch {epoch}")
            
            # === 早停 ===
            if config.early_stopping and patience_counter >= config.early_stopping_patience:
                cp.warning(f"早停触发! 验证损失已经{patience_counter}轮未改善")
                break
    
    except KeyboardInterrupt:
        cp.warning("训练被用户中断!")
    
    except Exception as e:
        cp.error(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if best_model_state:
            if use_multi_gpu:
                model.module.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_model_state)
            cp.success(f"已恢复最佳模型 (Epoch {logger.best_stats['val_loss_epoch']})")
        
        logger.save()
        logger.plot_curves()
        
        # 保存最终模型
        model_path = os.path.join(config.save_dir,
                                  f"unet_{config.task_type}_final_{logger.timestamp}.pth")
        model_to_save = model.module if use_multi_gpu else model
        
        torch.save({
            'task_type': config.task_type,
            'model_state_dict': model_to_save.state_dict(),
            'config': config.to_dict(),
            'p_stats': p_stats,
            't_stats': t_stats,
            'best_stats': logger.best_stats,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids,
        }, model_path)
        cp.success(f"最终模型已保存: {model_path}")
    
    stats = {
        'p_stats': p_stats,
        't_stats': t_stats,
        'best_stats': logger.best_stats,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
    }
    
    return model, stats


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 设置随机种子
    set_seed(seed=42)
    
    cp.header("💡 统一训练脚本 - 温度/应力预测")
    
    # ========== 配置 ==========
    config = TrainingConfig()
    
    # ============================================================
    # 🎯 【关键】任务类型选择
    # ============================================================
    config.task_type = 'power_to_stress'  # 'power_to_temperature' 或 'power_to_stress'
    # ============================================================
    
    # ============================================================
    # 📁 数据路径配置
    # ============================================================
    # 功率数据（输入，两个任务共用）
    config.power_dir = "/home/txr/power_a_input/"
    
    # 温度预测路径
    config.temp_dir = "/home/txr/power_a_output/"
    
    # 应力预测路径
    config.stress_dir = "/home/txr/power_a_output/"  # 根据实际路径修改
    
    # FLP文件
    config.flp_file_path = "/home/txr/code/final_version/1_data/data_generate/input_power_data_generate/power_data_flp/ev6_3D_core_layer.flp"
    
    # 缓存目录
    config.cache_dir = f"/home/txr/code/final_version/2_unet_architecture/cache/{config.task_type}"
    # ============================================================
    
    # GPU配置
    config.gpu_ids = "0,1,2,3"
    
    # 数据集划分
    config.dataset_config.train_id_range = (1, 1500)
    config.dataset_config.val_id_range = (1501, 2000)
    config.dataset_config.test_id_range = (2001, 3500)
    
    # 训练参数
    config.epochs = 300
    config.batch_size = 200 # 
    config.num_workers = 4
    config.learning_rate = 1e-3
    config.use_amp = False # 启用混合精度，节省显存
    config.use_cache = True
    config.precision = 'float64'
    
    # 学习率调度器
    config.use_scheduler = True
    config.plateau_patience = 10
    config.plateau_factor = 0.5
    config.plateau_min_lr = 1e-5
    
    # 保存目录
    config.save_dir = f"/home/txr/code/final_version/2_unet_architecture/training_logs/{config.task_type}_prediction/final_2"
    
    # ========== 开始训练 ==========
    try:
        model, stats = train_model(config)
        cp.header("🎉 训练成功完成!")
        
        task_name = "温度预测" if config.task_type == 'power_to_temperature' else "应力预测"
        unit = config.get_target_unit()
        
        print(f"\n{Fore.GREEN}任务类型: {task_name}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}最佳结果:{Style.RESET_ALL}")
        print(f"  最佳验证损失: {Fore.YELLOW}{stats['best_stats']['val_loss']:.6f}{Style.RESET_ALL} "
              f"(Epoch {stats['best_stats']['val_loss_epoch']})")
        
        if 'max_error' in stats['best_stats']:
            if config.task_type == 'power_to_temperature':
                print(f"  最佳最大误差: {Fore.YELLOW}{stats['best_stats']['max_error']:.3f}{unit}{Style.RESET_ALL} "
                      f"(Epoch {stats['best_stats']['max_error_epoch']})")
            else:
                print(f"  最佳最大误差: {Fore.YELLOW}{stats['best_stats']['max_error']:.3e}{unit}{Style.RESET_ALL} "
                      f"(Epoch {stats['best_stats']['max_error_epoch']})")
    
    except Exception as e:
        cp.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    