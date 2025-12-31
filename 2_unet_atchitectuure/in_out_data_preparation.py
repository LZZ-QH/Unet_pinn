#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
统一数据准备模块 - 功率→温度 / 功率→应力
=============================================================================
缓存命名:
  - 功率→温度: power_temp_XXXX.pkl
  - 功率→应力: power_stress_XXXX.pkl

边界说明:
  - 功率→温度: FLP边界 X[-36.20, 36.20], Y[-11.50, 11.50]
  - 功率→应力: 
    - 功率: FLP边界
    - 应力: 全局应力数据范围（动态计算）
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F  # ⭐ V4风格：使用torch插值
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from typing import List, Tuple, Optional, Dict
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 常量配置
# =============================================================================

# 默认padding (与功率生成脚本一致)
DEFAULT_PADDING_X = 3.6
DEFAULT_PADDING_Y = 3.43

# 功率插值参数
POWER_MID_RES = 200
POWER_TARGET_SIZE = 256
POWER_MASK_PADDING_MM = 0.1
POWER_ZOOM_ORDER = 1

# FLP边界 (功能单元范围)
FLP_X_MIN = -36.20
FLP_X_MAX = 36.20
FLP_Y_MIN = -11.50
FLP_Y_MAX = 11.50

# 芯片边界 (FLP + padding)
CHIP_X_MIN = -40.0
CHIP_X_MAX = 40.0
CHIP_Y_MIN = -15.0
CHIP_Y_MAX = 15.0

# =============================================================================
# 数据读取函数
# =============================================================================

def read_txt_power(path: str, skip_rows: int = 0) -> np.ndarray:
    """读取功率文件 (x,y,power)"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(skip_rows):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if not data:
        raise ValueError(f"未能从 {path} 解析到有效数据")
    return np.array(data, dtype=np.float64)


def read_txt_temperature(path: str, skip_rows: int = 8) -> np.ndarray:
    """读取温度文件，Z方向平均"""
    import pandas as pd
    df = pd.read_csv(path, skiprows=skip_rows, header=None,
                     names=['x', 'y', 'z', 'temperature'],
                     usecols=['x', 'y', 'temperature'],
                     comment='%', on_bad_lines='skip', dtype=np.float64)
    temp_2d = df.groupby(['x', 'y'], as_index=False)['temperature'].mean()
    return temp_2d[['x', 'y', 'temperature']].values.astype(np.float64)


def read_txt_stress(path: str, skip_rows: int = 1, z_min: Optional[float] = None) -> np.ndarray:
    """
    读取应力文件，Z方向平均
    
    参数:
        path: 应力文件路径
        skip_rows: 跳过的行数
        z_min: Z值下限，None表示不过滤（V4风格）
    
    返回:
        (N, 3) 数组: [x, y, von_mises]
    """
    import pandas as pd
    df = pd.read_csv(path, skiprows=skip_rows, header=None,
                     names=['x', 'y', 'z', 'von_mises', 'u', 'v', 'w'],
                     usecols=['x', 'y', 'z', 'von_mises'],
                     comment='%', on_bad_lines='skip', dtype=np.float64)
    
    if df.empty:
        raise ValueError(f"未能从 {path} 解析到有效数据")
    
    # 可选的Z过滤
    if z_min is not None:
        df = df[df['z'] >= z_min].copy()
    
    stress_2d = df.groupby(['x', 'y'], as_index=False)['von_mises'].mean()
    return stress_2d[['x', 'y', 'von_mises']].values.astype(np.float64)


# =============================================================================
# FLP解析
# =============================================================================

def parse_flp_data(flp_file_path: str, padding_x: float = DEFAULT_PADDING_X,
                   padding_y: float = DEFAULT_PADDING_Y) -> Tuple[List[Dict], Dict]:
    """解析FLP布局文件"""
    try:
        with open(flp_file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        units, all_x_min, all_x_max, all_y_min, all_y_max = [], [], [], [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    w, h, cx, cy = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x_min, x_max = cx - w/2, cx + w/2
                    y_min, y_max = cy - h/2, cy + h/2
                    units.append({'name': parts[0], 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
                    all_x_min.append(x_min); all_x_max.append(x_max)
                    all_y_min.append(y_min); all_y_max.append(y_max)
                except: continue
        
        if units:
            flp_x_min, flp_x_max = min(all_x_min), max(all_x_max)
            flp_y_min, flp_y_max = min(all_y_min), max(all_y_max)
            bounds = {
                'flp_x_min': flp_x_min, 'flp_x_max': flp_x_max,
                'flp_y_min': flp_y_min, 'flp_y_max': flp_y_max,
                'chip_x_min': flp_x_min - padding_x, 'chip_x_max': flp_x_max + padding_x,
                'chip_y_min': flp_y_min - padding_y, 'chip_y_max': flp_y_max + padding_y,
            }
            print(f"✓ 成功解析FLP文件: {len(units)}个单元")
            return units, bounds
    except Exception as e:
        print(f"❌ FLP解析失败: {e}")
    return [], {}


def create_physical_mask(Xg, Yg, flp_units, padding_mm=0.1):
    """创建物理掩码"""
    if not flp_units:
        return np.ones_like(Xg, dtype=bool)
    mask = np.zeros_like(Xg, dtype=bool)
    for u in flp_units:
        mask |= ((Xg >= u['x_min']-padding_mm) & (Xg <= u['x_max']+padding_mm) &
                 (Yg >= u['y_min']-padding_mm) & (Yg <= u['y_max']+padding_mm))
    return mask


# =============================================================================
# 温度预测插值器（保持原有实现）
# =============================================================================

class TempInterpolator:
    """
    温度预测插值器（保持原有实现）
    
    边界配置:
      - 功率: FLP边界 → 200×200 → 256×256 (scipy.ndimage.zoom)
      - 温度: FLP边界 → 256×256
    """
    
    def __init__(self, flp_units: List[Dict], method: str = 'linear',
                 precision: str = 'float64', padding_x: float = DEFAULT_PADDING_X,
                 padding_y: float = DEFAULT_PADDING_Y):
        self.flp_units = flp_units
        self.method = method
        self.np_dtype = np.float64 if precision == 'float64' else np.float32
        
        if flp_units:
            self.flp_x_min = min(u['x_min'] for u in flp_units)
            self.flp_x_max = max(u['x_max'] for u in flp_units)
            self.flp_y_min = min(u['y_min'] for u in flp_units)
            self.flp_y_max = max(u['y_max'] for u in flp_units)
        else:
            self.flp_x_min, self.flp_x_max = FLP_X_MIN, FLP_X_MAX
            self.flp_y_min, self.flp_y_max = FLP_Y_MIN, FLP_Y_MAX
    
    def _interp(self, x, y, vals, Xg, Yg):
        """基础插值"""
        pts = np.column_stack((x, y))
        grid = griddata(pts, vals, (Xg, Yg), method=self.method)
        if np.isnan(grid).any():
            grid_nn = griddata(pts, vals, (Xg, Yg), method='nearest')
            grid[np.isnan(grid)] = grid_nn[np.isnan(grid)]
        return grid.astype(self.np_dtype)
    
    def interpolate_power(self, x, y, power, target_size=256):
        """功率插值：FLP边界→200×200→256×256 (scipy.ndimage.zoom)"""
        # 阶段1: 插值到200×200
        xi = np.linspace(self.flp_x_min, self.flp_x_max, POWER_MID_RES)
        yi = np.linspace(self.flp_y_min, self.flp_y_max, POWER_MID_RES)
        Xg, Yg = np.meshgrid(xi, yi)
        grid = self._interp(x, y, power, Xg, Yg)
        
        # 阶段2: 应用掩码
        if self.flp_units:
            mask = create_physical_mask(Xg, Yg, self.flp_units, POWER_MASK_PADDING_MM)
            grid[~mask] = 0.0
        grid = np.nan_to_num(grid, nan=0.0)
        
        # 阶段3: 缩放到256×256 (scipy.ndimage.zoom)
        grid = zoom(grid, target_size / POWER_MID_RES, order=POWER_ZOOM_ORDER)
        return grid.astype(self.np_dtype)
    
    def interpolate_temperature(self, x, y, temp, target_size=256):
        """温度插值：FLP边界→256×256"""
        xi = np.linspace(self.flp_x_min, self.flp_x_max, target_size)
        yi = np.linspace(self.flp_y_min, self.flp_y_max, target_size)
        Xg, Yg = np.meshgrid(xi, yi)
        grid = self._interp(x, y, temp, Xg, Yg)
        return np.nan_to_num(grid, nan=0.0).astype(self.np_dtype)
    
    def get_flp_bounds(self):
        return {'x_min': self.flp_x_min, 'x_max': self.flp_x_max,
                'y_min': self.flp_y_min, 'y_max': self.flp_y_max}


# =============================================================================
# 应力预测插值器（V4风格）
# =============================================================================

class StressInterpolatorV4:
    """
    应力预测插值器（V4风格）
    
    ⭐ V4风格特点:
      - 功率: FLP边界 → 200×200 → 256×256 (torch.nn.functional.interpolate)
      - 应力: 全局应力数据范围（动态计算）→ 256×256
    """
    
    def __init__(self, flp_units: List[Dict], method: str = 'linear',
                 precision: str = 'float64'):
        self.flp_units = flp_units
        self.method = method
        self.has_flp = len(flp_units) > 0
        self.precision = precision
        self.np_dtype = np.float64 if precision == 'float64' else np.float32
        
        if self.has_flp:
            self.chip_x_min = min(u['x_min'] for u in flp_units)
            self.chip_x_max = max(u['x_max'] for u in flp_units)
            self.chip_y_min = min(u['y_min'] for u in flp_units)
            self.chip_y_max = max(u['y_max'] for u in flp_units)
        else:
            self.chip_x_min = self.chip_x_max = None
            self.chip_y_min = self.chip_y_max = None
        
        # ⭐ V4风格：应力边界初始化为None，由数据集动态计算
        self.stress_xmin = None
        self.stress_xmax = None
        self.stress_ymin = None
        self.stress_ymax = None
        self.Xg_global = None
        self.Yg_global = None
    
    def _interp(self, x, y, vals, Xg, Yg):
        """基础插值"""
        pts = np.column_stack((x, y))
        grid = griddata(pts, vals, (Xg, Yg), method=self.method)
        if np.isnan(grid).any():
            grid_nn = griddata(pts, vals, (Xg, Yg), method='nearest')
            grid[np.isnan(grid)] = grid_nn[np.isnan(grid)]
        return grid.astype(self.np_dtype)
    
    def interpolate_power(self, x, y, power, target_size=256):
        """
        功率插值：FLP边界→200×200→256×256
        
        ⭐ V4风格：使用 torch.nn.functional.interpolate
        """
        if not self.has_flp:
            # 无FLP时使用数据范围
            x_range = [x.min(), x.max()]
            y_range = [y.min(), y.max()]
            xi = np.linspace(x_range[0], x_range[1], target_size)
            yi = np.linspace(y_range[0], y_range[1], target_size)
            Xg, Yg = np.meshgrid(xi, yi)
            grid = self._interp(x, y, power, Xg, Yg)
            return grid.astype(self.np_dtype)
        
        # === 阶段1: 高分辨率插值到200×200 ===
        mid_res = POWER_MID_RES
        xi_mid = np.linspace(self.chip_x_min, self.chip_x_max, mid_res)
        yi_mid = np.linspace(self.chip_y_min, self.chip_y_max, mid_res)
        Xg_mid, Yg_mid = np.meshgrid(xi_mid, yi_mid)
        grid_mid = self._interp(x, y, power, Xg_mid, Yg_mid)
        
        # === 阶段2: 应用FLP掩码 ===
        mask_mid = create_physical_mask(Xg_mid, Yg_mid, self.flp_units, POWER_MASK_PADDING_MM)
        grid_mid[~mask_mid] = 0.0
        grid_mid = np.nan_to_num(grid_mid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # === 阶段3: 调整到目标分辨率 ===
        # ⭐ V4风格：使用 torch.nn.functional.interpolate
        if target_size != mid_res:
            original_dtype = grid_mid.dtype
            grid_mid_tensor = torch.from_numpy(grid_mid).unsqueeze(0).unsqueeze(0)
            grid_target_tensor = F.interpolate(
                grid_mid_tensor, 
                size=(target_size, target_size),
                mode='bilinear', 
                align_corners=False
            )
            # 恢复原始精度
            grid_target = grid_target_tensor.squeeze().numpy().astype(original_dtype)
        else:
            grid_target = grid_mid
        
        return grid_target.astype(self.np_dtype)
    
    def interpolate_stress(self, x, y, stress, Xg, Yg):
        """
        应力插值：到指定的全局网格
        
        ⭐ V4风格：使用外部传入的全局网格（动态计算的应力数据范围）
        """
        grid = self._interp(x, y, stress, Xg, Yg)
        return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0).astype(self.np_dtype)
    
    def set_stress_bounds(self, xmin, xmax, ymin, ymax, grid_res=(256, 256)):
        """
        设置应力边界（由数据集动态计算后调用）
        
        ⭐ V4风格：应力边界来自全局应力数据范围
        """
        self.stress_xmin = xmin
        self.stress_xmax = xmax
        self.stress_ymin = ymin
        self.stress_ymax = ymax
        
        H, W = grid_res
        xi = np.linspace(xmin, xmax, W, dtype=self.np_dtype)
        yi = np.linspace(ymin, ymax, H, dtype=self.np_dtype)
        self.Xg_global, self.Yg_global = np.meshgrid(xi, yi)
    
    def get_power_bounds(self):
        return {'x_min': self.chip_x_min, 'x_max': self.chip_x_max,
                'y_min': self.chip_y_min, 'y_max': self.chip_y_max}
    
    def get_stress_bounds(self):
        return {'x_min': self.stress_xmin, 'x_max': self.stress_xmax,
                'y_min': self.stress_ymin, 'y_max': self.stress_ymax}


# =============================================================================
# 数据集类
# =============================================================================

class PowerTempDataset(torch.utils.data.Dataset):
    """功率→温度数据集 (缓存: power_temp_XXXX.pkl)"""
    
    def __init__(self, power_files, temp_files, file_ids, flp_file_path=None,
                 grid_res=(256,256), cache_dir=None, precision='float64',
                 padding_x=DEFAULT_PADDING_X, padding_y=DEFAULT_PADDING_Y):
        assert len(power_files) == len(temp_files) == len(file_ids)
        self.power_files, self.temp_files, self.file_ids = power_files, temp_files, file_ids
        self.grid_res = grid_res
        self.np_dtype = np.float64 if precision == 'float64' else np.float32
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        flp_units, _ = parse_flp_data(flp_file_path, padding_x, padding_y) if flp_file_path else ([], {})
        self.interpolator = TempInterpolator(flp_units, precision=precision,
                                              padding_x=padding_x, padding_y=padding_y)
        self.cache = {}
        self.bounds = self.interpolator.get_flp_bounds()
    
    def _cache_path(self, idx):
        return self.cache_dir / f"power_temp_{self.file_ids[idx]:04d}.pkl" if self.cache_dir else None
    
    def __len__(self): return len(self.power_files)
    
    def __getitem__(self, idx):
        if idx in self.cache: return self.cache[idx]
        
        cp = self._cache_path(idx)
        if cp and cp.exists():
            with open(cp, 'rb') as f: 
                data = pickle.load(f)
                self.cache[idx] = data
                return data
        
        try:
            p_data = read_txt_power(self.power_files[idx])
            t_data = read_txt_temperature(self.temp_files[idx])
            p_grid = self.interpolator.interpolate_power(p_data[:,0], p_data[:,1], p_data[:,2], self.grid_res[0])
            t_grid = self.interpolator.interpolate_temperature(t_data[:,0], t_data[:,1], t_data[:,2], self.grid_res[0])
            result = (p_grid, t_grid)
            self.cache[idx] = result
            if cp:
                with open(cp, 'wb') as f: pickle.dump(result, f)
            return result
        except Exception as e:
            print(f"⚠️ 样本{idx}(ID={self.file_ids[idx]})失败: {e}")
            return (np.zeros(self.grid_res, self.np_dtype), np.zeros(self.grid_res, self.np_dtype))
    
    def get_bounds(self): return self.bounds


class PowerStressDataset(torch.utils.data.Dataset):
    """
    功率→应力数据集 (缓存: power_stress_XXXX.pkl)
    
    ⭐ V4风格:
      - 功率插值使用 torch.nn.functional.interpolate
      - 应力边界使用全局应力数据范围（动态计算）
    """
    
    def __init__(self, power_files, stress_files, file_ids, flp_file_path=None,
                 grid_res=(256,256), cache_dir=None, precision='float64',
                 interp_method='linear', stress_z_min=None):
        """
        参数:
            stress_z_min: 应力Z值下限，None表示不过滤（V4风格）
        """
        assert len(power_files) == len(stress_files) == len(file_ids)
        self.power_files, self.stress_files, self.file_ids = power_files, stress_files, file_ids
        self.grid_res = grid_res
        self.precision = precision
        self.np_dtype = np.float64 if precision == 'float64' else np.float32
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.stress_z_min = stress_z_min  # ⭐ None表示不过滤（V4风格）
        self.interp_method = interp_method
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        flp_units, _ = parse_flp_data(flp_file_path) if flp_file_path else ([], {})
        self.flp_units = flp_units
        self.interpolator = StressInterpolatorV4(flp_units, method=interp_method, precision=precision)
        self.cache = {}
        
        # ⭐ V4风格：动态分析全局空间域
        self._analyze_spatial_domain()
        
        self.power_bounds = self.interpolator.get_power_bounds()
        self.stress_bounds = self.interpolator.get_stress_bounds()
    
    def _analyze_spatial_domain(self):
        """
        分析全局空间域
        
        ⭐ V4风格：从应力数据中动态计算边界
        """
        print("\n分析数据空间域...")
        
        sample_size = min(10, len(self.power_files))
        sample_indices = np.linspace(0, len(self.power_files)-1, sample_size, dtype=int)
        
        # 分析功率数据范围
        all_x, all_y = [], []
        for idx in sample_indices:
            try:
                data = read_txt_power(self.power_files[idx])
                all_x.append(data[:, 0])
                all_y.append(data[:, 1])
            except:
                continue
        
        if all_x:
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            print(f"  功率数据范围: X[{all_x.min():.2f}, {all_x.max():.2f}], "
                  f"Y[{all_y.min():.2f}, {all_y.max():.2f}]")
        
        # ⭐ V4风格：分析应力数据范围
        all_x, all_y = [], []
        for idx in sample_indices:
            try:
                data = read_txt_stress(self.stress_files[idx], z_min=self.stress_z_min)
                all_x.append(data[:, 0])
                all_y.append(data[:, 1])
            except:
                continue
        
        if all_x:
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            
            stress_xmin = all_x.min()
            stress_xmax = all_x.max()
            stress_ymin = all_y.min()
            stress_ymax = all_y.max()
            
            print(f"  应力数据范围: X[{stress_xmin:.2f}, {stress_xmax:.2f}], "
                  f"Y[{stress_ymin:.2f}, {stress_ymax:.2f}]")
            
            # ⭐ V4风格：设置应力边界
            self.interpolator.set_stress_bounds(
                stress_xmin, stress_xmax, stress_ymin, stress_ymax, self.grid_res
            )
            print(f"  统一网格: {self.grid_res[0]}x{self.grid_res[1]} (精度: {self.precision})")
        else:
            # 回退到默认边界
            self.interpolator.set_stress_bounds(-40, 40, -15, 15, self.grid_res)
            print(f"  ⚠️ 无法分析应力数据范围，使用默认边界")
    
    def _cache_path(self, idx):
        return self.cache_dir / f"power_stress_{self.file_ids[idx]:04d}.pkl" if self.cache_dir else None
    
    def __len__(self): return len(self.power_files)
    
    def __getitem__(self, idx):
        if idx in self.cache: return self.cache[idx]
        
        cp = self._cache_path(idx)
        if cp and cp.exists():
            with open(cp, 'rb') as f:
                data = pickle.load(f)
                self.cache[idx] = data
                return data
        
        try:
            p_data = read_txt_power(self.power_files[idx])
            s_data = read_txt_stress(self.stress_files[idx], z_min=self.stress_z_min)
            
            # 功率插值
            p_grid = self.interpolator.interpolate_power(
                p_data[:,0], p_data[:,1], p_data[:,2], self.grid_res[0]
            )
            
            # ⭐ V4风格：应力插值到全局网格
            s_grid = self.interpolator.interpolate_stress(
                s_data[:,0], s_data[:,1], s_data[:,2],
                self.interpolator.Xg_global, self.interpolator.Yg_global
            )
            
            result = (p_grid, s_grid)
            self.cache[idx] = result
            if cp:
                with open(cp, 'wb') as f: pickle.dump(result, f)
            return result
        except Exception as e:
            print(f"⚠️ 样本{idx}(ID={self.file_ids[idx]})失败: {e}")
            return (np.zeros(self.grid_res, self.np_dtype), np.zeros(self.grid_res, self.np_dtype))
    
    def get_power_bounds(self): return self.power_bounds
    def get_stress_bounds(self): return self.stress_bounds
    
    def get_spatial_domain(self):
        """获取空间域信息"""
        if self.interpolator.Xg_global is not None:
            return self.interpolator.Xg_global, self.interpolator.Yg_global
        else:
            H, W = self.grid_res
            xi = np.linspace(-40, 40, W, dtype=self.np_dtype)
            yi = np.linspace(-15, 15, H, dtype=self.np_dtype)
            return np.meshgrid(xi, yi)


# =============================================================================
# DataLoader创建函数
# =============================================================================

def create_power_temp_dataloaders(power_files, temp_files, file_ids, train_idx, val_idx,
                                   test_idx=None, flp_file_path=None, grid_res=(256,256),
                                   batch_size=64, cache_dir=None, precision='float64',
                                   num_workers=0, padding_x=DEFAULT_PADDING_X, padding_y=DEFAULT_PADDING_Y):
    """创建功率→温度DataLoader（保持原有实现）"""
    print(f"\n{'='*60}\n创建 功率→温度 数据集\n{'='*60}")
    ds = PowerTempDataset(power_files, temp_files, file_ids, flp_file_path,
                          grid_res, cache_dir, precision, padding_x, padding_y)
    b = ds.get_bounds()
    print(f"✓ 样本:{len(ds)} | 训练:{len(train_idx)} | 验证:{len(val_idx)}" +
          (f" | 测试:{len(test_idx)}" if test_idx else ""))
    print(f"  缓存: power_temp_XXXX.pkl")
    print(f"  边界: X[{b['x_min']:.2f},{b['x_max']:.2f}] Y[{b['y_min']:.2f},{b['y_max']:.2f}] (FLP)")
    print(f"  功率缩放: scipy.ndimage.zoom")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, train_idx),
                                                batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, val_idx),
                                              batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, test_idx),
                                               batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_idx else None
    return train_loader, val_loader, test_loader, ds


def create_power_stress_dataloaders(power_files, stress_files, file_ids, train_idx, val_idx,
                                     test_idx=None, flp_file_path=None, grid_res=(256,256),
                                     batch_size=64, cache_dir=None, precision='float64',
                                     num_workers=0, interp_method='linear', stress_z_min=None):
    """
    创建功率→应力DataLoader（V4风格）
    
    ⭐ V4风格:
      - 功率插值使用 torch.nn.functional.interpolate
      - 应力边界使用全局应力数据范围（动态计算）
      - stress_z_min=None 表示不过滤Z值
    """
    print(f"\n{'='*60}\n创建 功率→应力 数据集 (V4风格)\n{'='*60}")
    ds = PowerStressDataset(power_files, stress_files, file_ids, flp_file_path,
                            grid_res, cache_dir, precision, interp_method, stress_z_min)
    pb, sb = ds.get_power_bounds(), ds.get_stress_bounds()
    print(f"✓ 样本:{len(ds)} | 训练:{len(train_idx)} | 验证:{len(val_idx)}" +
          (f" | 测试:{len(test_idx)}" if test_idx else ""))
    print(f"  缓存: power_stress_XXXX.pkl")
    print(f"  功率边界: X[{pb['x_min']:.2f},{pb['x_max']:.2f}] Y[{pb['y_min']:.2f},{pb['y_max']:.2f}] (FLP)")
    print(f"  应力边界: X[{sb['x_min']:.2f},{sb['x_max']:.2f}] Y[{sb['y_min']:.2f},{sb['y_max']:.2f}] ⭐ 全局动态")
    print(f"  功率缩放: torch.nn.functional.interpolate (V4风格)")
    if stress_z_min is not None:
        print(f"  应力Z过滤: Z >= {stress_z_min} mm")
    else:
        print(f"  应力Z过滤: 无 (V4风格)")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, train_idx),
                                                batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, val_idx),
                                              batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, test_idx),
                                               batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_idx else None
    return train_loader, val_loader, test_loader, ds


# =============================================================================
# 兼容旧版本的接口（用于应力预测训练脚本）
# =============================================================================

def create_dynamic_dataloaders(power_files: List[str],
                              stress_files: List[str],
                              file_ids: List[int],
                              train_indices: List[int],
                              val_indices: List[int],
                              test_indices: List[int] = None,
                              flp_file_path: Optional[str] = None,
                              grid_res: Tuple[int, int] = (256, 256),
                              batch_size: int = 64,
                              interp_method: str = 'linear',
                              use_cache: bool = True,
                              use_gpu_interp: bool = False,  # 保留参数但不使用
                              num_workers: int = 0,
                              pin_memory: bool = True,
                              cache_dir: Optional[str] = None,
                              precision: str = 'float64',
                              stress_z_min: Optional[float] = None) -> Tuple:
    """
    创建动态插值的DataLoader（V4风格，兼容旧版本接口）
    
    ⭐ V4风格:
      - 功率插值使用 torch.nn.functional.interpolate
      - 应力边界使用全局应力数据范围（动态计算）
    """
    return create_power_stress_dataloaders(
        power_files=power_files,
        stress_files=stress_files,
        file_ids=file_ids,
        train_idx=train_indices,
        val_idx=val_indices,
        test_idx=test_indices,
        flp_file_path=flp_file_path,
        grid_res=grid_res,
        batch_size=batch_size,
        cache_dir=cache_dir,
        precision=precision,
        num_workers=num_workers,
        interp_method=interp_method,
        stress_z_min=stress_z_min
    )


# =============================================================================
# 归一化工具
# =============================================================================

def normalize(data, method='zscore'):
    """归一化: zscore/minmax/robust"""
    if method == 'zscore':
        mean, std = float(data.mean()), float(data.std())
        std = std if std > 1e-8 else 1.0
        return (data - mean) / std, {'method': 'zscore', 'mean': mean, 'std': std}
    elif method == 'minmax':
        vmin, vmax = float(data.min()), float(data.max())
        rng = vmax - vmin if vmax - vmin > 1e-8 else 1.0
        return (data - vmin) / rng, {'method': 'minmax', 'min': vmin, 'max': vmax}
    elif method == 'robust':
        nz = data[data > 1e-6]
        if len(nz) > 0:
            med, iqr = float(np.median(nz)), float(np.percentile(nz,75) - np.percentile(nz,25))
            iqr = iqr if iqr > 1e-8 else 1.0
            out = np.zeros_like(data)
            out[data > 1e-6] = (data[data > 1e-6] - med) / iqr
            return out, {'method': 'robust', 'median': med, 'iqr': iqr}
        return data.copy(), {'method': 'robust', 'median': 0.0, 'iqr': 1.0}
    raise ValueError(f"未知方法: {method}")

def denormalize(data, stats):
    """反归一化"""
    m = stats.get('method', 'zscore')
    if m == 'zscore': return data * stats['std'] + stats['mean']
    if m == 'minmax': return data * (stats['max'] - stats['min']) + stats['min']
    if m == 'robust': return data * stats['iqr'] + stats['median']
    return data


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("统一数据准备模块 ")
    print("="*60)
    print(f"""
┌────────────────────────────────────────────────────────────┐
│  功率→温度（保持原有实现）                                 │
│  ─────────────────────────────────────────────────────── │
│  功率插值: FLP边界 → 200×200 → 256×256                    │
│            使用 scipy.ndimage.zoom                        │
│  温度插值: FLP边界 → 256×256                              │
├────────────────────────────────────────────────────────────┤
│  功率→应力                             │
│  ─────────────────────────────────────────────────────── │
│  功率插值: FLP边界 → 200×200 → 256×256                    │
│            使用 torch.nn.functional.interpolate        │
│  应力插值: 全局应力数据范围（动态计算）→ 256×256     │
├────────────────────────────────────────────────────────────┤
│  缓存命名:                                                 │
│    功率→温度: power_temp_XXXX.pkl                         │
│    功率→应力: power_stress_XXXX.pkl                       │
└────────────────────────────────────────────────────────────┘
""")