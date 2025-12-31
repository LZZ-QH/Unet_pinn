#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成功率生成器 - 一站式生成功率密度数据
合并SimplePowerGenerator和PowerDensityConverter
直接输出到多个目标文件夹，无中间文件

作者: 集成版
日期: 2024
"""

import random
import numpy as np
import os
from typing import List, Dict, Tuple


class IntegratedPowerGenerator:
    """
    集成功率生成器 - 直接生成功率密度网格数据
    
    特点:
    1. 合并功率生成和密度转换
    2. 无中间.ptrace文件
    3. 直接输出到多个目标文件夹
    4. 精简输出，只保留必要文件
    """
    
    def __init__(self, flp_path: str, coord_type: str = 'center'):
        """
        初始化集成生成器
        
        参数:
        -----
        flp_path : str
            FLP布局文件路径
        coord_type : str
            坐标类型，'center'（中心点）或'corner'（左下角）
        """
        # ========== 功率生成器配置 ==========
        self.n_units = 72
        
        # Light工况模式配置
        self.mode_config = {
            'power_range': (8, 15),      # 8-15W per unit
            'hotspot_prob': 0.1,         # 10%单元为热点
            'hotspot_multiplier': 1.3    # 热点功率×1.3
        }
        
        # ========== 功率密度转换器配置 ==========
        self.flp_path = flp_path
        self.coord_type = coord_type
        self.thickness_mm = 0.12  # 芯片厚度(mm)
        
        # 读取FLP布局
        self.unit_dict = self._read_flp()
        
        # 生成单元标签
        self.labels = [f'{chr(65 + i//9)}{i%9 + 1}' for i in range(self.n_units)]
        
        print("="*70)
        print("集成功率生成器初始化完成")
        print("="*70)
        print(f"FLP布局: {flp_path}")
        print(f"功能单元: {len(self.unit_dict)} 个")
        print(f"功率范围: {self.mode_config['power_range'][0]}-{self.mode_config['power_range'][1]} W/unit")
        print(f"热点比例: {self.mode_config['hotspot_prob']*100:.1f}%")
        
        # 显示芯片信息
        self._print_chip_info()
        print("="*70 + "\n")
    
    def _read_flp(self) -> Dict[str, Dict[str, float]]:
        """
        读取FLP布局文件
        
        返回:
        -----
        unit_dict : dict
            {label: {'width': w, 'height': h, 'x': x, 'y': y}}
            其中x,y为左下角坐标
        """
        unit_dict = {}
        
        with open(self.flp_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    label = parts[0]
                    width = float(parts[1])
                    height = float(parts[2])
                    x_raw = float(parts[3])
                    y_raw = float(parts[4])
                    
                    # 转换为左下角坐标
                    if self.coord_type == 'center':
                        x = x_raw - width / 2
                        y = y_raw - height / 2
                    else:
                        x = x_raw
                        y = y_raw
                    
                    unit_dict[label] = {
                        'width': width,
                        'height': height,
                        'x': x,
                        'y': y
                    }
        
        return unit_dict
    
    def _calculate_chip_bounds(self, padding: float = 0.0, 
                               padding_y: float = None) -> Tuple[float, float, float, float]:
        """计算芯片边界"""
        x_coords = []
        y_coords = []
        
        for label, info in self.unit_dict.items():
            x_coords.extend([info['x'], info['x'] + info['width']])
            y_coords.extend([info['y'], info['y'] + info['height']])
        
        if padding_y is None:
            padding_y = padding
        
        x_min = min(x_coords) - padding
        x_max = max(x_coords) + padding
        y_min = min(y_coords) - padding_y
        y_max = max(y_coords) + padding_y
        
        return x_min, x_max, y_min, y_max
    
    def _print_chip_info(self):
        """打印芯片基本信息"""
        x_min, x_max, y_min, y_max = self._calculate_chip_bounds()
        
        chip_width = x_max - x_min
        chip_height = y_max - y_min
        
        print(f"\n芯片信息:")
        print(f"  尺寸: {chip_width:.2f} × {chip_height:.2f} mm²")
        print(f"  厚度: {self.thickness_mm} mm")
    
    def _generate_power_pattern(self, add_correlation: bool = True,
                                correlation_level: float = 0.3) -> np.ndarray:
        """
        生成Light模式的功率分布（内存中）
        
        参数:
        -----
        add_correlation : bool
            是否添加空间相关性
        correlation_level : float
            相关性水平 (0-1)
        
        返回:
        -----
        powers : np.ndarray
            72个单元的功率值 (W)
        """
        powers = np.zeros(self.n_units)
        
        # 步骤1: 基础功率分配
        power_min, power_max = self.mode_config['power_range']
        for i in range(self.n_units):
            powers[i] = random.uniform(power_min, power_max)
        
        # 步骤2: 添加热点
        hotspot_prob = self.mode_config['hotspot_prob']
        if hotspot_prob > 0:
            n_hotspots = int(self.n_units * hotspot_prob)
            if n_hotspots > 0:
                hotspot_indices = random.sample(range(self.n_units), n_hotspots)
                multiplier = self.mode_config['hotspot_multiplier']
                for idx in hotspot_indices:
                    powers[idx] *= multiplier
        
        # 步骤3: 空间相关性（可选）
        if add_correlation and correlation_level > 0:
            powers = self._add_spatial_correlation(powers, correlation_level)
        
        # 步骤4: 添加噪声
        noise_std = power_max * 0.03
        noise = np.random.normal(0, noise_std, self.n_units)
        powers += noise
        
        # 步骤5: 限制范围
        powers = np.clip(powers, 1.0, 100.0)
        
        return powers
    
    def _add_spatial_correlation(self, powers: np.ndarray, 
                                 correlation: float) -> np.ndarray:
        """添加空间相关性"""
        try:
            from scipy.ndimage import gaussian_filter
            
            power_matrix = powers.reshape(8, 9)
            sigma = correlation * 1.5
            smoothed = gaussian_filter(power_matrix, sigma=sigma)
            power_matrix = correlation * smoothed + (1 - correlation) * power_matrix
            
            return power_matrix.flatten()
        
        except ImportError:
            return powers
    
    def _convert_to_power_density(self, powers: np.ndarray,
                                  grid_size: int = 200,
                                  padding: float = 0.0,
                                  padding_y: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将功率分布转换为功率密度网格
        
        参数:
        -----
        powers : np.ndarray
            72个单元的功率值
        grid_size : int
            网格分辨率
        padding : float
            X方向边界扩展(mm)
        padding_y : float
            Y方向边界扩展(mm)
        
        返回:
        -----
        X, Y, P : np.ndarray
            网格坐标和功率密度数组
        """
        # 计算芯片范围
        x_min, x_max, y_min, y_max = self._calculate_chip_bounds(padding, padding_y)
        
        # 创建网格
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 初始化功率密度数组
        P = np.zeros((grid_size, grid_size))
        
        # 为每个网格点分配功率密度
        for i in range(grid_size):
            for j in range(grid_size):
                point_x = X[i, j]
                point_y = Y[i, j]
                
                # 检查该点属于哪个功能单元
                for k, label in enumerate(self.labels):
                    if label in self.unit_dict:
                        unit = self.unit_dict[label]
                        
                        # 判断点是否在该单元内
                        if (point_x >= unit['x'] and 
                            point_x <= unit['x'] + unit['width'] and
                            point_y >= unit['y'] and 
                            point_y <= unit['y'] + unit['height']):
                            
                            # 计算功率密度 (W/m³)
                            volume = (unit['width'] * unit['height'] * 
                                    self.thickness_mm * 1e-9)  # m³
                            P[i, j] = powers[k] / volume
                            break
        
        return X, Y, P
    
    def _save_power_density(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray,
                           output_path: str):
        """
        保存功率密度数据为txt文件
        
        参数:
        -----
        X, Y, P : np.ndarray
            网格坐标和功率密度
        output_path : str
            输出文件路径
        """
        x_flat = X.flatten()
        y_flat = Y.flatten()
        p_flat = P.flatten()
        
        data = np.column_stack([x_flat, y_flat, p_flat])
        np.savetxt(output_path, data, delimiter=', ', fmt='%.6f')
    
    def generate_to_multiple_dirs(self,
                                  n_light: int,
                                  output_dirs: List[str],
                                  grid_size: int = 200,
                                  padding: float = 3.6,
                                  padding_y: float = 3.43,
                                  start_id: int = 1):
        """
        生成功率密度数据并输出到多个目录
        
        参数:
        -----
        n_light : int
            每个目录生成的样本数量
        output_dirs : List[str]
            输出目录列表（如power_a_input3到power_a_input7）
        grid_size : int
            网格分辨率
        padding : float
            X方向边界扩展(mm)
        padding_y : float
            Y方向边界扩展(mm)
        start_id : int
            起始编号
        """
        print("="*70)
        print("开始批量生成功率密度数据")
        print("="*70)
        print(f"样本数量: {n_light} 个/目录")
        print(f"输出目录数: {len(output_dirs)} 个")
        print(f"网格分辨率: {grid_size}×{grid_size}")
        print(f"边界扩展: X={padding}mm, Y={padding_y}mm")
        print(f"起始编号: {start_id}")
        print("="*70)
        
        # 为每个输出目录生成数据
        for dir_idx, output_dir in enumerate(output_dirs, 1):
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            dir_name = os.path.basename(output_dir)
            print(f"\n[目录 {dir_idx}/{len(output_dirs)}] {dir_name}")
            print("-"*70)
            
            # 生成n_light个样本
            for i in range(n_light):
                sample_id = start_id + i
                
                # 步骤1: 生成功率分布（内存中）
                powers = self._generate_power_pattern()
                
                # 步骤2: 转换为功率密度网格（内存中）
                X, Y, P = self._convert_to_power_density(
                    powers, grid_size, padding, padding_y
                )
                
                # 步骤3: 保存到文件
                filename = f"power_a_{sample_id}.txt"
                filepath = os.path.join(output_dir, filename)
                self._save_power_density(X, Y, P, filepath)
                
                # 进度显示（每20个或最后一个）
                if (i + 1) % 20 == 0 or (i + 1) == n_light:
                    total_power = powers.sum()
                    max_power = powers.max()
                    progress = (i + 1) / n_light * 100
                    print(f"  进度: {i+1}/{n_light} ({progress:.1f}%) | "
                          f"样本 {sample_id} | "
                          f"总功率: {total_power:.1f}W | "
                          f"最大: {max_power:.1f}W")
            
            print(f"  ✓ 完成: {output_dir}")
        
        # 总结
        total_files = len(output_dirs) * n_light
        print("\n" + "="*70)
        print(f"✓ 全部生成完成！")
        print(f"  生成文件总数: {total_files}")
        print(f"  每个目录: {n_light} 个文件")
        print(f"  文件编号范围: {start_id} ~ {start_id + n_light - 1}")
        print("="*70)
        
        # 验证输出
        self._verify_outputs(output_dirs, n_light, start_id)
    
    def _verify_outputs(self, output_dirs: List[str], n_light: int, start_id: int):
        """验证输出文件"""
        print("\n验证输出文件:")
        print("-"*70)
        
        all_ok = True
        for output_dir in output_dirs:
            dir_name = os.path.basename(output_dir)
            
            if not os.path.exists(output_dir):
                print(f"  ⚠️  {dir_name}: 目录不存在")
                all_ok = False
                continue
            
            # 检查文件数量
            files = [f for f in os.listdir(output_dir) if f.startswith('power_a_') and f.endswith('.txt')]
            
            if len(files) == n_light:
                print(f"  ✓ {dir_name}: {len(files)} 个文件")
            else:
                print(f"  ⚠️  {dir_name}: {len(files)} 个文件（预期 {n_light}）")
                all_ok = False
        
        if all_ok:
            print("\n✓ 所有目录验证通过！")
        else:
            print("\n⚠️  部分目录存在问题，请检查")
        
        print("-"*70)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("集成功率生成器 - 批量生成到多个目录")
    print("="*70)
    
    # ========== 配置参数 ==========
    # FLP布局文件
    flp_path = "/home/txr/code/second_version/power_data_generate/power_ptarce_generate_data/ev6_3D_core_layer.flp"
    
    # 输出目录列表（power_a_input3 到 power_a_input7）
    output_dirs = [
        "/home/txr/power_a_input3",
        "/home/txr/power_a_input4",
        "/home/txr/power_a_input5",
        "/home/txr/power_a_input6",
        "/home/txr/power_a_input7"
    ]
    
    # 生成参数
    n_light = 200    # 每个目录生成400个样本
    grid_size = 200      # 网格分辨率
    padding = 3.6        # X方向边界扩展(mm)
    padding_y = 3.43     # Y方向边界扩展(mm)
    start_id = 1         # 起始编号
    
    # ========== 执行生成 ==========
    # 创建集成生成器
    generator = IntegratedPowerGenerator(
        flp_path=flp_path,
        coord_type='center'
    )
    
    # 生成并输出到5个目录
    generator.generate_to_multiple_dirs(
        n_light=n_light,
        output_dirs=output_dirs,
        grid_size=grid_size,
        padding=padding,
        padding_y=padding_y,
        start_id=start_id
    )
    
    print("\n" + "="*70)
    print("完成！所有功率密度数据已生成")
    print("="*70)
    print("\n输出目录:")
    for i, dir_path in enumerate(output_dirs, 1):
        print(f"  {i}. {dir_path}")
    print("\n每个目录包含:")
    print(f"  - power_a_1.txt ~ power_a_{n_light}.txt ({n_light}个文件)")
    print("\n下一步:")
    print("  → 使用COMSOL导入这些功率密度数据进行热仿真")
    print("="*70)

    