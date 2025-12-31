"""
查看和分析.flp布局文件 
输出floorplan_layout图片
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FloorplanVisualizer:
    """布局文件可视化工具"""
    
    def __init__(self, flp_file: str):
        self.flp_file = flp_file
        self.load_floorplan()
    
    def load_floorplan(self):
        """加载布局文件"""
        print(f"Loading floorplan: {self.flp_file}")
        
        # 读取原始数据
        with open(self.flp_file, 'r') as f:
            lines = f.readlines()
        
        # 过滤注释和空行
        data_lines = [line.strip() for line in lines 
                     if line.strip() and not line.startswith('#')]
        
        if not data_lines:
            raise ValueError("Empty floorplan file")
        
        print(f"  Found {len(data_lines)} data lines")
        
        # 解析数据
        self.unit_names = []
        widths = []
        heights = []
        x_coords = []
        y_coords = []
        
        for i, line in enumerate(data_lines):
            parts = line.split()
            
            if len(parts) < 5:
                continue
            
            try:
                # 格式: name width height center_x center_y
                name = parts[0]
                w, h, cx, cy = map(float, parts[1:5])
                
                # 转换中心坐标为左下角坐标
                x = cx - w / 2
                y = cy - h / 2
                
                self.unit_names.append(name)
                widths.append(w)
                heights.append(h)
                x_coords.append(x)
                y_coords.append(y)
                
            except ValueError:
                continue
        
        # 转换为numpy数组
        self.widths = np.array(widths)
        self.heights = np.array(heights)
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)
        
        print(f"✓ Loaded {len(self.widths)} functional units")
        
        # 计算边界
        self.x_min = self.x_coords.min()
        self.x_max = (self.x_coords + self.widths).max()
        self.y_min = self.y_coords.min()
        self.y_max = (self.y_coords + self.heights).max()
        
        print(f"\nLayout dimensions:")
        print(f"  X: [{self.x_min:.2f}, {self.x_max:.2f}] mm")
        print(f"  Y: [{self.y_min:.2f}, {self.y_max:.2f}] mm")
        print(f"  Total area: {(self.x_max-self.x_min):.2f} × {(self.y_max-self.y_min):.2f} mm²")
    
    def visualize(self, save_path: str):
        """可视化布局并保存"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制每个单元
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.widths)))
        
        for i in range(len(self.widths)):
            x, y = self.x_coords[i], self.y_coords[i]
            w, h = self.widths[i], self.heights[i]
            
            # 绘制矩形
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5,
                edgecolor='black',
                facecolor=colors[i],
                alpha=0.6
            )
            ax.add_patch(rect)
            
            # 添加标签
            label = self.unit_names[i] if self.unit_names else f"{i}"
            ax.text(x + w/2, y + h/2, label,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlim(self.x_min - 2, self.x_max + 2)
        ax.set_ylim(self.y_min - 2, self.y_max + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title('Chip Floorplan Layout', fontsize=14, fontweight='bold')
        
        # 添加统计信息
        total_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        unit_area = (self.widths * self.heights).sum()
        
        info_text = f"Total units: {len(self.widths)}\n"
        info_text += f"Chip area: {total_area:.1f} mm²\n"
        info_text += f"Unit coverage: {unit_area/total_area*100:.1f}%"
        
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
        plt.close()


if __name__ == "__main__":
    # 输入路径
    flp_file = "/home/txr/code/second_version/final_version/1_data/data_generate/input_power_data_generate/ev6_3D_core_layer.flp"
    
    # 输出路径
    output_path = "/home/txr/code/second_version/final_version/1_data/data_generate/input_power_data_generate/floorplan_layout.png"
    
    # 创建可视化工具并生成图片
    viz = FloorplanVisualizer(flp_file)
    viz.visualize(save_path=output_path)
    
    print("\n✅ 完成！")