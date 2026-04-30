"""
焊接轨迹数据分析完整代码
功能：读取焊接轨迹CSV数据，进行2D/3D可视化分析，输出关键统计信息
作者：数据分析师
日期：2026年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. 核心修复：解决中文方块问题
# ======================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'

# ======================
# 1. 基础设置与数据读取
# ======================
def setup_environment():
    """设置绘图环境"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def load_welding_data(file_path):
    """
    读取焊接轨迹数据
    参数: file_path - CSV文件路径
    返回: df - 数据框
    """
    try:
        df = pd.read_csv(file_path)
        print(f"成功读取数据，形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
        
        # 检查缺失值
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"发现缺失值: {missing[missing > 0]}")
        else:
            print("无缺失值")
            
        return df
    except Exception as e:
        print(f"数据读取失败: {str(e)}")
        raise

# ======================
# 2. 数据预处理与计算
# ======================
def preprocess_data(df):
    """
    数据预处理：计算速度、轨迹长度等
    参数: df - 原始数据框
    返回: df - 处理后的数据框, segment_stats - 段统计信息
    """
    # 计算末端执行器速度
    df['ee_speed'] = np.sqrt(df['ee_x'].diff()**2 + df['ee_y'].diff()**2)
    df['ee_speed_smoothed'] = df['ee_speed'].rolling(window=20, center=True).mean()
    
    # 计算各段轨迹长度
    segment_stats = []
    for seg in sorted(df['segment_idx'].unique()):
        seg_data = df[df['segment_idx'] == seg].copy()
        if len(seg_data) > 1:
            # 计算段内轨迹长度
            x_diff = seg_data['ee_x'].diff().dropna()
            y_diff = seg_data['ee_y'].diff().dropna()
            seg_length = np.sqrt(x_diff**2 + y_diff**2).sum()
            
            segment_stats.append({
                'segment_idx': seg,
                'point_count': len(seg_data),
                'length_mm': seg_length,
                'avg_speed': seg_data['ee_speed'].mean(),
                'max_speed': seg_data['ee_speed'].max()
            })
    
    segment_df = pd.DataFrame(segment_stats)
    return df, segment_df

# ======================
# 3. 2D可视化分析
# ======================
def plot_2d_analysis(df, save_path='welding_trajectory_2d.png'):
    """
    生成2D轨迹分析图
    参数: df - 处理后的数据框, save_path - 保存路径
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('焊接轨迹数据2D分析', fontsize=16, fontweight='bold', y=0.95)
    
    # 颜色配置
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    segment_colors = plt.cm.Set3(np.linspace(0, 1, df['segment_idx'].nunique()))
    
    # 图1: 末端执行器位置时间序列
    ax1.plot(df['sample_idx'], df['ee_x'], color=colors[0], linewidth=1.5, label='EE-X坐标')
    ax1.plot(df['sample_idx'], df['ee_y'], color=colors[1], linewidth=1.5, label='EE-Y坐标')
    ax1.set_title('末端执行器位置随时间变化', fontweight='bold')
    ax1.set_xlabel('采样索引')
    ax1.set_ylabel('位置 (mm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 末端执行器运动轨迹
    for segment in sorted(df['segment_idx'].unique()):
        seg_data = df[df['segment_idx'] == segment]
        ax2.plot(seg_data['ee_x'], seg_data['ee_y'], 
                 color=segment_colors[segment], linewidth=2, 
                 label=f'段{segment}')
    
    # 标记基座位置
    base_x, base_y = df['base_x'].iloc[0], df['base_y'].iloc[0]
    ax2.scatter(base_x, base_y, color='red', s=100, marker='s', 
               label='基座位置', zorder=5)
    
    ax2.set_title('末端执行器运动轨迹', fontweight='bold')
    ax2.set_xlabel('X坐标 (mm)')
    ax2.set_ylabel('Y坐标 (mm)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 图3: 关节1、2角度变化
    ax3.plot(df['sample_idx'], df['theta1'], color=colors[0], linewidth=1.5, label='Theta1')
    ax3.plot(df['sample_idx'], df['theta2'], color=colors[1], linewidth=1.5, label='Theta2')
    ax3.plot(df['sample_idx'], df['theta3'], color=colors[2], linewidth=1.5, label='Theta2')
    ax3.set_title('θ1、2、3角度随时间变化', fontweight='bold')
    ax3.set_xlabel('采样索引')
    ax3.set_ylabel('角度 (弧度)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 末端执行器速度
    ax4.plot(df['sample_idx'], df['ee_speed'], color=colors[0], alpha=0.3, label='瞬时速度')
    ax4.plot(df['sample_idx'], df['ee_speed_smoothed'], color=colors[3], linewidth=2, label='平滑速度')
    ax4.set_title('末端执行器运动速度', fontweight='bold')
    ax4.set_xlabel('采样索引')
    ax4.set_ylabel('速度 (mm/步)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D分析图已保存至: {save_path}")

# ======================
# 4. 3D可视化分析
# ======================
def plot_3d_comprehensive_analysis(df, segment_df, save_path='welding_trajectory_3d.png'):
    """
    生成3D综合分析图
    参数: df - 处理后的数据框, segment_df - 段统计信息, save_path - 保存路径
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 颜色配置
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    segment_colors = plt.cm.Set3(np.linspace(0, 1, df['segment_idx'].nunique()))
    
    # 图1: 3D轨迹可视化
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # 绘制末端执行器轨迹
    ax1.plot(df['ee_x'], df['ee_y'], range(len(df)), 
             color=colors[0], linewidth=3, label='末端执行器轨迹', alpha=0.8)
    
    # 绘制关节轨迹（辅助线）
    joints = [('j1', colors[1]), ('j2', colors[2]), ('j3', colors[3])]
    for joint_name, color in joints:
        x_col = f'{joint_name}_x'
        y_col = f'{joint_name}_y'
        ax1.plot(df[x_col], df[y_col], range(len(df)), 
                 color=color, linewidth=1.5, alpha=0.5, label=f'{joint_name}关节轨迹')
    
    # 标记基座
    base_x, base_y = df['base_x'].iloc[0], df['base_y'].iloc[0]
    ax1.scatter(base_x, base_y, 0, color='red', s=200, marker='s', 
               label='基座原点', zorder=10)
    
    ax1.set_title('焊接机器人3D运动轨迹', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X坐标 (mm)')
    ax1.set_ylabel('Y坐标 (mm)')
    ax1.set_zlabel('时间步长')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 各段轨迹长度分布
    ax2 = fig.add_subplot(2, 3, 2)
    bars = ax2.bar(segment_df['segment_idx'], segment_df['length_mm'], 
                   color=segment_colors, alpha=0.8)
    ax2.set_title('各段轨迹长度分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('轨迹段编号')
    ax2.set_ylabel('轨迹长度 (mm)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: 关节角度相关性
    ax3 = fig.add_subplot(2, 3, 3)
    scatter = ax3.scatter(df['theta1'], df['theta2'], c=range(len(df)), 
                         cmap='viridis', alpha=0.6, s=5)
    ax3.set_title('关节1与关节2角度相关性', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Theta1 (弧度)')
    ax3.set_ylabel('Theta2 (弧度)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='时间步长')
    
    # 图4: 速度分析
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df['sample_idx'], df['ee_speed'], color=colors[0], alpha=0.3, label='瞬时速度')
    ax4.plot(df['sample_idx'], df['ee_speed_smoothed'], color=colors[3], linewidth=2, label='平滑速度')
    ax4.set_title('末端执行器运动速度', fontsize=14, fontweight='bold')
    ax4.set_xlabel('采样索引')
    ax4.set_ylabel('速度 (mm/步)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 图5: 关节角度分布
    ax5 = fig.add_subplot(2, 3, 5)
    angle_data = [df['theta1'], df['theta2'], df['theta3']]
    angle_labels = ['Theta1', 'Theta2', 'Theta3']
    
    for data, label, color in zip(angle_data, angle_labels, colors[:3]):
        ax5.hist(data, bins=30, alpha=0.6, label=label, color=color, density=True)
    
    ax5.set_title('关节角度分布', fontsize=14, fontweight='bold')
    ax5.set_xlabel('角度 (弧度)')
    ax5.set_ylabel('概率密度')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 图6: 各段采样点数量
    ax6 = fig.add_subplot(2, 3, 6)
    point_counts = segment_df['point_count'].values
    segment_ids = segment_df['segment_idx'].values
    ax6.plot(segment_ids, point_counts, marker='o', linewidth=2, 
             markersize=6, color=colors[3])
    ax6.set_title('各段轨迹采样点数量', fontsize=14, fontweight='bold')
    ax6.set_xlabel('轨迹段编号')
    ax6.set_ylabel('采样点数量')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('焊接机器人轨迹全面分析报告', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D综合分析图已保存至: {save_path}")

# ======================
# 5. 统计信息输出
# ======================
def print_statistical_summary(df, segment_df):
    """
    输出关键统计信息
    参数: df - 处理后的数据框, segment_df - 段统计信息
    """
    print("\n" + "="*60)
    print("焊接轨迹数据分析报告 - 关键统计信息")
    print("="*60)
    
    # 基本信息
    print(f"基本信息:")
    print(f"   • 总采样点数: {len(df):,}")
    print(f"   • 轨迹段数量: {df['segment_idx'].nunique()}")
    print(f"   • 基座位置: X={df['base_x'].iloc[0]:.1f} mm, Y={df['base_y'].iloc[0]:.1f} mm")
    
    # 运动范围
    print(f"\n运动范围:")
    for axis in ['ee_x', 'ee_y']:
        min_val = df[axis].min()
        max_val = df[axis].max()
        range_val = max_val - min_val
        print(f"   • {axis.upper()}: {min_val:.2f} ~ {max_val:.2f} mm (范围: {range_val:.2f} mm)")
    
    # 轨迹长度统计
    print(f"\n轨迹长度统计:")
    total_length = segment_df['length_mm'].sum()
    print(f"   • 轨迹总长度: {total_length:.2f} mm")
    print(f"   • 平均段长度: {segment_df['length_mm'].mean():.2f} mm (±{segment_df['length_mm'].std():.2f})")
    
    max_seg = segment_df.loc[segment_df['length_mm'].idxmax()]
    min_seg = segment_df.loc[segment_df['idxmin']]
    print(f"   • 最长段: 段{int(max_seg['segment_idx'])} ({max_seg['length_mm']:.2f} mm)")
    print(f"   • 最短段: 段{int(min_seg['segment_idx'])} ({min_seg['length_mm']:.2f} mm)")
    
    # 速度统计
    print(f"\n速度统计:")
    valid_speed = df['ee_speed'].dropna()
    print(f"   • 平均速度: {valid_speed.mean():.4f} mm/步")
    print(f"   • 最大速度: {valid_speed.max():.4f} mm/步")
    print(f"   • 95%速度分位数: {np.percentile(valid_speed, 95):.4f} mm/步")
    
    # 关节角度统计
    print(f"\n关节角度范围 (弧度):")
    for angle_col in ['theta1', 'theta2', 'theta3']:
        min_angle = df[angle_col].min()
        max_angle = df[angle_col].max()
        print(f"   • {angle_col}: {min_angle:.4f} ~ {max_angle:.4f}")
    
    print("="*60)

# ======================
# 6. 主函数
# ======================
def main(file_path):
    """
    主函数：执行完整的焊接轨迹分析流程
    参数: file_path - CSV文件路径
    """
    try:
        # 1. 初始化环境
        setup_environment()
        print("初始化分析环境完成")
        
        # 2. 读取数据
        df = load_welding_data(file_path)
        
        # 3. 数据预处理
        df_processed, segment_df = preprocess_data(df)
        print("数据预处理完成")
        
        # 4. 生成可视化图表
        plot_2d_analysis(df_processed)
        plot_3d_comprehensive_analysis(df_processed, segment_df)
        
        # 5. 输出统计信息
        print_statistical_summary(df_processed, segment_df)
        
        print("\n焊接轨迹数据分析完成！")
        return df_processed, segment_df
        
    except Exception as e:
        print(f"分析过程出错: {str(e)}")
        return None, None

# ======================
# 执行分析
# ======================
if __name__ == "__main__":
    # 这里直接写你的文件绝对路径
    WELDING_DATA_PATH = r"welding_trajectory_detailed.csv"
    
    # 执行分析
    result_df, result_segment_df = main(WELDING_DATA_PATH)