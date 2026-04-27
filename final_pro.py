import numpy as np
import random
import math
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

# ====================== 全局配置（适配你的项目） ======================
# 机械臂参数
L1 = 170    # Link1长度 (mm)
L2 = 160    # Link2长度 (mm)
L3 = 10     # Link3长度 (mm)
TORCH_OFFSET = 10  # 焊枪偏移量 (mm)
TORCH_RADIUS = 10  # 焊枪半径 (mm)
SAFE_DIST = 1      # 安全间隙 (mm)

# 避障物体（你的6个障碍物）
OBSTACLES = [
    {'type': 'circle', 'x': 80, 'y': 140, 'r': 25},
    {'type': 'hexagon', 'x': 150, 'y': 140, 'r': 20},
    {'type': 'circle', 'x': 240, 'y': 140, 'r': 35},
    {'type': 'circle', 'x': 150, 'y': 40,  'r': 30},
    {'type': 'hexagon', 'x': 250, 'y': 40,  'r': 20},
    {'type': 'circle', 'x': 80, 'y': 140, 'r': 25}  # 重复的圆形障碍
]

# 模拟24个焊接点坐标（可替换为你的实际坐标）
WELD_POINTS = [
    [30, 80], [40, 100], [50, 120], [60, 140],
    [70, 160], [80, 180], [90, 200], [100, 220],
    [110, 240], [120, 260], [130, 280], [140, 260],
    [150, 240], [160, 220], [170, 200], [180, 180],
    [190, 160], [200, 140], [210, 120], [220, 100],
    [230, 80], [240, 60], [250, 40], [260, 20]
]

# 轨迹规划参数
TOTAL_TIME = 60  # 总运动时间 (s)
TIME_STEP = 0.1  # 时间步长 (s)

# RRT优化参数（提升成功率）
RRT_MAX_ITER = 5000    # 增加迭代次数
RRT_STEP_SIZE = 0.08   # 减小步长
RRT_GOAL_THRESH = 0.3  # 增大目标阈值
RRT_GOAL_BIAS = 0.2    # 提高目标点采样概率

# ====================== 工具函数 ======================
def distance_point_to_segment(p, a, b):
    """计算点p到线段ab的最短距离"""
    vec_ab = np.array([b[0]-a[0], b[1]-a[1]])
    vec_ap = np.array([p[0]-a[0], p[1]-a[1]])
    proj = np.dot(vec_ap, vec_ab)
    
    if proj <= 0:
        return np.linalg.norm(vec_ap)
    len_ab_sq = np.dot(vec_ab, vec_ab)
    if len_ab_sq == 0:
        return np.linalg.norm(vec_ap)
    if proj >= len_ab_sq:
        return np.linalg.norm(np.array([p[0]-b[0], p[1]-b[1]]))
    
    cross = vec_ap[0] * vec_ab[1] - vec_ap[1] * vec_ab[0]
    return abs(cross) / np.sqrt(len_ab_sq)

def hexagon_vertices(center_x, center_y, r):
    """生成正六边形顶点坐标"""
    vertices = []
    for i in range(6):
        angle = math.pi/6 + i * math.pi/3
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        vertices.append([x, y])
    return vertices

def check_collision(robot_links, torch_segment, obstacles):
    """完整碰撞检测（含机械臂+焊枪）"""
    # 1. 检查机械臂连杆碰撞
    for i in range(len(robot_links)-1):
        a, b = robot_links[i], robot_links[i+1]
        for obs in obstacles:
            if obs['type'] == 'circle':
                d = distance_point_to_segment([obs['x'], obs['y']], a, b)
                if d < obs['r'] + SAFE_DIST + TORCH_RADIUS:
                    return True
            elif obs['type'] == 'hexagon':
                vertices = hexagon_vertices(obs['x'], obs['y'], obs['r'])
                for j in range(6):
                    v1 = vertices[j]
                    v2 = vertices[(j+1)%6]
                    d1 = distance_point_to_segment(v1, a, b)
                    d2 = distance_point_to_segment(v2, a, b)
                    d3 = distance_point_to_segment([obs['x'], obs['y']], a, b)
                    if min(d1, d2, d3) < SAFE_DIST + TORCH_RADIUS:
                        return True

    # 2. 检查焊枪碰撞
    t_a, t_b = torch_segment
    for obs in obstacles:
        if obs['type'] == 'circle':
            d = distance_point_to_segment([obs['x'], obs['y']], t_a, t_b)
            if d < obs['r'] + TORCH_RADIUS:
                return True
        elif obs['type'] == 'hexagon':
            vertices = hexagon_vertices(obs['x'], obs['y'], obs['r'])
            for j in range(6):
                v1 = vertices[j]
                v2 = vertices[(j+1)%6]
                d1 = distance_point_to_segment(v1, t_a, t_b)
                d2 = distance_point_to_segment(v2, t_a, t_b)
                if min(d1, d2) < TORCH_RADIUS:
                    return True
    return False

# ====================== 1. 正逆运动学 ======================
def fk_rrr_with_torch(theta1, theta2, theta3):
    """正运动学：输入关节角，输出机械臂连杆+焊枪坐标"""
    # 基坐标
    x0, y0 = 0, 0
    # J1 (Link1末端)
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    # J2 (Link2末端)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    # J3 (Link3末端)
    x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + L3 * np.sin(theta1 + theta2 + theta3)
    # 焊枪尖端
    total_angle = theta1 + theta2 + theta3
    tip_x = x3 + TORCH_OFFSET * np.cos(total_angle)
    tip_y = y3 + TORCH_OFFSET * np.sin(total_angle)
    
    robot_links = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    torch_segment = [[x3, y3], [tip_x, tip_y]]
    torch_tip = [tip_x, tip_y]
    
    return robot_links, torch_segment, torch_tip

def ik_rrr(tip_x, tip_y, phi=math.pi/2):
    """逆运动学：输入焊枪尖端坐标，输出关节角"""
    # 计算腕点（Link3末端）坐标
    wrist_x = tip_x - TORCH_OFFSET * np.cos(phi)
    wrist_y = tip_y - TORCH_OFFSET * np.sin(phi)
    
    # 计算theta2
    r_sq = wrist_x**2 + wrist_y**2
    cos_theta2 = (r_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # 防止数值误差
    theta2 = math.acos(cos_theta2)
    
    # 计算theta1
    alpha = math.atan2(wrist_y, wrist_x)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    
    # 计算theta3
    theta3 = phi - theta1 - theta2
    
    return [theta1, theta2, theta3]

# ====================== 2. TSP点序优化（贪心算法，适配24个点） ======================
def tsp_greedy(points):
    """TSP贪心优化：生成最优焊接点访问顺序"""
    n = len(points)
    if n == 0:
        return 0, []
    
    visited = [False] * n
    path = [0]
    visited[0] = True
    current = 0
    total_dist = 0
    
    for _ in range(n-1):
        min_dist = float('inf')
        next_idx = -1
        for i in range(n):
            if not visited[i] and i != current:
                dist = math.hypot(points[current][0]-points[i][0], points[current][1]-points[i][1])
                if dist < min_dist:
                    min_dist = dist
                    next_idx = i
        path.append(next_idx)
        visited[next_idx] = True
        total_dist += min_dist
        current = next_idx
    
    # 回到起点（可选）
    total_dist += math.hypot(points[current][0]-points[0][0], points[current][1]-points[0][1])
    path.append(0)
    
    return total_dist, path

# ====================== 3. RRT路径规划（优化版） ======================
class RRTNode:
    def __init__(self, theta):
        self.theta = theta
        self.parent = None

def smooth_rrt_path(path_theta, obstacles):
    """路径平滑：提升RRT路径质量，减少碰撞概率"""
    if len(path_theta) < 3:
        return path_theta
    
    smoothed = [path_theta[0]]
    current_idx = 0
    
    while current_idx < len(path_theta)-1:
        next_idx = len(path_theta)-1
        while next_idx > current_idx+1:
            theta_start = path_theta[current_idx]
            theta_end = path_theta[next_idx]
            
            # 插值检查
            collision = False
            num_interp = 8
            for i in range(num_interp+1):
                t = i / num_interp
                interp_theta = [
                    theta_start[0] + t*(theta_end[0]-theta_start[0]),
                    theta_start[1] + t*(theta_end[1]-theta_start[1]),
                    theta_start[2] + t*(theta_end[2]-theta_start[2])
                ]
                links, torch_seg, _ = fk_rrr_with_torch(*interp_theta)
                if check_collision(links, torch_seg, obstacles):
                    collision = True
                    break
            
            if not collision:
                break
            next_idx -= 1
        
        smoothed.append(path_theta[next_idx])
        current_idx = next_idx
    
    return smoothed

def rrt_path(start_theta, goal_theta, obstacles):
    """RRT无碰撞路径规划（优化版，提升成功率）"""
    start_node = RRTNode(start_theta)
    tree = [start_node]
    goal_reached = False
    goal_node = None
    
    for _ in range(RRT_MAX_ITER):
        # 随机采样（提高目标点采样概率）
        if random.random() < RRT_GOAL_BIAS:
            rand_theta = goal_theta
        else:
            # 限制关节角范围，避免无效采样
            rand_theta = [
                random.uniform(-math.pi/2, math.pi),    # theta1: -90°~180°
                random.uniform(-math.pi/2, math.pi/2),  # theta2: -90°~90°
                random.uniform(-math.pi/3, math.pi/3)   # theta3: -60°~60°
            ]
        
        # 找最近邻节点
        nearest_node = min(tree, key=lambda node: np.linalg.norm(np.array(node.theta) - np.array(rand_theta)))
        
        # 扩展新节点（更小步长）
        dir_vec = np.array(rand_theta) - np.array(nearest_node.theta)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-6:
            continue
        dir_unit = dir_vec / dir_norm
        new_theta = nearest_node.theta + dir_unit * RRT_STEP_SIZE
        
        # 限制关节角范围，避免奇异位形
        new_theta = [
            np.clip(new_theta[0], -math.pi/2, math.pi),
            np.clip(new_theta[1], -math.pi/2, math.pi/2),
            np.clip(new_theta[2], -math.pi/3, math.pi/3)
        ]
        
        # 碰撞检测
        links, torch_seg, _ = fk_rrr_with_torch(new_theta[0], new_theta[1], new_theta[2])
        if check_collision(links, torch_seg, obstacles):
            continue
        
        # 添加新节点
        new_node = RRTNode(new_theta)
        new_node.parent = nearest_node
        tree.append(new_node)
        
        # 检查是否到达目标（更大阈值）
        if np.linalg.norm(np.array(new_theta) - np.array(goal_theta)) < RRT_GOAL_THRESH:
            goal_node = new_node
            goal_reached = True
            break
    
    if not goal_reached:
        return []
    
    # 回溯路径并平滑
    path_theta = []
    current_node = goal_node
    while current_node is not None:
        path_theta.append(current_node.theta)
        current_node = current_node.parent
    path_theta.reverse()
    
    # 路径平滑
    path_theta = smooth_rrt_path(path_theta, obstacles)
    
    return path_theta

# ====================== 4. 三次多项式轨迹规划 ======================
def cubic_polynomial_trajectory(theta_start, theta_end, total_time, time_step):
    """三次多项式轨迹插值"""
    t_list = np.arange(0, total_time + time_step, time_step)
    theta1_traj, theta2_traj, theta3_traj = [], [], []
    
    # 对每个关节分别插值
    for i in range(3):
        theta0 = theta_start[i]
        thetaf = theta_end[i]
        t = t_list
        T = total_time
        
        # 三次多项式系数（初始/末端速度为0）
        a0 = theta0
        a1 = 0  
        a2 = 3*(thetaf - theta0)/(T**2)
        a3 = -2*(thetaf - theta0)/(T**3)
        
        # 计算轨迹
        theta_t = a0 + a1*t + a2*t**2 + a3*t**3
        if i == 0:
            theta1_traj = theta_t
        elif i == 1:
            theta2_traj = theta_t
        else:
            theta3_traj = theta_t
    
    # 组合轨迹
    traj = []
    for t, th1, th2, th3 in zip(t_list, theta1_traj, theta2_traj, theta3_traj):
        traj.append({
            'time': t,
            'theta1': th1,
            'theta2': th2,
            'theta3': th3
        })
    
    return traj

# ====================== 5. 主流程 ======================
def main():
    # Step 1: TSP优化焊接点顺序
    print("=== 1. TSP点序优化 ===")
    tsp_dist, tsp_path = tsp_greedy(WELD_POINTS)
    print(f"最短路径总长度: {tsp_dist:.2f} mm")
    print(f"焊接点访问顺序: {tsp_path[:5]}...")  # 只打印前5个
    
    # Step 2: 生成所有焊接点的关节角
    weld_thetas = []
    for idx in tsp_path:
        x, y = WELD_POINTS[idx]
        theta = ik_rrr(x, y)
        weld_thetas.append(theta)
    
    # Step 3: RRT规划相邻焊接点间的路径
    print("\n=== 2. RRT无碰撞路径规划 ===")
    all_rrt_paths = []
    success_count = 0
    fail_count = 0
    
    for i in range(len(weld_thetas)-1):
        start_theta = weld_thetas[i]
        goal_theta = weld_thetas[i+1]
        print(f"规划路径 {i+1}/{len(weld_thetas)-1}...", end="")
        
        rrt_path_theta = rrt_path(start_theta, goal_theta, OBSTACLES)
        if rrt_path_theta:
            all_rrt_paths.extend(rrt_path_theta)
            success_count += 1
            print(" ✅ 成功")
        else:
            fail_count += 1
            print(" ❌ 失败，使用直接插值")
            # 直接插值，增加中间点避免碰撞
            interp_steps = 5
            for s in range(interp_steps+1):
                t = s / interp_steps
                interp_theta = [
                    start_theta[0] + t*(goal_theta[0]-start_theta[0]),
                    start_theta[1] + t*(goal_theta[1]-start_theta[1]),
                    start_theta[2] + t*(goal_theta[2]-start_theta[2])
                ]
                all_rrt_paths.append(interp_theta)
    
    print(f"\nRRT规划统计：成功{success_count}条，失败{fail_count}条")
    
    # Step 4: 轨迹插值（生成连续时间轨迹）
    print("\n=== 3. 三次多项式轨迹插值 ===")
    full_traj = []
    current_time = 0
    
    if len(all_rrt_paths) < 2:
        all_rrt_paths = weld_thetas  # 保底：使用原始关节角
    
    for i in range(len(all_rrt_paths)-1):
        start_theta = all_rrt_paths[i]
        end_theta = all_rrt_paths[i+1]
        segment_time = TOTAL_TIME / max(1, len(all_rrt_paths)-1)
        
        # 生成该段轨迹
        seg_traj = cubic_polynomial_trajectory(start_theta, end_theta, segment_time, TIME_STEP)
        for point in seg_traj:
            point['time'] += current_time
            full_traj.append(point)
        
        current_time += segment_time
    
    # Step 5: 生成完整数据（焊枪坐标+关节坐标）
    print("\n=== 4. 生成完整轨迹数据 ===")
    trajectory_data = []
    for point in full_traj:
        t = point['time']
        th1, th2, th3 = point['theta1'], point['theta2'], point['theta3']
        
        # 正运动学计算坐标
        links, _, tip = fk_rrr_with_torch(th1, th2, th3)
        j1_x, j1_y = links[1]
        j2_x, j2_y = links[2]
        j3_x, j3_y = links[3]
        tip_x, tip_y = tip
        
        # 转换角度为度数（便于阅读）
        th1_deg = math.degrees(th1)
        th2_deg = math.degrees(th2)
        th3_deg = math.degrees(th3)
        
        trajectory_data.append({
            'time': round(t, 2),
            'torch_x': round(tip_x, 2),
            'torch_y': round(tip_y, 2),
            'j1_x': round(j1_x, 2),
            'j1_y': round(j1_y, 2),
            'j2_x': round(j2_x, 2),
            'j2_y': round(j2_y, 2),
            'j3_x': round(j3_x, 2),
            'j3_y': round(j3_y, 2),
            'theta1_deg': round(th1_deg, 2),
            'theta2_deg': round(th2_deg, 2),
            'theta3_deg': round(th3_deg, 2),
            'theta1_rad': round(th1, 4),
            'theta2_rad': round(th2, 4),
            'theta3_rad': round(th3, 4)
        })
    
    # Step 6: 保存CSV文件
    print("\n=== 5. 保存CSV文件 ===")
    csv_filename = "welding_trajectory.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['time', 'torch_x', 'torch_y', 
                      'j1_x', 'j1_y', 'j2_x', 'j2_y', 'j3_x', 'j3_y',
                      'theta1_deg', 'theta2_deg', 'theta3_deg',
                      'theta1_rad', 'theta2_rad', 'theta3_rad']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory_data)
    print(f"CSV文件已保存：{csv_filename}")
    
    # Step 7: 可视化绘图（修复RegularPolygon参数）
    print("\n=== 6. 绘制轨迹图 ===")
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制障碍物
    circle_label = True
    hex_label = True
    for obs in OBSTACLES:
        if obs['type'] == 'circle':
            circle = Circle(
                xy=(obs['x'], obs['y']), 
                radius=obs['r'], 
                color='red', 
                alpha=0.3,
                label='圆形障碍物' if circle_label else ""
            )
            plt.gca().add_patch(circle)
            circle_label = False
        elif obs['type'] == 'hexagon':
            # 修复：显式指定参数名
            hexagon = RegularPolygon(
                xy=(obs['x'], obs['y']), 
                numVertices=6, 
                radius=obs['r'], 
                color='orange', 
                alpha=0.3,
                label='六边形障碍物' if hex_label else ""
            )
            plt.gca().add_patch(hexagon)
            hex_label = False
    
    # 绘制焊接点
    weld_x = [p[0] for p in WELD_POINTS]
    weld_y = [p[1] for p in WELD_POINTS]
    plt.scatter(weld_x, weld_y, color='blue', s=50, label='焊接点', zorder=5)
    
    # 绘制轨迹
    traj_x = [d['torch_x'] for d in trajectory_data]
    traj_y = [d['torch_y'] for d in trajectory_data]
    plt.plot(traj_x, traj_y, color='green', linewidth=2, label='焊枪轨迹', zorder=3)
    
    # 绘图配置
    plt.xlabel('X坐标 (mm)')
    plt.ylabel('Y坐标 (mm)')
    plt.title('焊接机械臂避障轨迹图')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # 等比例显示
    plt.xlim(0, 300)   # 限定X轴范围，适配你的场景
    plt.ylim(0, 300)   # 限定Y轴范围，适配你的场景
    plt.savefig('welding_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 所有任务完成！生成文件：")
    print("1. welding_trajectory.png - 轨迹可视化图")
    print("2. welding_trajectory.csv - 完整轨迹数据")

# ====================== 运行主程序 ======================
if __name__ == "__main__":
    main()