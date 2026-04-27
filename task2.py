import numpy as np
import random
import math

# ====================== 基础工具函数（碰撞检测核心） ======================
def distance_point_to_segment(p, a, b):
    """
    计算点p到线段ab的最短距离（碰撞检测核心公式）
    :param p: 点坐标 [x,y]
    :param a: 线段起点 [x,y]
    :param b: 线段终点 [x,y]
    :return: 最短距离
    """
    # 向量AB和AP
    vec_ab = np.array([b[0]-a[0], b[1]-a[1]])
    vec_ap = np.array([p[0]-a[0], p[1]-a[1]])
    
    # 投影长度（点积）
    proj = np.dot(vec_ap, vec_ab)
    if proj <= 0:
        return np.linalg.norm(vec_ap)  # 投影在A外侧，距离=AP
    
    len_ab_sq = np.dot(vec_ab, vec_ab)
    if len_ab_sq <= 0:
        return np.linalg.norm(vec_ap)  # 线段退化为点
    
    if proj >= len_ab_sq:
        vec_bp = np.array([p[0]-b[0], p[1]-b[1]])
        return np.linalg.norm(vec_bp)  # 投影在B外侧，距离=BP
    
    # 投影在线段上，用叉乘算垂直距离
    cross = vec_ap[0]*vec_ab[1] - vec_ap[1]*vec_ab[0]
    return abs(cross) / np.sqrt(len_ab_sq)

def check_collision(robot_links, obstacles, safe_dist=11):
    """
    检查机械臂是否碰撞（焊枪半径10mm + 1mm间隙 = 11mm安全距离）
    :param robot_links: 机械臂连杆端点列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    :param obstacles: 障碍物列表 [[x1,y1,r1], [x2,y2,r2], ...]（圆形障碍，x/y为中心，r为半径）
    :param safe_dist: 安全距离（默认11mm）
    :return: True=碰撞，False=无碰撞
    """
    # 检查每个连杆与每个障碍物的距离
    for i in range(len(robot_links)-1):
        link_start = robot_links[i]
        link_end = robot_links[i+1]
        for obs in obstacles:
            obs_x, obs_y, obs_r = obs
            # 障碍物中心到连杆的距离
            dist = distance_point_to_segment([obs_x, obs_y], link_start, link_end)
            # 距离 < 障碍物半径 + 安全距离 → 碰撞
            if dist < obs_r + safe_dist:
                return True
    return False

def fk_rrr(theta1, theta2, theta3, L1=170, L2=160, L3=10):
    """
    RRR平面机械臂正运动学（第2章核心），返回各连杆端点坐标
    :param theta1/theta2/theta3: 关节角度（弧度）
    :param L1/L2/L3: 连杆长度（mm，适配你们的机械臂参数）
    :return: 连杆端点列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    """
    # 基坐标系原点
    x0, y0 = 0, 0
    # Link1末端（关节2）
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    # Link2末端（关节3）
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    # Link3末端（焊枪尖端）
    x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + L3 * np.sin(theta1 + theta2 + theta3)
    
    return [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]

# ====================== RRT核心算法（第6章路径规划） ======================
class RRTNode:
    """RRT节点类：存储关节角度和父节点"""
    def __init__(self, theta):
        self.theta = theta  # 关节角度 [theta1, theta2, theta3]（弧度）
        self.parent = None  # 父节点引用

def rrt_path(start_theta, goal_theta, obstacles, L1=170, L2=160, L3=10, 
             max_iter=2000, step_size=0.1, goal_threshold=0.2):
    """
    RRT路径规划（关节空间）：从起点关节角到终点关节角，生成无碰撞路径
    :param start_theta: 起点关节角 [theta1, theta2, theta3]（弧度）
    :param goal_theta: 终点关节角 [theta1, theta2, theta3]（弧度）
    :param obstacles: 障碍物列表 [[x,y,r], ...]
    :param L1/L2/L3: 连杆长度
    :param max_iter: 最大迭代次数（防止死循环）
    :param step_size: 每次扩展的步长（弧度）
    :param goal_threshold: 到达终点的阈值（弧度）
    :return: 
        - path_theta: 路径关节角列表 [[th1,th2,th3], ...]
        - path_cartesian: 路径笛卡尔坐标列表 [[x3,y3], ...]（焊枪尖端）
    """
    # 初始化RRT树
    start_node = RRTNode(start_theta)
    tree = [start_node]
    goal_reached = False

    # 迭代扩展树
    for _ in range(max_iter):
        # 步骤1：随机采样一个节点（偏向目标点，提高收敛速度）
        if random.random() < 0.1:  # 10%概率直接采样目标点
            rand_theta = goal_theta
        else:
            # 在关节角范围内随机采样（-π ~ π）
            rand_theta = [
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi)
            ]
        
        # 步骤2：找到树中离随机节点最近的节点
        def calc_theta_dist(theta1, theta2):
            """计算关节角空间的距离（欧氏距离）"""
            return np.linalg.norm(np.array(theta1) - np.array(theta2))
        
        nearest_node = min(tree, key=lambda node: calc_theta_dist(node.theta, rand_theta))
        
        # 步骤3：从最近节点向随机节点扩展一小步
        dir_vec = np.array(rand_theta) - np.array(nearest_node.theta)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm == 0:
            continue  # 方向为0，跳过
        dir_unit = dir_vec / dir_norm
        new_theta = nearest_node.theta + dir_unit * step_size
        
        # 步骤4：检查新节点是否碰撞
        # 正运动学得到机械臂连杆坐标
        robot_links = fk_rrr(new_theta[0], new_theta[1], new_theta[2], L1, L2, L3)
        if check_collision(robot_links, obstacles):
            continue  # 碰撞，跳过
        
        # 步骤5：添加新节点到树中
        new_node = RRTNode(new_theta)
        new_node.parent = nearest_node
        tree.append(new_node)
        
        # 步骤6：检查是否到达目标点
        if calc_theta_dist(new_theta, goal_theta) < goal_threshold:
            goal_node = new_node
            goal_reached = True
            break
    
    if not goal_reached:
        print("RRT未找到路径！")
        return [], []
    
    # 步骤7：回溯路径（从目标点到起点）
    path_theta = []
    current_node = goal_node
    while current_node is not None:
        path_theta.append(current_node.theta)
        current_node = current_node.parent
    path_theta.reverse()  # 反转路径（起点→终点）
    
    # 步骤8：路径平滑（移除冗余节点，可选但建议加）
    path_theta = smooth_path(path_theta, obstacles, L1, L2, L3)
    
    # 步骤9：转换为笛卡尔坐标（焊枪尖端）
    path_cartesian = []
    for theta in path_theta:
        links = fk_rrr(theta[0], theta[1], theta[2], L1, L2, L3)
        path_cartesian.append(links[-1])  # 取焊枪尖端坐标
    
    return path_theta, path_cartesian

def smooth_path(path_theta, obstacles, L1, L2, L3):
    """
    路径平滑：尝试跳过中间节点，减少路径拐点（提升运动顺滑性）
    """
    if len(path_theta) < 3:
        return path_theta
    
    smoothed = [path_theta[0]]
    current_idx = 0
    
    while current_idx < len(path_theta)-1:
        next_idx = len(path_theta)-1
        # 尝试从current_idx直接到next_idx
        while next_idx > current_idx+1:
            # 检查两点之间的路径是否无碰撞
            theta_start = path_theta[current_idx]
            theta_end = path_theta[next_idx]
            # 插值生成中间点
            num_interp = 10
            interp_thetas = []
            for i in range(num_interp+1):
                t = i / num_interp
                interp_theta = [
                    theta_start[0] + t*(theta_end[0]-theta_start[0]),
                    theta_start[1] + t*(theta_end[1]-theta_start[1]),
                    theta_start[2] + t*(theta_end[2]-theta_start[2])
                ]
                interp_thetas.append(interp_theta)
            # 检查所有中间点是否碰撞
            collision = False
            for theta in interp_thetas:
                links = fk_rrr(theta[0], theta[1], theta[2], L1, L2, L3)
                if check_collision(links, obstacles):
                    collision = True
                    break
            if not collision:
                break  # 无碰撞，直接跳
            next_idx -= 1
        smoothed.append(path_theta[next_idx])
        current_idx = next_idx
    
    return smoothed

# ====================== 调用示例（适配你们的焊接项目） ======================
if __name__ == "__main__":
    # 1. 定义障碍物（示例：2个圆形障碍，替换成你们的实际障碍）
    # 格式：[[x,y,r], ...]，x/y为障碍中心，r为半径（mm）
    obstacles = [
        [100, 100, 30],  # 障碍1：中心(100,100)，半径30mm
        [200, 200, 25]   # 障碍2：中心(200,200)，半径25mm
    ]
    
    # 2. 定义起点和终点关节角（示例，替换成逆解得到的实际角度）
    # 起点：焊枪在第一个焊接点
    start_theta = [np.pi/4, np.pi/6, np.pi/3]  # 45°, 30°, 60°（弧度）
    # 终点：焊枪在第二个焊接点
    goal_theta = [np.pi/3, np.pi/4, np.pi/6]   # 60°, 45°, 30°（弧度）
    
    # 3. 运行RRT路径规划
    path_theta, path_cartesian = rrt_path(
        start_theta=start_theta,
        goal_theta=goal_theta,
        obstacles=obstacles,
        L1=170, L2=160, L3=10,
        max_iter=2000,
        step_size=0.1,
        goal_threshold=0.2
    )
    
    # 4. 输出结果
    if path_theta:
        print("✅ RRT找到无碰撞路径！")
        print(f"路径节点数：{len(path_theta)}")
        print("\n关节角路径（前5个节点）：")
        for i, theta in enumerate(path_theta[:5]):
            theta_deg = [np.degrees(t) for t in theta]
            print(f"节点{i}：θ1={theta_deg[0]:.1f}°, θ2={theta_deg[1]:.1f}°, θ3={theta_deg[2]:.1f}°")
        
        print("\n焊枪尖端笛卡尔路径（前5个节点）：")
        for i, (x, y) in enumerate(path_cartesian[:5]):
            print(f"节点{i}：x={x:.1f}mm, y={y:.1f}mm")
    else:
        print("❌ RRT未找到路径，请调整参数或障碍位置！")