import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class WeldingProject:
    def __init__(self):
        self.link_lengths = [170, 160, 10]
        self.link_radii = [7.0, 7.0, 5.0]
        self.real_obstacles = [
            {'type': 'circle',  'x': 80,  'y': 140, 'r': 25},
            {'type': 'hexagon', 'x': 150, 'y': 140, 'r': 20},
            {'type': 'circle',  'x': 240, 'y': 140, 'r': 35},
            {'type': 'circle',  'x': 150, 'y': 40,  'r': 30},
            {'type': 'hexagon', 'x': 250, 'y': 40,  'r': 20},
        ]
        self.base_pos = np.array([15.0, 15.0]) # 机器人底座位置
        self.obstacles = self.real_obstacles
        self.welding_points = self.load_welding_points()
        self.clearance = 1.0  # Task 4: 碰撞检测间隙 1mm

    def load_welding_points(self):
        """根据尺寸图生成24个焊接点坐标，焊接点距物体边界6mm"""
        objects = [
            {'type': 'circle',  'x': 80,  'y': 140, 'r': 25},
            {'type': 'hexagon', 'x': 150, 'y': 140, 'r': 20},
            {'type': 'circle',  'x': 240, 'y': 140, 'r': 35},
            {'type': 'circle',  'x': 150, 'y': 40,  'r': 30},
            {'type': 'hexagon', 'x': 250, 'y': 40,  'r': 20}
        ]
        offset = 6.0
        # 起点姿态可以设定为指向左侧 (np.pi) 避免初始姿态别扭
        self.start_point = np.array([295.0, 15.0, np.pi])  
        points = []
        for obj in objects:
            if obj['type'] == 'circle':
                r_w = obj['r'] + offset
                for a in [0, 90, 180, 270]:
                    rad = np.deg2rad(a)
                    # 【核心修复】：姿态角应该朝向圆心，即向内。向量方向与向外的辐射方向相反。
                    phi = rad + np.pi 
                    # 规范化到 [-pi, pi] 之间（可选，但对运动学计算更友好）
                    phi = np.arctan2(np.sin(phi), np.cos(phi))
                    points.append([obj['x'] + r_w*np.cos(rad), obj['y'] + r_w*np.sin(rad), phi])
                    
            elif obj['type'] == 'hexagon':
                # 平顶六边形，焊接点在各边中点法线方向
                r_inner = obj['r'] * np.cos(np.deg2rad(30))
                r_w = r_inner + offset
                for a in [30, 90, 150, 210, 270, 330]:
                    rad = np.deg2rad(a)
                    # 【核心修复】：姿态角垂直于边朝向中心
                    phi = rad + np.pi
                    phi = np.arctan2(np.sin(phi), np.cos(phi))
                    points.append([obj['x'] + r_w*np.cos(rad), obj['y'] + r_w*np.sin(rad), phi])
                    
        return np.array(points)

    # ==================== 运动学 ====================
    def inverse_kinematics(self, x, y, phi):
        l1, l2, l3 = self.link_lengths
        # 减去底座偏移量
        dx = x - self.base_pos[0]
        dy = y - self.base_pos[1]
        
        x2 = dx - l3 * np.cos(phi)
        y2 = dy - l3 * np.sin(phi)
        dist_sq = x2**2 + y2**2
        cos_t2 = (dist_sq - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(cos_t2) > 1.0: return None
        theta2 = np.arccos(cos_t2)
        theta1 = np.arctan2(y2, x2) - np.arctan2(l2*np.sin(theta2), l1 + l2*np.cos(theta2))
        theta3 = phi - theta1 - theta2
        return np.array([theta1, theta2, theta3])

    def forward_kinematics(self, joints):
        l1, l2, l3 = self.link_lengths
        t1, t2, t3 = joints
        p1 = np.array([l1*np.cos(t1), l1*np.sin(t1)])
        p2 = p1 + np.array([l2*np.cos(t1+t2), l2*np.sin(t1+t2)])
        p3 = p2 + np.array([l3*np.cos(t1+t2+t3), l3*np.sin(t1+t2+t3)])
        # 加上底座偏移量
        return [self.base_pos, p1 + self.base_pos, p2 + self.base_pos, p3 + self.base_pos]

    # ==================== 碰撞检测辅助 ====================
    def point_to_segment_dist(self, p, a, b):
        a, b, p = np.asarray(a), np.asarray(b), np.asarray(p)
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0.0: return np.linalg.norm(p - a)
        t = max(0.0, min(1.0, np.dot(p - a, ab) / ab_len_sq))
        return np.linalg.norm(p - (a + t * ab))

    def segments_intersect(self, a, b, c, d):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

    def get_hex_verts(self, obj):
        """获取平顶六边形的6个顶点"""
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        return [np.array([obj['x'] + obj['r']*np.cos(a), obj['y'] + obj['r']*np.sin(a)]) for a in angles]

    # ==================== 笛卡尔空间碰撞检测 ====================
    def point_in_obstacle(self, x, y, margin):
        """检查点(x,y)是否在障碍物内部或距离<margin"""
        for obj in self.real_obstacles:
            cx, cy = obj['x'], obj['y']
            if obj['type'] == 'circle':
                if np.hypot(x - cx, y - cy) < obj['r'] + margin:
                    return True
            elif obj['type'] == 'hexagon':
                verts = self.get_hex_verts(obj)
                p = np.array([x, y])
                # 检查到各边的距离
                for i in range(6):
                    if self.point_to_segment_dist(p, verts[i], verts[(i+1)%6]) < margin:
                        return True
                # 检查是否在六边形内部（用内切圆快速判断）
                r_inner = obj['r'] * np.cos(np.deg2rad(30))
                if np.hypot(x - cx, y - cy) < r_inner:
                    return True
        return False

    def line_hits_obstacle(self, p1, p2, margin, n_checks=30):
        """检查线段p1->p2是否穿过障碍物"""
        for t in np.linspace(0, 1, n_checks):
            px = p1[0] + t*(p2[0]-p1[0])
            py = p1[1] + t*(p2[1]-p1[1])
            if self.point_in_obstacle(px, py, margin):
                return True
        return False

    # ==================== 连杆碰撞检测 ====================
    def check_collision(self, joints, obstacles):
        """检查机器人连杆是否与障碍物碰撞"""
        points = self.forward_kinematics(joints)
        segments = [
            (points[0], points[1], self.link_radii[0]),
            (points[1], points[2], self.link_radii[1]),
            (points[2], points[3], self.link_radii[2])
        ]
        for p_start, p_end, l_radius in segments:
            dist_limit = l_radius + self.clearance
            for obj in obstacles:
                obj_center = np.array([obj['x'], obj['y']])
                if obj['type'] == 'circle':
                    dist = self.point_to_segment_dist(obj_center, p_start, p_end)
                    if dist < obj['r'] + dist_limit:
                        return True
                elif obj['type'] == 'hexagon':
                    dist_to_center = self.point_to_segment_dist(obj_center, p_start, p_end)
                    r_inner = obj['r'] * np.cos(np.deg2rad(30))
                    if dist_to_center > obj['r'] + dist_limit:
                        continue
                    if dist_to_center < r_inner + dist_limit:
                        return True
                    verts = self.get_hex_verts(obj)
                    for i in range(6):
                        v1, v2 = verts[i], verts[(i+1)%6]
                        if self.point_to_segment_dist(p_start, v1, v2) < dist_limit: return True
                        if self.point_to_segment_dist(p_end, v1, v2) < dist_limit: return True
                        if self.point_to_segment_dist(v1, p_start, p_end) < dist_limit: return True
                        if self.segments_intersect(p_start, p_end, v1, v2): return True
        return False

    # ==================== Task 1: TSP (最近邻 + 2-opt) ====================
    def solve_tsp(self):
        """使用最近邻启发式 + 2-opt局部搜索优化焊接顺序 (近似Hamiltonian cycle)"""
        print("Task 1: 正在计算最佳焊接顺序 (最近邻 + 2-opt)...")
        points = self.welding_points
        n = len(points)
        # Step 1: 最近邻构造初始解
        remaining = list(range(n))
        order = []
        curr_pos = self.start_point[:2]
        while remaining:
            nearest = min(remaining, key=lambda i: np.linalg.norm(points[i][:2] - curr_pos))
            order.append(nearest)
            curr_pos = points[nearest][:2]
            remaining.remove(nearest)

        # Step 2: 2-opt 局部优化
        def total_dist(o):
            d = np.linalg.norm(points[o[0]][:2] - self.start_point[:2])
            for i in range(len(o)-1):
                d += np.linalg.norm(points[o[i+1]][:2] - points[o[i]][:2])
            d += np.linalg.norm(self.start_point[:2] - points[o[-1]][:2])
            return d

        improved = True
        while improved:
            improved = False
            best = total_dist(order)
            for i in range(n-1):
                for j in range(i+1, n):
                    new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                    nd = total_dist(new_order)
                    if nd < best - 1e-10:
                        order = new_order
                        best = nd
                        improved = True

        # 构建完整序列: 起点 -> 焊接点 -> 返回起点 (Hamiltonian cycle)
        sequence = [self.start_point]
        for idx in order:
            sequence.append(points[idx])
        sequence.append(self.start_point)
        print(f"  优化后总路径长度: {best:.1f}mm")
        return np.array(sequence)

    def plan_cartesian_path(self, start_cart, end_cart):
        """人工势场法 (APF) 路径规划，带墙壁跟随逃逸和RRT fallback"""
        margin = self.link_radii[2] + self.clearance
        start = np.array(start_cart[:2])
        goal = np.array(end_cart[:2])

        # 若直线无碰撞则直接连线
        if not self.line_hits_obstacle(start, goal, margin):
            return [np.array(start_cart), np.array(end_cart)]

        path_2d = self._apf_plan(start, goal, margin)
        
        # 如果 APF 未能到达目标，使用 RRT fallback
        if np.linalg.norm(path_2d[-1] - goal) > 5.0:
            path_2d = self._rrt_fallback(start, goal, margin)
        
        # --- 路径后处理 ---
        return self._postprocess_path(path_2d, start_cart, end_cart, margin)
    
    def _apf_plan(self, start, goal, margin):
        """APF 核心逻辑"""
        curr = start.copy()
        path_2d = [curr.copy()]
        
        k_att = 1.0
        k_rep = 800.0
        rho0 = 30.0
        step_size = 1.5
        max_iter = 3000
        stall_count = 0
        wall_follow_dir = 1  # 1=逆时针, -1=顺时针
        prev_dist = np.linalg.norm(goal - curr)

        for iteration in range(max_iter):
            d_goal = np.linalg.norm(goal - curr)
            if d_goal < 2.0:
                path_2d.append(goal.copy())
                break

            # --- 引力 ---
            f_att = k_att * (goal - curr) / d_goal

            # --- 斥力 ---
            f_rep = np.zeros(2)
            for obj in self.real_obstacles:
                oc = np.array([obj['x'], obj['y']])
                diff = curr - oc
                
                if obj['type'] == 'circle':
                    d_surface = np.linalg.norm(diff) - obj['r']
                    rep_dir = diff / (np.linalg.norm(diff) + 1e-6)
                elif obj['type'] == 'hexagon':
                    verts = self.get_hex_verts(obj)
                    min_d = float('inf')
                    rep_dir = diff / (np.linalg.norm(diff) + 1e-6)
                    for vi in range(6):
                        v1, v2 = verts[vi], verts[(vi+1)%6]
                        d_edge = self.point_to_segment_dist(curr, v1, v2)
                        if d_edge < min_d:
                            min_d = d_edge
                            edge = v2 - v1
                            t_val = max(0, min(1, np.dot(curr - v1, edge) / (np.dot(edge, edge) + 1e-6)))
                            closest = v1 + t_val * edge
                            n_dir = curr - closest
                            n_len = np.linalg.norm(n_dir)
                            if n_len > 1e-6:
                                rep_dir = n_dir / n_len
                    d_surface = min_d

                if d_surface < rho0 and d_surface > 0:
                    d_s = max(d_surface, 0.5)
                    magnitude = k_rep * (1.0/d_s - 1.0/rho0) / (d_s**2)
                    f_rep += magnitude * rep_dir

            f_total = f_att + f_rep

            # --- 局部极小值检测与墙壁跟随逃逸 ---
            if d_goal >= prev_dist - 0.05:
                stall_count += 1
            else:
                stall_count = 0
            prev_dist = d_goal

            if stall_count > 10:
                # 墙壁跟随：沿合力的切线方向持续移动
                f_norm = np.linalg.norm(f_total)
                if f_norm > 1e-6:
                    tang = f_total / f_norm
                else:
                    tang = (goal - curr) / d_goal
                # 旋转 90 度 (wall following)
                f_total = np.array([-tang[1], tang[0]]) * wall_follow_dir
                if stall_count > 50:
                    # 切换跟随方向
                    wall_follow_dir *= -1
                    stall_count = 0

            # --- 步进 ---
            f_norm = np.linalg.norm(f_total)
            if f_norm > 1e-6:
                direction = f_total / f_norm
            else:
                direction = (goal - curr) / d_goal
            
            next_p = curr + direction * step_size

            # 碰撞回退
            if self.point_in_obstacle(next_p[0], next_p[1], margin):
                tangent = np.array([-direction[1], direction[0]])
                next_p = curr + tangent * step_size
                if self.point_in_obstacle(next_p[0], next_p[1], margin):
                    next_p = curr - tangent * step_size
                    if self.point_in_obstacle(next_p[0], next_p[1], margin):
                        continue  # 跳过此步

            curr = next_p
            path_2d.append(curr.copy())
        
        return path_2d

    def _rrt_fallback(self, start, goal, margin):
        """RRT 后备方案：当 APF 无法到达目标时使用"""
        tree = {tuple(start): None}
        nodes = [start.copy()]
        
        for _ in range(5000):
            sample = goal if np.random.rand() < 0.3 else np.array([np.random.uniform(0, 300), np.random.uniform(0, 200)])
            nearest = nodes[np.argmin([np.linalg.norm(n - sample) for n in nodes])]
            
            diff = sample - nearest
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            new_pt = nearest + 5.0 * diff / dist

            if self.point_in_obstacle(new_pt[0], new_pt[1], margin): continue
            if self.line_hits_obstacle(nearest, new_pt, margin, 8): continue

            tree[tuple(new_pt)] = nearest
            nodes.append(new_pt)

            if np.linalg.norm(new_pt - goal) < 5.0:
                if not self.line_hits_obstacle(new_pt, goal, margin, 8):
                    tree[tuple(goal)] = new_pt
                    break

        # 回溯
        path = []
        curr = goal if tuple(goal) in tree else nodes[np.argmin([np.linalg.norm(n - goal) for n in nodes])]
        while curr is not None:
            path.append(curr)
            curr = tree.get(tuple(curr))
        return path[::-1]

    def _postprocess_path(self, path_2d, start_cart, end_cart, margin):
        """路径后处理：剪枝 + 转换为 [x, y, phi]"""
        smoothed = [path_2d[0]]
        i = 0
        while i < len(path_2d) - 1:
            j = len(path_2d) - 1
            while j > i + 1:
                if not self.line_hits_obstacle(path_2d[i], path_2d[j], margin, 15):
                    break
                j -= 1
            smoothed.append(path_2d[j])
            i = j

        # 多轮随机剪枝
        for _ in range(30):
            if len(smoothed) < 3: break
            idx = np.random.randint(0, len(smoothed) - 2)
            if not self.line_hits_obstacle(smoothed[idx], smoothed[idx + 2] if idx + 2 < len(smoothed) else smoothed[-1], margin, 15):
                if idx + 2 < len(smoothed):
                    smoothed.pop(idx + 1)

        # 转换为 [x, y, phi]
        phi_s, phi_e = start_cart[2], end_cart[2]
        total_len = sum(np.linalg.norm(smoothed[k+1]-smoothed[k]) for k in range(len(smoothed)-1))
        cart_path = []
        cum = 0.0
        for k, pt in enumerate(smoothed):
            if k > 0:
                cum += np.linalg.norm(pt - smoothed[k-1])
            ratio = cum / (total_len + 1e-6)
            cart_path.append(np.array([pt[0], pt[1], phi_s + ratio*(phi_e - phi_s)]))
        return cart_path

    # ==================== Task 2: 关节空间RRT (用于连杆避障) ====================
    def plan_joint_path(self, start_state, end_state):
        """在关节空间用RRT规划路径，避免连杆碰撞"""
        start_joints = self.inverse_kinematics(*start_state)
        end_joints = self.inverse_kinematics(*end_state)
        if start_joints is None or end_joints is None:
            return [start_joints if start_joints is not None else end_joints,
                    end_joints if end_joints is not None else start_joints]

        tree = {tuple(start_joints): None}
        nodes = [start_joints]

        for _ in range(10000):
            if np.random.rand() < 0.2:
                sample = end_joints
            else:
                sample = np.random.uniform(-np.pi, np.pi, 3)

            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest = nodes[np.argmin(dists)]

            step_size = 0.05
            direction = sample - nearest
            dist = np.linalg.norm(direction)
            if dist < 1e-6: continue
            new_node = nearest + step_size * direction / dist

            collision = False
            for alpha in np.linspace(0, 1, 5):
                inter = nearest + alpha * (new_node - nearest)
                if self.check_collision(inter, self.obstacles):
                    collision = True; break

            if not collision:
                tree[tuple(new_node)] = nearest
                nodes.append(new_node)
                if np.linalg.norm(new_node - end_joints) < step_size:
                    ok = True
                    for alpha in np.linspace(0, 1, 5):
                        inter = new_node + alpha * (end_joints - new_node)
                        if self.check_collision(inter, self.obstacles):
                            ok = False; break
                    if ok:
                        tree[tuple(end_joints)] = new_node
                        break

        path = []
        curr = end_joints
        while curr is not None:
            path.append(curr)
            curr = tree.get(tuple(curr))
        return path[::-1] if len(path) > 1 else [start_joints, end_joints]

    # ==================== Task 3: 轨迹规划 (三次多项式) ====================
    def generate_cartesian_trajectory(self, cart_path, steps_total=100):
        """
        使用全局三次样条插值 (Cubic Spline) 替代分段零速度插值。
        这能保证末端执行器在路径点之间的速度 (C1) 和加速度 (C2) 连续，使路径平滑。
        """
        if len(cart_path) < 2:
            return []
            
        cart_path = np.array(cart_path)
        # 计算累计路径长度作为参数化时间 t
        dists = np.sqrt(np.sum(np.diff(cart_path[:, :2], axis=0)**2, axis=1))
        t_steps = np.zeros(len(cart_path))
        t_steps[1:] = np.cumsum(dists)
        
        # 归一化时间到 [0, 1]
        if t_steps[-1] > 1e-6:
            t_steps /= t_steps[-1]
        else:
            # 起点终点重合的情况
            joints = self.inverse_kinematics(*cart_path[0])
            return [joints] * steps_total if joints is not None else []

        # 对 x, y, phi 分别进行样条插值 (clamped 边界确保起止速度平稳)
        cs_x = CubicSpline(t_steps, cart_path[:, 0], bc_type='clamped')
        cs_y = CubicSpline(t_steps, cart_path[:, 1], bc_type='clamped')
        cs_phi = CubicSpline(t_steps, cart_path[:, 2], bc_type='clamped')

        trajectory = []
        for t in np.linspace(0, 1, steps_total):
            px = cs_x(t)
            py = cs_y(t)
            pphi = cs_phi(t)
            
            joints = self.inverse_kinematics(px, py, pphi)
            if joints is not None:
                trajectory.append(joints)
        return trajectory

    # ==================== Task 4: 轨迹碰撞检查与修正 ====================
    def check_and_fix_trajectory(self, traj, label=""):
        """检查轨迹碰撞，返回(碰撞数, 是否通过)"""
        collisions = 0
        for q in traj:
            if q is not None and self.check_collision(q, self.obstacles):
                collisions += 1
        return collisions

    # ==================== 可视化 ====================
    def _draw_scene(self, ax, title):
        """在ax上绘制底板、障碍物、焊点样式"""
        ax.plot([0, 300, 300, 0, 0], [0, 0, 200, 200, 0], 'k--', alpha=0.5, label='Base Plate')
        
        # 起点 (Start Point)
        ax.add_patch(plt.Circle((295, 15), 3, color='green', alpha=0.6, label='Start Point'))
        
        # 障碍物
        for obj in self.real_obstacles:
            if obj['type'] == 'circle':
                c = plt.Circle((obj['x'], obj['y']), obj['r'], color='gray', alpha=0.3)
                ax.add_patch(c)
            elif obj['type'] == 'hexagon':
                angles = np.deg2rad([0, 60, 120, 180, 240, 300])
                verts = [(obj['x']+obj['r']*np.cos(a), obj['y']+obj['r']*np.sin(a)) for a in angles]
                ax.add_patch(plt.Polygon(verts, color='gray', alpha=0.3))
        
        # 焊点 (Welding Points) - 半径 5mm (直径 10mm)，颜色红色
        for p in self.welding_points:
            ax.add_patch(plt.Circle((p[0], p[1]), 5.0, color='red', alpha=0.6, zorder=4))
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-50, 350); ax.set_ylim(-30, 250)
        ax.set_title(title); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle=':', alpha=0.4)

    def visualize(self, cart_trajs, sequence):
        """可视化底板、障碍物、焊接点和机器人笛卡尔空间轨迹"""
        plt.figure(figsize=(12, 9))
        ax = plt.gca()
        # 绘制背景（底板、起点、障碍物、焊接点）
        self._draw_scene(ax, "Robot Welding Path Planning")
        # 绘制末端执行器轨迹
        print("正在生成轨迹图...")
        plotted = False
        for traj in cart_trajs:
            xs, ys = [], []
            for q in traj:
                if q is not None:
                    _, _, _, p3 = self.forward_kinematics(q)
                    xs.append(p3[0]); ys.append(p3[1])
            if xs:
                lbl = 'Welding Path' if not plotted else None
                ax.plot(xs, ys, 'b-', lw=1.5, alpha=0.7, label=lbl)
                plotted = True
            
        # 绘制焊接顺序编号
        for k in range(1, len(sequence)-1):
            ax.annotate(str(k), xy=sequence[k][:2], fontsize=8, color='darkred',
                       ha='center', va='bottom', fontweight='bold')
            
        ax.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.show()

    # ==================== 主流程 ====================
    def run(self):
        # Task 1: 焊接顺序
        sequence = self.solve_tsp()
        n_segs = len(sequence) - 1
        print(f"  焊接顺序点位数: {len(sequence)} (含起止点)")

        cart_trajs = []
        total_cart_collisions = 0

        for i in range(n_segs):
            print(f"Task 2&3: 规划路径段 {i+1}/{n_segs}...", end="\r")

            # Task 2: 笛卡尔空间路径规划 (末端轨迹绕开障碍物)
            cart_path = self.plan_cartesian_path(sequence[i], sequence[i+1])

            # Task 3: 笛卡尔空间轨迹 — 笛卡尔路径 + 三次多项式插值 + IK
            ct = self.generate_cartesian_trajectory(cart_path)
            cart_trajs.append(ct)

            # Task 4: 碰撞检查
            total_cart_collisions += self.check_and_fix_trajectory(ct)

        print(f"\nTask 4 碰撞检查结果:")
        print(f"  笛卡尔空间轨迹碰撞点数: {total_cart_collisions}")
        print("所有路径规划完成！正在生成可视化...")
        self.visualize(cart_trajs, sequence)

if __name__ == "__main__":
    project = WeldingProject()
    project.run()