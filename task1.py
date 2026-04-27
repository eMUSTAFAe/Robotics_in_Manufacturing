import numpy as np

def tsp_dynamic_programming(points):
    """
    动态规划求解 TSP 问题（最短哈密顿回路）
    :param points: 焊接点坐标列表，格式为 [[x1,y1], [x2,y2], ..., [xn,yn]]
    :return: 
        - min_distance: 最短路径总长度
        - best_path: 最优访问顺序（索引列表，如 [0,5,3,...,2]）
    """
    # 步骤1：计算距离矩阵（任意两点间的欧氏距离）
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # 欧氏距离公式：d = √[(x1-x2)² + (y1-y2)²]
                dist_matrix[i][j] = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            else:
                dist_matrix[i][j] = 0  # 自身到自身距离为0

    # 步骤2：初始化DP表（状态压缩）
    # DP[mask][u]：mask表示已访问的点集合（二进制），u表示当前所在点，值为最短路径长度
    # mask用二进制表示：比如mask=0b1011表示访问了第0、1、3个点
    INF = float('inf')
    DP = np.full((1 << n, n), INF)
    # 前驱节点表：记录最优路径的前驱，用于回溯路径
    prev = np.full((1 << n, n), -1)
    
    # 初始状态：从第0个点出发，只访问第0个点，距离为0
    DP[1 << 0][0] = 0

    # 步骤3：动态规划核心循环
    # 遍历所有可能的点集合（mask）
    for mask in range(1, 1 << n):
        # 遍历当前所在的点u
        for u in range(n):
            # 如果u不在当前集合中，跳过
            if not (mask & (1 << u)):
                continue
            # 如果当前状态距离为无穷大，跳过（不可达）
            if DP[mask][u] == INF:
                continue
            # 遍历下一个要访问的点v（未访问过的点）
            for v in range(n):
                if mask & (1 << v):
                    continue  # v已访问，跳过
                # 新的集合：加入v
                new_mask = mask | (1 << v)
                # 更新DP表：取更小的距离
                if DP[new_mask][v] > DP[mask][u] + dist_matrix[u][v]:
                    DP[new_mask][v] = DP[mask][u] + dist_matrix[u][v]
                    prev[new_mask][v] = u  # 记录前驱节点

    # 步骤4：回溯找最优路径
    # 最终状态：所有点都访问过（mask=(1<<n)-1），回到起点0
    min_distance = INF
    end_node = -1
    full_mask = (1 << n) - 1  # 所有位都为1，表示所有点都访问过
    for u in range(n):
        if DP[full_mask][u] + dist_matrix[u][0] < min_distance:
            min_distance = DP[full_mask][u] + dist_matrix[u][0]
            end_node = u

    # 回溯路径
    best_path = []
    current_mask = full_mask
    current_node = end_node
    while current_node != -1:
        best_path.append(current_node)
        next_node = prev[current_mask][current_node]
        current_mask = current_mask & ~(1 << current_node)  # 移除当前节点
        current_node = next_node

    # 反转路径（回溯是倒序），并补回起点（形成回路）
    best_path = best_path[::-1]
    best_path.append(0)  # 最后回到起点（如果不需要回路可删除此行）

    return min_distance, best_path

# ====================== 调用示例（适配你们的项目） ======================
if __name__ == "__main__":
    # 模拟24个焊接点坐标（替换成你们的实际坐标）
    # 格式：[[x1,y1], [x2,y2], ..., [x24,y24]]
    weld_points = [
        [100, 200], [150, 220], [120, 250], [180, 230],
        [200, 210], [220, 240], [190, 260], [170, 280],
        [140, 270], [110, 290], [90, 260], [80, 230],
        [70, 200], [90, 180], [120, 170], [150, 190],
        [180, 160], [210, 180], [230, 200], [250, 220],
        [240, 250], [220, 270], [200, 290], [170, 280]
    ]

    # 求解TSP
    min_dist, best_seq = tsp_dynamic_programming(weld_points)

    # 输出结果
    print(f"最短路径总长度：{min_dist:.2f} mm")
    print(f"最优焊接点访问顺序（索引）：{best_seq}")
    # 输出对应的坐标
    print("最优顺序对应的焊接点坐标：")
    for idx in best_seq:
        print(f"第{idx+1}个点：{weld_points[idx]}")