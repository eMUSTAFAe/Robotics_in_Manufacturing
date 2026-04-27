% ==========================================
% SMMG 6000M 机器人制造 - 机械臂动画 (V10 - 终极空间版)
% ==========================================
clear; clc; close all;

%% 1. 加载轨迹数据
try
    data = readmatrix('trajectory.csv'); 
catch
    error('错误：未找到 trajectory.csv！请确保 Python 代码已运行并生成该文件。');
end

%% 2. 机械臂几何参数 (同步自 main(1).py)
L1 = 170; % Link-1 长度
L2 = 160; % Link-2 长度
L3_h = 10; % 焊枪水平偏移
L3_v = 40; % 垂直连接段高度
base_pos = [15, 15]; % 底座偏移
arm_z = 80; % 机械臂运动高度

%% 3. 初始化 3D 绘图舞台 (大幅增加"透明空间")
fig = figure('Name', 'Robotic Welding Simulation V10', 'Color', 'w');
hold on; grid on; axis equal; view(3);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');

% --- 核心修改：设定超大坐标轴范围，彻底解决显示不全问题 ---
% X 从 -250 到 550，Y 从 -250 到 550
% Z 轴也适当加高到 250
axis([-250 550 -250 550 0 250]); 

% 绘制底板
patch([0 300 300 0], [0 0 200 200], [0 0 0 0], [0.9 0.9 0.9], 'FaceAlpha', 0.2);

% --- 绘制障碍物 (包含真实六棱柱) ---
circles = [80, 140, 25; 240, 140, 35; 150, 40, 30];
[cx, cy, cz] = cylinder(1, 50);
for i = 1:size(circles, 1)
    o = circles(i,:);
    surf(cx*o(3) + o(1), cy*o(3) + o(2), cz*50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
end

hexagons = [150, 140, 20; 250, 40, 20];
angles = deg2rad(0:60:360); 
for i = 1:size(hexagons, 1)
    h = hexagons(i,:);
    xv = h(3) * cos(angles) + h(1);
    yv = h(3) * sin(angles) + h(2);
    zv_bot = zeros(size(xv));
    zv_top = ones(size(xv)) * 50; 
    for j = 1:6
        patch([xv(j) xv(j+1) xv(j+1) xv(j)], [yv(j) yv(j+1) yv(j+1) yv(j)], ...
              [zv_bot(j) zv_bot(j+1) zv_top(j+1) zv_top(j)], [0.5 0.5 0.5], 'EdgeColor', 'k');
    end
    patch(xv, yv, zv_top, [0.5 0.5 0.5], 'EdgeColor', 'k');
end

%% 4. 初始化图形对象 (全部采用 line 函数独立定义)
% 独立渲染每一段，防止因坐标数组过大导致的显示丢失
h_link1  = line([0,0], [0,0], [0,0], 'Color', 'k', 'LineWidth', 6); % 黑色
h_link2  = line([0,0], [0,0], [0,0], 'Color', 'k', 'LineWidth', 6); % 黑色
h_link3v = line([0,0], [0,0], [0,0], 'Color', 'k', 'LineWidth', 4); % 黑色
h_link3h = line([0,0], [0,0], [0,0], 'Color', 'k', 'LineWidth', 4); % 黑色
h_joints = scatter3(zeros(1,4), zeros(1,4), zeros(1,4), 70, 'ro', 'filled'); % 红色关节
h_torch  = line([0,0], [0,0], [0,0], 'Color', 'b', 'LineWidth', 5); % 蓝色焊针

%% 5. 播放动画循环
total_frames = size(data, 1);
for i = 1:total_frames
    th1 = data(i, 1); th2 = data(i, 2); th3 = data(i, 3);
    
    % --- 正向运动学计算 (带底座偏移) ---
    x0 = base_pos(1); y0 = base_pos(2);
    x1 = x0 + L1 * cos(th1); y1 = y0 + L1 * sin(th1);
    x2 = x1 + L2 * cos(th1 + th2); y2 = y1 + L2 * sin(th1 + th2);
    phi = th1 + th2 + th3;
    x3 = x2 + L3_h * cos(phi); y3 = y2 + L3_h * sin(phi);
    
    % --- 独立更新各个部件的数据 ---
    set(h_link1, 'XData', [x0, x1], 'YData', [y0, y1], 'ZData', [arm_z, arm_z]);
    set(h_link2, 'XData', [x1, x2], 'YData', [y1, y2], 'ZData', [arm_z, arm_z]);
    set(h_link3v, 'XData', [x2, x2], 'YData', [y2, y2], 'ZData', [arm_z, arm_z - L3_v]);
    set(h_link3h, 'XData', [x2, x3], 'YData', [y2, y3], 'ZData', [arm_z - L3_v, arm_z - L3_v]);
    set(h_joints, 'XData', [x0, x1, x2, x2], 'YData', [y0, y1, y2, y2], 'ZData', [arm_z, arm_z, arm_z, arm_z - L3_v]);
    set(h_torch,  'XData', [x3, x3], 'YData', [y3, y3], 'ZData', [arm_z - L3_v, 0]);
    
    drawnow; 
    pause(0.01); % 调节此处改变速度
end