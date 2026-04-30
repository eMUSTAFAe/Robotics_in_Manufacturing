% Simulation using welding_trajectory_detailed.csv
% 基于 new.m 改写，读取 welding_trajectory_detailed.csv 并复现 3D 动画
clear; clc; close all;

%% 1. 加载轨迹数据
csvFile = 'welding_trajectory_detailed.csv';
if ~isfile(csvFile)
	error(['未找到 ', csvFile, '。请先运行 Python 脚本以生成该文件。']);
end

% 读取 CSV（跳过表头）
opts = detectImportOptions(csvFile);
data = readmatrix(csvFile, opts);

if isempty(data)
	error('CSV 文件为空或格式不正确。');
end

% CSV 中我们期望 theta1,theta2,theta3 分别在最后三列
th_cols = size(data,2) - 2 : size(data,2);
th = data(:, th_cols);

%% 2.  （与 main.py 保持一致）
L1 = 170; L2 = 160; L3_h = 10; L3_v = 40; % mm
base_pos = [15, 15];
arm_z = 80; % 绘图中机械臂高度

%% 3. 初始化图形
fig = figure('Name','Welding Trajectory Simulation','Color','w');
hold on; grid on; axis equal; view(3);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis([-250 550 -250 550 0 250]);

% 绘制底板
patch([0 300 300 0],[0 0 200 200],[0 0 0 0],[0.9 0.9 0.9],'FaceAlpha',0.2);

% 绘制障碍物（与 new.m 对齐）
circles = [80,140,25; 240,140,35; 150,40,30];
[cx, cy, cz] = cylinder(1,50);
for i=1:size(circles,1)
	o = circles(i,:);
	surf(cx*o(3)+o(1), cy*o(3)+o(2), cz*50, 'FaceColor',[0.7 0.7 0.7],'EdgeColor','none');
end

hexagons = [150,140,20; 250,40,20];
angles = deg2rad(0:60:360);
for i=1:size(hexagons,1)
	h = hexagons(i,:);
	xv = h(3)*cos(angles)+h(1);
	yv = h(3)*sin(angles)+h(2);
	zv_bot = zeros(size(xv)); zv_top = ones(size(xv))*50;
	for j=1:6
		patch([xv(j) xv(j+1) xv(j+1) xv(j)], [yv(j) yv(j+1) yv(j+1) yv(j)], ...
			  [zv_bot(j) zv_bot(j+1) zv_top(j+1) zv_top(j)], [0.5 0.5 0.5], 'EdgeColor','k');
	end
	patch(xv, yv, zv_top, [0.5 0.5 0.5], 'EdgeColor','k');
end

%% 4. 初始化图形对象
h_link1  = line([0,0],[0,0],[0,0],'Color','k','LineWidth',6);
h_link2  = line([0,0],[0,0],[0,0],'Color','k','LineWidth',6);
h_link3v = line([0,0],[0,0],[0,0],'Color','k','LineWidth',4);
h_link3h = line([0,0],[0,0],[0,0],'Color','k','LineWidth',4);
h_joints = scatter3(zeros(1,4),zeros(1,4),zeros(1,4),70,'ro','filled');
h_torch  = line([0,0],[0,0],[0,0],'Color','b','LineWidth',5);

total_frames = size(th,1);
fprintf('Loaded %d frames from %s\n', total_frames, csvFile);

%% 5. 播放动画
% --- 视频输出设置（可选） ---
videoFile = 'welding_sim.mp4';
videoWriterAvailable = false;
try
	v = VideoWriter(videoFile, 'MPEG-4');
	v.FrameRate = 30;
	v.Quality = 90;
	open(v);
	videoWriterAvailable = true;
catch
	warning('MPEG-4 编码不可用，尝试 Motion JPEG AVI...');
	try
		videoFile = 'welding_sim.avi';
		v = VideoWriter(videoFile, 'Motion JPEG AVI');
		v.FrameRate = 30;
		open(v);
		videoWriterAvailable = true;
	catch
		warning('无法创建视频写入器，视频将不会保存。');
		videoWriterAvailable = false;
	end
end

for k = 1:total_frames
	th1 = th(k,1); th2 = th(k,2); th3 = th(k,3);

	% 正向运动学（带底座偏移）
	x0 = base_pos(1); y0 = base_pos(2);
	x1 = x0 + L1 * cos(th1); y1 = y0 + L1 * sin(th1);
	x2 = x1 + L2 * cos(th1 + th2); y2 = y1 + L2 * sin(th1 + th2);
	phi = th1 + th2 + th3;
	x3 = x2 + L3_h * cos(phi); y3 = y2 + L3_h * sin(phi);

	% 更新图形对象
	set(h_link1, 'XData', [x0, x1], 'YData', [y0, y1], 'ZData', [arm_z, arm_z]);
	set(h_link2, 'XData', [x1, x2], 'YData', [y1, y2], 'ZData', [arm_z, arm_z]);
	set(h_link3v,'XData',[x2,x2],'YData',[y2,y2],'ZData',[arm_z, arm_z - L3_v]);
	set(h_link3h,'XData',[x2,x3],'YData',[y2,y3],'ZData',[arm_z - L3_v, arm_z - L3_v]);
	set(h_joints, 'XData', [x0, x1, x2, x2], 'YData', [y0, y1, y2, y2], 'ZData', [arm_z, arm_z, arm_z, arm_z - L3_v]);
	set(h_torch,  'XData', [x3, x3], 'YData', [y3, y3], 'ZData', [arm_z - L3_v, 0]);

	drawnow;
	pause(0.01);

	% 写入视频帧（若可用）
	if videoWriterAvailable
		try
			frame = getframe(fig);
			writeVideo(v, frame);
		catch
			% 写帧出错则继续（不阻止动画）
		end
	end
end

fprintf('Simulation finished.\n');

% 关闭视频写入器并提示文件位置
if videoWriterAvailable
	try
		close(v);
		fprintf('Saved video: %s\n', videoFile);
	catch
		warning('关闭视频写入器时出错。');
	end
end

