%% 数据
M = [20000,0,2000; 19000,600,2100; 18000,-600,1900]; % 导弹 M1-M3
FY = [17800,0,1800; 12000,1400,1400; 6000,-3000,700; 11000,2000,1800; 13000,-2000,1300]; % 无人机 FY1-FY5

% 目标参数（圆柱：半径7，高10；假目标在原点，真目标底面圆心(0,200,0)）
r = 7; h = 10; seg = 100;  % seg: 圆周分段数，越大越圆滑

%% 画布布局：左-全局，右-局部放大（能清楚看到圆柱）
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% ========== 左：全局概览 ==========
nexttile; hold on; grid on; view(3);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('全局概览');

% 导弹与无人机
scatter3(M(:,1),M(:,2),M(:,3),40,'r','filled'); 
text(M(:,1),M(:,2),M(:,3),{'M1','M2','M3'},'Color','r','FontSize',8,'VerticalAlignment','bottom');
scatter3(FY(:,1),FY(:,2),FY(:,3),40,'b','filled');
text(FY(:,1),FY(:,2),FY(:,3),{'FY1','FY2','FY3','FY4','FY5'},'Color','b','FontSize',8,'VerticalAlignment','bottom');

% 为保证 z≥0，只设正向 z 轴范围
maxZ = max([M(:,3); FY(:,3); h]);
zlim([0, maxZ*1.1]);

legend('Missiles','UAVs','Location','northeast');

%% ========== 右：局部放大（目标区，显示两个圆柱） ==========
nexttile; hold on; grid on; axis equal; view(3);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('目标区放大（含假/真目标圆柱）');

% 仍然放上导弹/无人机，但缩小点大小
scatter3(M(:,1),M(:,2),M(:,3),15,'r','filled'); 
scatter3(FY(:,1),FY(:,2),FY(:,3),15,'b','filled');

% 圆柱外表面（假目标：中心(0,0,0)；真目标：中心(0,200,0)）
[XC,YC,ZC] = cylinder(r, seg);  % ZC 在 [0,1]
ZC = ZC * h;                    % 高度缩放到 [0,h]

% 假目标（绿色）
surf(XC + 0, YC + 0, ZC + 0, 'FaceColor',[0.2 0.8 0.2], 'FaceAlpha',0.6, 'EdgeColor','none');
% 盖子
th = linspace(0,2*pi,seg+1);
xcap = r*cos(th); ycap = r*sin(th);
fill3(xcap, ycap, 0*th + 0, [0.2 0.8 0.2], 'FaceAlpha',0.6, 'EdgeColor','none');      % 底盖 z=0
fill3(xcap, ycap, 0*th + h, [0.2 0.8 0.2], 'FaceAlpha',0.6, 'EdgeColor','none');      % 顶盖 z=h

% 真目标（紫色，y 平移 200）
surf(XC + 0, YC + 200, ZC + 0, 'FaceColor',[0.6 0.2 0.8], 'FaceAlpha',0.6, 'EdgeColor','none');
fill3(xcap + 0, ycap + 200, 0*th + 0, [0.6 0.2 0.8], 'FaceAlpha',0.6, 'EdgeColor','none'); % 底盖
fill3(xcap + 0, ycap + 200, 0*th + h, [0.6 0.2 0.8], 'FaceAlpha',0.6, 'EdgeColor','none'); % 顶盖

% 只显示 z≥0 的范围
zlim([0, max(h, maxZ*0.15)]);

% 为了看得清楚：只围绕目标区设定范围（x 方向±20 m，y 覆盖两个圆柱）
xlim([-20, 20]);
ylim([-20, 220+20]);

% 灯光与材质增强立体感
camlight headlight; lighting gouraud; material dull;

legend({'Missiles','UAVs','假目标(0,0,0)','真目标(0,200,0)'},'Location','northeast');

%% 可选：把两个圆柱顶部连线画出来（强调二者相对位置）
plot3([0,0],[0,200],[h,h],'k--','LineWidth',1);
