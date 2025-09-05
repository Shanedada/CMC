## 导弹位置 M(xm, ym, zm) 和目标圆柱体 C（底面圆心 (0,200,0)，半径 7，高 10）之间的连线（即所有圆柱表面点到 M 的连线），到烟团球心 S(xs, ys, zs) 的最短距离。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
def sample_cylinder_surface(center, radius, height, num_samples=1000000):
    """
    在圆柱体表面进行均匀采样
    
    参数:
    center: 底面圆心坐标 (x, y, z)
    radius: 圆柱体半径
    height: 圆柱体高度
    num_samples: 采样点数量
    
    返回:
    samples: 采样点坐标数组 (num_samples, 3)
    """
    samples = np.zeros((num_samples, 3))
    
    # 底面和顶面采样点数量 (各占1/3)
    num_circular = num_samples // 3
    # 侧面采样点数量 (占2/3)
    num_lateral = num_samples - 2 * num_circular
    
    # 1. 底面采样 (z = center[2])
    for i in range(num_circular):
        # 在圆内均匀采样
        r = radius * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2]
        samples[i] = [x, y, z]
    
    # 2. 顶面采样 (z = center[2] + height)
    for i in range(num_circular, 2 * num_circular):
        # 在圆内均匀采样
        r = radius * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + height
        samples[i] = [x, y, z]
    
    # 3. 侧面采样
    for i in range(2 * num_circular, num_samples):
        # 在圆柱侧面均匀采样
        theta = 2 * np.pi * np.random.random()  # 角度
        z = center[2] + height * np.random.random()  # 高度
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        samples[i] = [x, y, z]
    
    return samples

# 可视化采样结果
def plot_cylinder_samples(samples, center, radius, height, max_points=10000):
    """
    可视化圆柱体采样点（为了性能，只显示部分点）
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 随机选择部分点进行显示
    if len(samples) > max_points:
        indices = np.random.choice(len(samples), max_points, replace=False)
        display_samples = samples[indices]
    else:
        display_samples = samples
    
    # 绘制采样点
    ax.scatter(display_samples[:, 0], display_samples[:, 1], display_samples[:, 2], 
               c='red', s=1, alpha=0.6, label='采样点')
    
    # 绘制圆柱体中心点
    ax.scatter(center[0], center[1], center[2], c='blue', s=100, label='底面圆心')
    ax.scatter(center[0], center[1], center[2] + height, c='green', s=100, label='顶面圆心')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'圆柱体表面采样 (半径={radius}m, 高度={height}m)')
    ax.legend()
    
    # 设置坐标轴范围
    margin = max(radius, height) * 0.2
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin)
    ax.set_zlim(center[2] - margin, center[2] + height + margin)
    
    plt.tight_layout()
    plt.show()

# 可视化
cylinder_center = np.array([0, 200, 0])
cylinder_radius = 7.0
cylinder_height = 10.0
num_samples = 1000000
cylinder_samples = sample_cylinder_surface(cylinder_center, cylinder_radius, cylinder_height, num_samples)
# 若要画图，取消注释
# plot_cylinder_samples(cylinder_samples, cylinder_center, cylinder_radius, cylinder_height)

def line_dis(x, y, z):
    '''
    计算点z到线段 x,y之间的距离。
    '''
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    
    # 向量
    xy = y - x
    xz = z - x
    
    # 计算线段长度的平方
    xy_len_sq = np.dot(xy, xy)
    
    # 如果线段长度为0，返回点到x的距离
    if xy_len_sq == 0:
        return np.linalg.norm(xz)
    
    # 计算投影参数t
    t = np.dot(xz, xy) / xy_len_sq
    
    # 如果投影在线段外，返回到端点的距离
    if t < 0:
        return np.linalg.norm(xz)  # 到x点的距离
    elif t > 1:
        return np.linalg.norm(z - y)  # 到y点的距离
    else:
        # 投影在线段内，计算垂直距离
        projection = x + t * xy
        return np.linalg.norm(z - projection)

def min_distance_missile_cylinder_to_cloud(M_pos, S_pos, cylinder_center, cylinder_radius, cylinder_height, num_samples=1000000):
    """
    计算导弹 M_pos 与圆柱表面所有点连线到烟团球心 S_pos 的最小距离

    参数:
    M_pos: 导弹坐标 (3,)
    S_pos: 烟团球心坐标 (3,)
    cylinder_center: 圆柱底面圆心 (3,)
    cylinder_radius: 圆柱半径
    cylinder_height: 圆柱高度
    num_samples: 圆柱表面采样点数量

    返回:
    min_dist: 最小距离
    """
    # 1. 采样圆柱表面
    surface_points = sample_cylinder_surface(cylinder_center, cylinder_radius, cylinder_height, num_samples)
    
    # 2. 遍历每个表面点，计算点到线段距离，求最小
    max_dist = float(0)
    for P in surface_points:
        d = line_dis(M_pos, P, S_pos)
        if d > max_dist:
            max_dist = d
    return max_dist