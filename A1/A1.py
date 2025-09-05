import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 初始数据 (x, y, z)
M1_pos = np.array([20000.0, 0.0, 2000.0])   # (x,z) 导弹位置
FY1_pos = np.array([17800.0, 0.0, 1800.0])  # (x,z) 无人机位置
false_target = np.array([0.0, 0.0, 0.0])  # 假目标
true_target = np.array([0.0, 200.0, 0.0])  # 真目标


v_M = 300.0  # m/s
v_FY = 120.0  # m/s

t_release = 1.5  # 投放时间
t_explode = t_release + 3.6  # 起爆时间

g = 9.80665  # 重力加速度
sin_theta = 1 / np.sqrt(100 + 1)
cos_theta = 10 / np.sqrt(100 + 1)

# 导弹运动方程计算 M1_pos -> M_pos
def missile_trajectory(t):
    # 从 M1_pos 到 target 的直线运动
    M_pos = M1_pos - v_M * t * np.array([cos_theta, 0, sin_theta])
    return M_pos

# ========  0 - t_release 无人机平飞 =========

FY1_pos = np.array([17800.0, 0.0, 1800.0]) - v_FY * t_release * np.array([1, 0, 0])

print("无人机投放时位置：", FY1_pos)
# ========  t_release - t_explode 干扰弹平抛运动 ======== 
Bomb_pos = FY1_pos

Bomb_pos = Bomb_pos - v_FY * (t_explode - t_release) * np.array([1, 0, 0]) - 0.5 * g * (t_explode - t_release) ** 2 * np.array([0, 0, 1])

print("干扰弹起爆时位置：", Bomb_pos)

# ========  t_release 之后 ========
def Bomb_cal_pos(t):
    '''
    计算干扰弹位置 自由下落 3 m/s
    '''
    t -= t_explode
    return Bomb_pos - 3 * t * np.array([0, 0, 1])

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

def dis_func(t):
    return line_dis(true_target, missile_trajectory(t), Bomb_cal_pos(t))

def find_roots(f, a, b, threshold=10, step=0.1, tol=1e-5):
    """
    在区间 [a,b] 内寻找 f(t) = threshold 的解
    返回所有交点列表
    """
    roots = []
    t = a
    prev_val = f(t) - threshold
    while t <= b:
        t_next = t + step
        val = f(t_next) - threshold
        # 如果跨过了 0，说明在 [t, t_next] 有解
        if prev_val * val <= 0:
            lo, hi = t, t_next
            # 二分法逼近
            while hi - lo > tol:
                mid = (lo + hi) / 2
                if (f(lo) - threshold) * (f(mid) - threshold) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        prev_val = val
        t = t_next
    return roots

# ===== 示例调用 =====
# dis_func(t) 是自己定义的距离函数

roots = find_roots(dis_func, a=t_explode, b=t_explode + 20, threshold=10, step=0.1)
print("交点:", roots)
if len(roots) == 2:
    print("有效遮蔽时长 =", roots[1] - roots[0], "秒")

# ========== 圆柱体表面采样 ==========
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

# 圆柱体参数
cylinder_center = true_target  # 底面圆心
cylinder_radius = 7.0  # 半径 7m
cylinder_height = 10.0  # 高度 10m
num_samples = 10000  # 采样点数量

# 进行圆柱体表面采样
cylinder_samples = sample_cylinder_surface(
    center=cylinder_center,
    radius=cylinder_radius,
    height=cylinder_height,
    num_samples=num_samples
)

print(f"\n圆柱体表面采样完成！")
print(f"采样点数量: {len(cylinder_samples)}")
print(f"圆柱体参数: 底面圆心 {cylinder_center}, 半径 {cylinder_radius}m, 高度 {cylinder_height}m")
print(f"采样点坐标数组形状: {cylinder_samples.shape}")

# 可视化采样结果
def plot_cylinder_samples(samples, center, radius, height, max_points=1000):
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
plot_cylinder_samples(cylinder_samples, cylinder_center, cylinder_radius, cylinder_height)

num = 0
for target in cylinder_samples[:1000000]:
    true_target = target
    roots_2 = find_roots(dis_func, a=t_explode, b=t_explode + 20, threshold=10, step=0.1)
    if len(roots_2) == 2:
        num += 1
        roots[0] = max(roots[0], roots_2[0])
        roots[1] = min(roots[1], roots_2[1])

print(num)
print(roots)
print(roots[1] - roots[0])

