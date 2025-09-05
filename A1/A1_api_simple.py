import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from efficient_judger import min_distance_missile_cylinder_to_cloud, sample_cylinder_surface
import config

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从配置文件导入参数
M1_pos = config.MISSILE_POSITIONS['M1']
FY1_pos = config.DRONE_POSITIONS['FY1']
false_target = config.FALSE_TARGET
true_target = config.TRUE_TARGET_CENTER

v_M = config.MISSILE_SPEED
v_FY = 120.0  # 无人机速度，可以根据需要调整

t_release = config.T_RELEASE
t_explode = config.T_EXPLODE

g = config.GRAVITY
sin_theta = config.SIN_THETA
cos_theta = config.COS_THETA

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

# 使用API替换原来的距离计算
def dis_func(t):
    M_pos = missile_trajectory(t)
    S_pos = Bomb_cal_pos(t)
    
    # 从配置文件获取圆柱体参数
    cylinder_center = config.TRUE_TARGET_CENTER
    cylinder_radius = config.CYLINDER_RADIUS
    cylinder_height = config.CYLINDER_HEIGHT
    
    # 调用API计算最小距离
    return min_distance_missile_cylinder_to_cloud(
        M_pos=M_pos,
        S_pos=S_pos,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        num_samples=config.CYLINDER_SAMPLES
    )

# ===== 求解 =====
roots = find_roots(dis_func, a=t_explode, b=t_explode + config.TIME_RANGE, 
                  threshold=config.DISTANCE_THRESHOLD, step=config.ROOT_FINDING_STEP)
if len(roots) == 2:
    print(roots)
    print("有效遮蔽时长 =", roots[1] - roots[0], "秒")