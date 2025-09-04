import numpy as np
import matplotlib.pyplot as plt

# 初始数据 (x, y, z)
M1_pos = np.array([20000.0, 0.0, 2000.0])   # (x,z) 导弹位置
FY1_pos = np.array([17800.0, 0.0, 1800.0])  # (x,z) 无人机位置
false_target = np.array([0.0, 0.0, 0.0])  # 假目标
true_target = np.array([0.0, 200.0, 0.0])  # 真目标

v_M = 300.0  # m/s
v_FY = 120.0  # m/s

t_release = 1.5  # 投放时间
t_explode = t_release + 3.6  # 起爆时间

g = 9.8  # 重力加速度
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

# def dis(pos1, pos2):
#     return np.sqrt(np.sum((pos1 - pos2) ** 2))

# def distance_function(t):
#     '''
#     5.1s之后，导弹和干扰弹之间的距离d与时间的函数
#     t应该大于t_explode
#     '''
#     d = dis(missile_trajectory(t), Bomb_cal_pos(t))
#     return d

# for i in range(21):
#     t = i + t_explode
#     print(t, distance_function(t))

def line_dis(x, y, z):
    '''
    计算点z到直线 x,y之间的距离。
    '''
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    
    # 向量
    xy = y - x
    xz = z - x
    
    # 叉乘长度 / 直线方向长度
    d = np.linalg.norm(np.cross(xz, xy)) / np.linalg.norm(xy)
    return d

# for i in range(21):
#     t = i + t_explode
#     dis = line_dis(true_target, missile_trajectory(t), Bomb_cal_pos(t)) 
#     print(t, dis)

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
# dis_func(t) 是你定义的距离函数
# 例如：dis_func = lambda t: line_dis(true_target, missile_trajectory(t), Bomb_cal_pos(t))

roots = find_roots(dis_func, a=t_explode, b=t_explode + 20, threshold=10, step=0.1)
print("交点:", roots)
if len(roots) == 2:
    print("有效遮蔽时长 =", roots[1] - roots[0], "秒")





