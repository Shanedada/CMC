import numpy as np

# -----------------------------
# 常量和参数
# -----------------------------
g = 9.81             # m/s^2
R_cloud = 10.0       # m
cloud_sinking = 3.0  # m/s
T_cloud = 20.0       # s (有效时长)
dt = 0.1             # s

# 场景参数
P_M0 = np.array([20000.0, 0.0, 2000.0])   # 导弹初始
P_fake = np.array([0.0, 0.0, 0.0])        # 假目标
P_target = np.array([0.0, 200.0, 0.0])    # 真目标
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])# 无人机初始(等高飞)

# -----------------------------
# 导弹速度向量（指向假目标）
# -----------------------------
def compute_missile_velocity(P0, P_fake, speed):
    u = P_fake - P0
    u = u / np.linalg.norm(u)
    return speed * u

v_M = compute_missile_velocity(P_M0, P_fake, 300.0)

def missile_position(t):
    return P_M0 + v_M * t

# -----------------------------
# 起爆后云团轨迹（修正：抛体→起爆点，再竖直下沉）
# -----------------------------
def smoke_trajectory(theta, v_UAV, t_drop, t_explode):
    """
    theta: 水平航向角
    v_UAV: 无人机速度 [70,140]
    t_drop: 投放时刻 >=0
    t_explode: 投放后引信延迟 >0
    返回: cloud_position(t_cloud)  (t_cloud>=0 表示起爆后的相对时间)
    """
    # 无人机等高直线飞行（投放前）
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])
    P_drop = P_FY1_0 + v_UAV * t_drop * dir_xy  # 投放点（保持初始高度）

    # ==== 抛体运动到起爆 ====
    tau = t_explode
    # 投放瞬间速度 = 无人机速度（水平），竖直速度=0（等高投放）
    V_drop = v_UAV * dir_xy + np.array([0.0, 0.0, 0.0])
    # 起爆点（只受重力的抛体）
    P_expl = P_drop + V_drop * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

    # ==== 起爆后：仅竖直匀速下沉 ====
    def cloud_position(t_cloud):
        # 仅在 [0, T_cloud] 内有效
        return P_expl + np.array([0.0, 0.0, -cloud_sinking * t_cloud])

    return cloud_position

# -----------------------------
# 线段-点距离（裁剪 λ）
# -----------------------------
def seg_point_distance(A, B, P):
    u = B - A
    uu = np.dot(u, u)
    if uu < 1e-12:
        return np.linalg.norm(P - A)
    lam = np.dot(u, P - A) / uu
    lam = np.clip(lam, 0.0, 1.0)
    closest = A + lam * u
    return np.linalg.norm(P - closest)

def is_blocked(P_cloud, P_missile, P_target):
    d = seg_point_distance(P_missile, P_target, P_cloud)
    return d <= R_cloud

# -----------------------------
# 计算遮蔽时间（修正：起爆后 20s 窗口）
# -----------------------------
def compute_cover_time(theta, v_UAV, t_drop, t_explode):
    cloud_pos = smoke_trajectory(theta, v_UAV, t_drop, t_explode)

    # 模拟时间上限 = 导弹到达假目标的时间
    t_end = np.linalg.norm(P_fake - P_M0) / np.linalg.norm(v_M)

    t = 0.0
    cover_time = 0.0
    while t <= t_end:
        Pm = missile_position(t)

        # 起爆后的相对时间
        t_cloud = t - (t_drop + t_explode)
        if 0.0 <= t_cloud <= T_cloud:  # 仅有效窗口内计算
            Pc = cloud_pos(t_cloud)
            if is_blocked(Pc, Pm, P_target):
                cover_time += dt
        t += dt
    return cover_time

# -----------------------------
# 模拟退火（参数不变）
# -----------------------------
def simulated_annealing(max_iter=3000):
    theta = np.random.uniform(0, 2*np.pi)
    v_UAV = np.random.uniform(70, 140)
    t_drop = np.random.uniform(0.0, 12.0)
    t_explode = np.random.uniform(1.0, 6.0)

    x = np.array([theta, v_UAV, t_drop, t_explode])
    best_x = x.copy()
    best_cover = compute_cover_time(*x)
    T = 1.0
    alpha = 0.995

    for _ in range(max_iter):
        x_new = x + np.random.normal(0, [0.15, 6.0, 0.6, 0.4])
        x_new[0] = x_new[0] % (2*np.pi)
        x_new[1] = np.clip(x_new[1], 70, 140)
        x_new[2] = max(0.0, x_new[2])
        x_new[3] = max(0.1, x_new[3])

        cover_new = compute_cover_time(*x_new)
        delta = cover_new - best_cover
        if (delta >= 0) or (np.exp(delta / max(T,1e-6)) > np.random.rand()):
            x = x_new
            if cover_new > best_cover:
                best_x, best_cover = x_new.copy(), cover_new
        T *= alpha

    return best_x, best_cover

if __name__ == "__main__":
    best_params, best_time = simulated_annealing()
    print("最佳参数(theta, v_UAV, t_drop, t_explode):", best_params)
    print("最大遮蔽时间:", best_time, "秒")
