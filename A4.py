import numpy as np

# -----------------------------
# 常量和参数
# -----------------------------
g = 9.81             # m/s^2
R_cloud = 10.0       # m
cloud_sinking = 3.0  # m/s
T_cloud = 20.0       # s (有效时长)
dt = 0.02            # 时间步长，更小提高精度

# 导弹与目标、诱饵、初始导弹位置
P_M0 = np.array([20000.0, 0.0, 2000.0])
P_fake = np.array([0.0, 0.0, 0.0])
P_target = np.array([0.0, 200.0, 0.0])

# 三架无人机初始位置：FY1, FY2, FY3
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
P_FY2_0 = np.array([12000.0, 1400.0, 1400.0])
P_FY3_0 = np.array([6000.0, -3000.0, 700.0])
UAV_INITIALS = [P_FY1_0, P_FY2_0, P_FY3_0]

# -----------------------------
# 导弹速度向量（指向假目标，固定速度）
# -----------------------------
def compute_missile_velocity(P0, P_fake, speed):
    u = P_fake - P0
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        return np.zeros_like(u)
    u = u / norm
    return speed * u

v_M = compute_missile_velocity(P_M0, P_fake, 300.0)

def missile_position(t):
    return P_M0 + v_M * t

# -----------------------------
# 单个无人机：投放点、爆炸点、云团位置函数
# -----------------------------
def smoke_trajectory(theta, v_UAV, t_drop, t_explode, P_UAV0):
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])
    # 投放点（投放时刻无人机位置）
    P_drop = P_UAV0 + v_UAV * t_drop * dir_xy
    tau = t_explode
    V_drop = v_UAV * dir_xy
    # 起爆点（抛体运动并受重力影响）
    P_expl = P_drop + V_drop * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

    def cloud_position(t_cloud):
        # t_cloud 是从起爆开始计时的时间（>=0）
        # 云竖直向下沉降
        return P_expl + np.array([0.0, 0.0, -cloud_sinking * t_cloud])

    return cloud_position, P_expl

# -----------------------------
# 线段-点距离
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
# 从布尔序列提取区间（闭开区间近似）
# -----------------------------
def extract_intervals(times, flags):
    intervals = []
    in_block = False
    start = None
    for i in range(len(times)):
        if flags[i] and not in_block:
            in_block = True
            start = times[i]
        elif not flags[i] and in_block:
            in_block = False
            # 结束时间取当前采样时间（近似）
            intervals.append((start, times[i]))
    if in_block:
        intervals.append((start, times[-1]))
    return intervals

# -----------------------------
# 计算给定三架无人机参数时的并集遮蔽时间
# 参数顺序（共12个）:
# [theta1, v1, t_drop1, t_explode1,
#  theta2, v2, t_drop2, t_explode2,
#  theta3, v3, t_drop3, t_explode3]
#
# 新增：可选返回每架 UAV 的区间和并集区间（return_intervals=True）
# -----------------------------
def compute_cover_time_multi(params, return_intervals=False):
    assert len(params) == 12
    params = np.asarray(params)
    uavs = []
    for i in range(3):
        theta = params[4*i + 0]
        v_uav = params[4*i + 1]
        t_drop = params[4*i + 2]
        t_explode = params[4*i + 3]
        P0 = UAV_INITIALS[i]
        cloud_pos_func, P_expl = smoke_trajectory(theta, v_uav, t_drop, t_explode, P0)
        uavs.append({
            "theta": theta, "v": v_uav, "t_drop": t_drop, "t_explode": t_explode,
            "P0": P0, "cloud_func": cloud_pos_func, "P_expl": P_expl,
            "t_expl_time": t_drop + t_explode
        })

    # 终止时间：导弹到假目标所需时间（沿既定速度）
    t_end = np.linalg.norm(P_fake - P_M0) / np.linalg.norm(v_M)
    times = np.arange(0.0, t_end + dt/2, dt)
    blocked_flags = np.zeros_like(times, dtype=bool)

    # 若需要返回每架 UAV 的布尔序列以提取区间
    uav_flags_list = [np.zeros_like(times, dtype=bool) for _ in range(3)]

    # 对每个时间步判断是否被任意云团遮挡
    for idx, t in enumerate(times):
        Pm = missile_position(t)
        for ui, u in enumerate(uavs):
            t_cloud = t - u["t_expl_time"]  # 从起爆开始计时
            if 0.0 <= t_cloud <= T_cloud:
                Pc = u["cloud_func"](t_cloud)
                if is_blocked(Pc, Pm, P_target):
                    blocked_flags[idx] = True
                    uav_flags_list[ui][idx] = True

    cover_time = blocked_flags.sum() * dt

    if return_intervals:
        uav_intervals = [extract_intervals(times, flags) for flags in uav_flags_list]
        union_intervals = extract_intervals(times, blocked_flags)
        return cover_time, uav_intervals, union_intervals
    else:
        return cover_time

# -----------------------------
# 粒子群优化（PSO）—— 扩展到 12 维
# （保持原样）
# -----------------------------
def particle_swarm_optimization_multi(num_particles=40, max_iter=300):
    theta_bounds = (0.0, 2*np.pi)    # 航向角
    v_bounds = (70.0, 140.0)         # 无人机速度范围（m/s）
    t_drop_bounds = (0.0, 30.0)      # 投放时间（s）
    t_explode_bounds = (0.1, 20.0)    # 引信延迟（s）

    bounds_per_uav = [theta_bounds, v_bounds, t_drop_bounds, t_explode_bounds]
    dim = 4 * 3  # 3架无人机
    bounds = bounds_per_uav * 3  # 重复三次

    particles = np.zeros((num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    for d in range(dim):
        low, high = bounds[d]
        particles[:, d] = np.random.uniform(low, high, num_particles)
        velocities[:, d] = np.random.uniform(-(high-low)/2, (high-low)/2, num_particles)

    p_best = particles.copy()
    p_best_scores = np.array([compute_cover_time_multi(p) for p in particles])
    g_best_idx = np.argmax(p_best_scores)
    g_best = p_best[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]

    w, c1, c2 = 0.7, 1.5, 1.5

    for it in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (p_best[i] - particles[i])
                + c2 * r2 * (g_best - particles[i])
            )
            particles[i] = particles[i] + velocities[i]

            for d in range(dim):
                low, high = bounds[d]
                particles[i, d] = np.clip(particles[i, d], low, high)

            score = compute_cover_time_multi(particles[i])
            if score > p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = score
                if score > g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = score

        if (it + 1) % 50 == 0:
            print(f"迭代 {it+1}/{max_iter} — 当前最优遮蔽时间: {g_best_score:.4f} s")

    return g_best, g_best_score

# -----------------------------
# 主程序（保留原始输出格式；新增：打印每架 UAV 的区间与并集）
# -----------------------------
if __name__ == "__main__":
    # np.random.seed(0)  # 可复现
    best_params, best_time = particle_swarm_optimization_multi(num_particles=40, max_iter=300)

    # 解包并打印每架无人机的参数（与原始格式一致）
    print("=== 优化结果（3 UAV）===")
    for i in range(3):
        theta = best_params[4*i + 0]
        v_uav = best_params[4*i + 1]
        t_drop = best_params[4*i + 2]
        t_explode = best_params[4*i + 3]
        print(f"UAV {i+1}:")
        print(f"  航向角 θ (rad): {theta:.6f}, θ (deg): {np.degrees(theta):.4f}°")
        print(f"  无人机速度 v_UAV: {v_uav:.6f} m/s")
        print(f"  投放时间 t_drop: {t_drop:.6f} s")
        print(f"  引信延迟 t_explode: {t_explode:.6f} s")
    print(f"最大并集遮蔽时间(PSO给出): {best_time:.6f} 秒")

    # 验证计算遮蔽时间（并同时获取每架 UAV 的区间与并集区间）
    verified_time, uav_intervals, union_intervals = compute_cover_time_multi(best_params, return_intervals=True)
    print(f"验证计算遮蔽时间: {verified_time:.6f} 秒")

    # --- 新增输出：每架 UAV 的遮蔽区间 与 并集区间 ---
    for i, intervals in enumerate(uav_intervals, 1):
        if len(intervals) == 0:
            print(f"UAV{i} 遮蔽区间: 无")
        else:
            # 每个区间以 (start, end) 打印，保留 3 位小数
            formatted = [f"({s:.3f}, {e:.3f})" for s, e in intervals]
            print(f"UAV{i} 遮蔽区间: {formatted}")
    if len(union_intervals) == 0:
        print("并集遮蔽区间: 无")
    else:
        formatted_union = [f"({s:.3f}, {e:.3f})" for s, e in union_intervals]
        print(f"并集遮蔽区间: {formatted_union}")
