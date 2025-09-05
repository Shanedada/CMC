import numpy as np

# --- 常量与参数定义 ---
# 物理常量
g = 9.81              # 重力加速度, m/s^2

# 烟雾云团参数
R_cloud = 10.0        # 烟雾云团半径, m
cloud_sinking = 3.0   # 烟雾云团下沉速度, m/s
T_cloud = 20.0        # 烟雾云团有效持续时间, s

# 仿真参数
dt = 0.1              # 仿真时间步长, s (更小的值可提高精度)

# 目标位置
P_M0 = np.array([20000.0, 0.0, 2000.0])      # 导弹初始位置
P_fake = np.array([0.0, 0.0, 0.0])           # 假目标位置
P_target = np.array([0.0, 200.0, 0.0])       # 真目标位置

# 无人机初始位置
UAV_INITIALS = [
    np.array([17800.0, 0.0, 1800.0]),   # FY1
    np.array([12000.0, 1400.0, 1400.0]),  # FY2
    np.array([6000.0, -3000.0, 700.0])    # FY3
]

# PSO 参数 (已优化，速度更快)
PSO_PARAMS = {
    "num_particles": 30,  # 减少粒子数以加速
    "max_iter": 200,      # 减少迭代次数以加速
    "w": 0.7,             # 惯性权重
    "c1": 1.5,            # 认知系数
    "c2": 1.5             # 社会系数
}

# --- 物理模型函数 ---

def compute_missile_velocity(P0, P_fake, speed):
    """
    计算并返回导弹的速度向量。
    导弹沿直线飞向假目标，速度固定。
    """
    u = P_fake - P0
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        return np.zeros_like(u)
    return speed * (u / norm)

# 导弹速度向量，假设固定速度为 300 m/s
v_M = compute_missile_velocity(P_M0, P_fake, 300.0)

def missile_position(t):
    """根据时间 t 计算导弹的位置。"""
    return P_M0 + v_M * t

def get_smoke_trajectory(theta, v_UAV, t_drop, t_explode, P_UAV0):
    """
    计算并返回烟雾云团的轨迹函数。
    
    参数:
    - theta: 无人机航向角 (弧度)
    - v_UAV: 无人机速度 (m/s)
    - t_drop: 烟雾弹投放时间 (s)
    - t_explode: 引信延迟时间 (s)
    - P_UAV0: 无人机初始位置
    
    返回:
    - cloud_position_func: 一个接受参数 t_cloud (从起爆开始的时间) 的函数，返回云团位置。
    - P_expl: 烟雾弹的起爆点坐标。
    """
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    # 投放点：无人机在投放时刻的位置
    P_drop = P_UAV0 + v_UAV * t_drop * dir_xy
    
    # 抛体运动时间
    tau = t_explode
    V_drop = v_UAV * dir_xy
    
    # 起爆点：考虑抛体运动和重力
    P_expl = P_drop + V_drop * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

    def cloud_position(t_cloud):
        """t_cloud 是从起爆开始计时的时间 (>=0)"""
        return P_expl + np.array([0.0, 0.0, -cloud_sinking * t_cloud])

    return cloud_position, P_expl

def seg_point_distance(A, B, P):
    """
    计算点 P 到线段 AB 的最短距离。
    
    参数:
    - A: 线段起点
    - B: 线段终点
    - P: 目标点
    """
    u = B - A
    uu = np.dot(u, u)
    if uu < 1e-12:
        return np.linalg.norm(P - A)
    lam = np.dot(u, P - A) / uu
    lam = np.clip(lam, 0.0, 1.0)
    closest = A + lam * u
    return np.linalg.norm(P - closest)

def is_blocked(P_cloud, P_missile, P_target):
    """判断云团是否遮蔽了导弹到目标的视线。"""
    d = seg_point_distance(P_missile, P_target, P_cloud)
    return d <= R_cloud

def extract_intervals(times, flags):
    """
    从布尔序列中提取连续 True 的时间区间。
    """
    intervals = []
    in_block = False
    start = None
    for i in range(len(times)):
        if flags[i] and not in_block:
            in_block = True
            start = times[i]
        elif not flags[i] and in_block:
            in_block = False
            intervals.append((start, times[i]))
    if in_block:
        intervals.append((start, times[-1]))
    return intervals

# --- 主计算函数 ---

def compute_cover_time_multi(params, return_intervals=False):
    """
    根据三架无人机的参数，计算总的并集遮蔽时间。
    
    参数:
    - params: 12个参数的数组，格式为
      [theta1, v1, t_drop1, t_explode1,
       theta2, v2, t_drop2, t_explode2,
       theta3, v3, t_drop3, t_explode3]
    - return_intervals: 是否同时返回每架无人机和总的遮蔽时间区间。
    
    返回:
    - cover_time: 总的遮蔽时间 (秒)。
    - (可选) uav_intervals: 每架无人机的遮蔽时间区间列表。
    - (可选) union_intervals: 并集遮蔽时间区间列表。
    """
    assert len(params) == 12

    uavs = []
    for i in range(3):
        theta, v_uav, t_drop, t_explode = params[4*i : 4*i + 4]
        P0 = UAV_INITIALS[i]
        cloud_pos_func, P_expl = get_smoke_trajectory(theta, v_uav, t_drop, t_explode, P0)
        uavs.append({
            "cloud_func": cloud_pos_func,
            "t_expl_time": t_drop + t_explode
        })

    # 仿真时间范围：从导弹发射到到达假目标
    t_end = np.linalg.norm(P_fake - P_M0) / np.linalg.norm(v_M)
    times = np.arange(0.0, t_end + dt/2, dt)
    blocked_flags = np.zeros_like(times, dtype=bool)
    uav_flags_list = [np.zeros_like(times, dtype=bool) for _ in range(3)]

    for idx, t in enumerate(times):
        Pm = missile_position(t)
        for ui, u in enumerate(uavs):
            t_cloud = t - u["t_expl_time"]
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
    return cover_time

# --- 粒子群优化（PSO）---

def particle_swarm_optimization_multi(pso_params):
    """
    使用粒子群优化算法寻找最佳无人机参数，以最大化遮蔽时间。
    
    参数:
    - pso_params: 包含 num_particles, max_iter, w, c1, c2 的字典。
    
    返回:
    - g_best: 最优参数数组。
    - g_best_score: 对应的最大遮蔽时间。
    """
    num_particles = pso_params["num_particles"]
    max_iter = pso_params["max_iter"]
    w, c1, c2 = pso_params["w"], pso_params["c1"], pso_params["c2"]

    # 参数边界
    theta_bounds = (0.0, 2*np.pi)
    v_bounds = (70.0, 140.0)
    t_drop_bounds = (0.0, 30.0)
    t_explode_bounds = (0.1, 8.0)
    bounds_per_uav = [theta_bounds, v_bounds, t_drop_bounds, t_explode_bounds]
    dim = 4 * 3
    bounds = bounds_per_uav * 3

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

    for it in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (p_best[i] - particles[i]) +
                             c2 * r2 * (g_best - particles[i]))
            particles[i] += velocities[i]

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
            print(f"迭代 {it+1}/{max_iter} - 当前最优遮蔽时间: {g_best_score:.4f} s")

    return g_best, g_best_score

# --- 主程序入口 ---
if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以保证结果可复现

    print("--- 启动粒子群优化（PSO）... ---")
    best_params, best_time = particle_swarm_optimization_multi(PSO_PARAMS)

    # 打印优化结果
    print("\n=== 优化结果（3 架无人机）===")
    for i in range(3):
        theta = best_params[4*i + 0]
        v_uav = best_params[4*i + 1]
        t_drop = best_params[4*i + 2]
        t_explode = best_params[4*i + 3]
        print(f"--- 无人机 {i+1} ---")
        print(f"  航向角 θ: {np.degrees(theta):.4f}° ({theta:.6f} rad)")
        print(f"  无人机速度: {v_uav:.6f} m/s")
        print(f"  投放时间: {t_drop:.6f} s")
        print(f"  引信延迟: {t_explode:.6f} s")

    print(f"\n最大并集遮蔽时间（PSO 优化）：{best_time:.6f} 秒")

    # 验证并打印详细遮蔽时间区间
    verified_time, uav_intervals, union_intervals = compute_cover_time_multi(best_params, return_intervals=True)
    print(f"验证计算遮蔽时间: {verified_time:.6f} 秒")

    print("\n--- 详细遮蔽时间区间 ---")
    for i, intervals in enumerate(uav_intervals, 1):
        formatted = [f"({s:.3f}, {e:.3f})" for s, e in intervals]
        print(f"无人机{i} 遮蔽区间: {'无' if not formatted else ', '.join(formatted)}")

    formatted_union = [f"({s:.3f}, {e:.3f})" for s, e in union_intervals]
    print(f"总并集遮蔽区间: {'无' if not formatted_union else ', '.join(formatted_union)}")