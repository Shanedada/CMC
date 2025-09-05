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
P_FY1_0 = np.array([12000, 1400, 1400])# 无人机初始(等高飞)

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
# 起爆后云团轨迹（抛体+竖直下沉）
# -----------------------------
def smoke_trajectory(theta, v_UAV, t_drop, t_explode):
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])
    P_drop = P_FY1_0 + v_UAV * t_drop * dir_xy  # 投放点

    tau = t_explode
    V_drop = v_UAV * dir_xy
    P_expl = P_drop + V_drop * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

    def cloud_position(t_cloud):
        return P_expl + np.array([0.0, 0.0, -cloud_sinking * t_cloud])

    return cloud_position

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
# 计算遮蔽时间
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
        if 0.0 <= t_cloud <= T_cloud:
            Pc = cloud_pos(t_cloud)
            if is_blocked(Pc, Pm, P_target):
                cover_time += dt
        t += dt
    return cover_time

# -----------------------------
# 粒子群优化（PSO）
# -----------------------------
def particle_swarm_optimization(num_particles=30, max_iter=200, top_k=20):
    # 参数范围
    theta_bounds = (0, 2*np.pi)
    v_bounds = (70, 140)
    t_drop_bounds = (0.0, 12.0)
    t_explode_bounds = (0.1, 6.0)

    dim = 4
    bounds = [theta_bounds, v_bounds, t_drop_bounds, t_explode_bounds]

    # 初始化
    particles = np.zeros((num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    for i in range(dim):
        low, high = bounds[i]
        particles[:, i] = np.random.uniform(low, high, num_particles)
        velocities[:, i] = np.random.uniform(-(high-low), (high-low), num_particles)

    # 个体和全局最优
    p_best = particles.copy()
    p_best_scores = np.array([compute_cover_time(*p) for p in particles])
    g_best_idx = np.argmax(p_best_scores)
    g_best = p_best[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]

    # 参数
    w, c1, c2 = 0.7, 1.5, 1.5

    all_solutions = []  # 保存所有解

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (p_best[i] - particles[i])
                + c2 * r2 * (g_best - particles[i])
            )
            particles[i] += velocities[i]

            # 边界约束
            for d in range(dim):
                low, high = bounds[d]
                particles[i, d] = np.clip(particles[i, d], low, high)

            # 适应度
            score = compute_cover_time(*particles[i])

            # 保存解
            all_solutions.append((score, particles[i].copy()))

            # 更新个体最优
            if score > p_best_scores[i]:
                p_best[i], p_best_scores[i] = particles[i].copy(), score
                if score > g_best_score:
                    g_best, g_best_score = particles[i].copy(), score

    # 取 top_k
    all_solutions = sorted(all_solutions, key=lambda x: -x[0])
    top_solutions = all_solutions[:top_k]

    return g_best, g_best_score, top_solutions, bounds

# -----------------------------
# 局部扰动搜索
# -----------------------------
def local_search(top_solutions, bounds, num_samples=50):
    best_local = None
    best_score = -1.0

    for score, params in top_solutions:
        for _ in range(num_samples):
            candidate = params.copy()
            for d in range(len(params)):
                low, high = bounds[d]
                span = (high - low) * 0.05  # 局部扰动范围 = 5%
                candidate[d] += np.random.uniform(-span, span)
                candidate[d] = np.clip(candidate[d], low, high)

            new_score = compute_cover_time(*candidate)
            if new_score > best_score:
                best_local = candidate.copy()
                best_score = new_score

    return best_local, best_score

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    best_params, best_time, top20, bounds = particle_swarm_optimization()
    print("初始全局最优解:", best_params, "遮蔽时间:", best_time)
    
    print("\nTop20 解:")
    for rank, (score, params) in enumerate(top20, 1):
        print(f"#{rank}: 遮蔽时间={score:.2f}, 参数={params}")

    # 局部优化
    best_local, best_local_score = local_search(top20, bounds)
    print("\n局部优化后更优解:", best_local, "遮蔽时间:", best_local_score)
