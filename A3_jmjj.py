import numpy as np

# =========================
# 常量（题面一致）
# =========================
g = 9.81
R_cloud = 10.0
V_down = 3.0
T_cloud = 20.0
dt = 0.03  # 调试步长；收敛后自动用 0.02 复核
EPS = 1e-9

# 开关：是否打印调试信息（每轮/结果）
DEBUG_PRINT = True   # True 打印每弹贡献等

# 场景
P_fake   = np.array([0.0,   0.0,  0.0], dtype=float)
P_target = np.array([0.0, 200.0,  0.0], dtype=float)
P_M0     = np.array([20000.0, 0.0, 2000.0], dtype=float)
P_FY1_0  = np.array([17800.0, 0.0, 1800.0], dtype=float)

# ============== 导弹运动 ==============
def missile_velocity():
    u = P_fake - P_M0
    n = np.linalg.norm(u)
    return 300.0 * (u / n)

v_M = missile_velocity()

def missile_pos(t):
    """t 可标量或一维数组；返回 (3,) 或 (K,3)。"""
    t = np.asarray(t)
    if t.ndim == 0:
        return P_M0 + v_M * float(t)
    return P_M0 + np.outer(t, v_M)

# ============== 线段-点距离（向量化） =============
def seg_point_distance_vec(A, B, P):
    """
    A,B,P: (K,3)
    返回: (K,) 线段 AB 到点 P 的最短距离
    """
    u = B - A
    uu = np.sum(u * u, axis=1)
    uu = np.maximum(uu, 1e-12)
    lam = np.sum(u * (P - A), axis=1) / uu
    lam = np.clip(lam, 0.0, 1.0)
    closest = A + lam[:, None] * u
    return np.linalg.norm(P - closest, axis=1)

# ============== 几何早剪枝（可选） =================
def quick_reject_xy(P_expl, t_expl, t_grid_valid, margin=4.0):
    """
    粗判：若竖直线段投影到 xy 平面到视线段(Pm->P_target)的横向距离
    下界始终远大于 R_cloud+margin，则可早退。
    这里只取云团水平坐标恒为 (x_expl, y_expl) 的简化下界。
    """
    if t_grid_valid.size == 0:
        return True
    xy_c = P_expl[:2]  # (x,y)
    Pm = missile_pos(t_grid_valid)          # (K,3)
    u = P_target - Pm                       # (K,3)
    # 线段到平面点的横向距离下界：近似取忽略 z 分量的 2D 距离
    A = Pm[:, :2]                           # (K,2)
    B = P_target[None, :2]                  # (K,2)
    AB = B - A
    AB2 = np.sum(AB * AB, axis=1)
    AB2 = np.maximum(AB2, 1e-12)
    tproj = np.sum((xy_c - A) * AB, axis=1) / AB2
    tproj = np.clip(tproj, 0.0, 1.0)
    closest_xy = A + tproj[:, None] * AB
    d_xy = np.linalg.norm(closest_xy - xy_c[None, :], axis=1)
    return np.all(d_xy > (R_cloud + margin))

# ============== 单枚弹覆盖掩码 =====================
def cover_mask_one(theta, vUAV, t_drop, tau, t_grid):
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])

    # 投放点（等高）
    P_drop = P_FY1_0 + vUAV * t_drop * dir_xy         # (3,)
    V_drop = vUAV * dir_xy                            # 水平速度

    # 抛体到起爆
    P_expl = P_drop + V_drop * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau**2)
    t_expl = t_drop + tau

    mask = np.zeros_like(t_grid, dtype=bool)

    # 有效时窗
    valid = (t_grid >= t_expl) & (t_grid <= t_expl + T_cloud)
    if not np.any(valid):
        return mask

    t_valid = t_grid[valid]
    # 几何早剪枝（可选），提高速度；拒绝则直接返回空掩码
    if quick_reject_xy(P_expl, t_expl, t_valid):
        return mask

    # 起爆后云团中心（竖直下沉）
    t_cloud = t_valid - t_expl                      # (K,)
    P_cloud = P_expl + np.column_stack([
        np.zeros_like(t_cloud),
        np.zeros_like(t_cloud),
        -V_down * t_cloud
    ])                                              # (K,3)

    Pm = missile_pos(t_valid)                       # (K,3)
    Pt = np.repeat(P_target[None, :], Pm.shape[0], axis=0)
    dist = seg_point_distance_vec(Pm, Pt, P_cloud)
    mask[valid] = (dist <= R_cloud)
    return mask

# ============== 三枚弹并集“纯覆盖时长” =============
def cover_time_union(theta, vUAV, tds, taus, t_grid, return_parts=False):
    union_mask = np.zeros_like(t_grid, dtype=bool)
    parts = []
    for k in range(3):
        mk = cover_mask_one(theta, vUAV, float(tds[k]), float(taus[k]), t_grid)
        parts.append(mk)
        union_mask |= mk
    cover = union_mask.sum() * (t_grid[1] - t_grid[0])
    if return_parts:
        indiv = [m.sum() * (t_grid[1] - t_grid[0]) for m in parts]
        return cover, indiv
    return cover

# ============== 目标函数（覆盖 - 惩罚） =============
def make_objective(t_grid):
    def objective(x):
        """
        x = [theta, vUAV, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3]
        返回：PSO 比较用的“覆盖 - 惩罚”
        """
        theta = x[0] % (2*np.pi)
        vUAV  = float(np.clip(x[1], 70.0, 140.0))
        tds   = np.array([x[2], x[4], x[6]], dtype=float)
        taus  = np.array([x[3], x[5], x[7]], dtype=float)

        # 物理边界
        if np.any(tds < -EPS) or np.any(taus < 0.1 - EPS):
            return -1e9

        # 强制排序（升序）以减少“交换”带来的随机跳动
        perm = np.argsort(tds)
        tds = tds[perm]
        taus = taus[perm]

        # 硬间隔：相邻投放 >= 1 s
        if np.any(np.diff(tds) < 1.0 - 1e-9):
            return -1e9

        # 纯覆盖
        cover = cover_time_union(theta, vUAV, tds, taus, t_grid)

        # 软惩罚：轻微偏离边界的情况
        pen = 0.0
        pen += 1.0 * np.sum(np.maximum(0.0, 0.1 - taus))  # >=0.1
        # 这里不再重复对 tds 罚，因为已硬约束

        return cover - pen
    return objective

# ============== PSO 优化 ============================
def pso(max_iter=240, swarm_size=48, seed=1717):
    rng = np.random.default_rng(seed)

    t_end = np.linalg.norm(P_fake - P_M0) / np.linalg.norm(v_M)
    t_grid = np.arange(0.0, t_end + 1e-9, dt)
    objective = make_objective(t_grid)

    # 初始粒子
    def rand_particle():
        theta = rng.uniform(0, 2*np.pi)
        v     = rng.uniform(90, 135)       # 缩窄一点更稳，也可放宽到[70,140]
        tds   = np.sort(rng.uniform(0.0, 14.0, size=3))  # 先排好序
        taus  = rng.uniform(1.5, 5.5,  size=3)
        return np.array([theta, v, tds[0], taus[0], tds[1], taus[1], tds[2], taus[2]], dtype=float)

    X = np.array([rand_particle() for _ in range(swarm_size)], dtype=float)
    V = rng.normal(0, [0.25, 5.0, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4], size=(swarm_size, 8))

    # 个体/全局最优
    pbest = X.copy()
    pbest_val = np.array([objective(x) for x in X])
    g_idx = int(np.argmax(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    w, c1, c2 = 0.72, 1.7, 1.7
    step_lim = np.array([0.3, 6.0, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5], dtype=float)

    for it in range(max_iter):
        r1 = rng.random((swarm_size, 8))
        r2 = rng.random((swarm_size, 8))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -step_lim, step_lim)
        X = X + V

        # 修边界
        X[:, 0] = X[:, 0] % (2*np.pi)
        X[:, 1] = np.clip(X[:, 1], 70.0, 140.0)
        X[:, 2::2] = np.maximum(X[:, 2::2], 0.0)    # t_drop >= 0
        X[:, 3::2] = np.maximum(X[:, 3::2], 0.1)    # tau    >= 0.1

        # 评估
        vals = np.array([objective(x) for x in X])

        improve = vals > pbest_val
        pbest[improve] = X[improve]
        pbest_val[improve] = vals[improve]

        g_idx = int(np.argmax(pbest_val))
        if pbest_val[g_idx] > gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()

        if (it + 1) % 20 == 0:
            print(f"[PSO] iter={it+1:3d}, best(obj)={gbest_val:.4f}")

    # === 复核：细 dt 的纯覆盖（不含惩罚）===
    theta = gbest[0] % (2*np.pi)
    vUAV  = float(np.clip(gbest[1], 70.0, 140.0))
    tds   = np.array([gbest[2], gbest[4], gbest[6]], dtype=float)
    taus  = np.array([gbest[3], gbest[5], gbest[7]], dtype=float)

    # 强制排序，确保间隔逻辑一致
    perm = np.argsort(tds)
    tds = tds[perm]
    taus = taus[perm]

    dt_old = globals()['dt']
    globals()['dt'] = 0.02
    t_end = np.linalg.norm(P_fake - P_M0) / np.linalg.norm(v_M)
    t_grid2 = np.arange(0.0, t_end + 1e-9, dt)

    cover_union, parts = cover_time_union(theta, vUAV, tds, taus, t_grid2, return_parts=True)
    globals()['dt'] = dt_old

    if DEBUG_PRINT:
        print(f"[DEBUG] indiv covers: {parts}, union: {cover_union:.3f}s",
              f"fill ratio: {cover_union / max(1e-6, sum(parts)):.3f}")

    res = {
        "theta": float(theta),
        "v": float(vUAV),
        "t_drop": tds.astype(float),
        "tau": taus.astype(float),
        "union_cover_time_refined": float(cover_union),
        "indiv_covers": [float(x) for x in parts]
    }
    return res, gbest

# =========================
# 运行
# =========================
if __name__ == "__main__":
    res, raw = pso(max_iter=10000, swarm_size=56, seed=2025)
    print("\n=== 第三题·三弹并集最优（PSO·tuned） ===")
    print(f"航向 theta = {res['theta']:.4f} rad")
    print(f"速度 v     = {res['v']:.3f} m/s")
    print(f"t_drop     = {res['t_drop']}")
    print(f"tau        = {res['tau']}")
    print(f"各弹遮蔽:   {res['indiv_covers']}  (s)")
    print(f"并集遮蔽时长(细dt复核) ≈ {res['union_cover_time_refined']:.3f} s")
    
    print(f'第一个弹 : {res['theta']:.8f}, {res['v']:.8f}, {res['t_drop'][0]:.8f}, {res['tau'][0]:.8f}')
    print(f'第二个弹 : {res['theta']:.8f}, {res['v']:.8f}, {res['t_drop'][1]:.8f}, {res['tau'][1]:.8f}')
    print(f'第三个弹 : {res['theta']:.8f}, {res['v']:.8f}, {res['t_drop'][2]:.8f}, {res['tau'][2]:.8f}')