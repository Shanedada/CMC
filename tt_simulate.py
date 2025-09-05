import numpy as np
import pandas as pd

# -----------------------------
# 常量和参数
# -----------------------------
g = 9.8             # m/s^2
R_cloud = 10.0      # 烟幕有效半径 m
cloud_sinking = 3.0 # 云团下沉速度 m/s
T_cloud = 20.0      # 烟幕有效时长 s
dt = 0.05           # 时间步长 s

# 场景参数
P_M0 = np.array([20000.0, 0.0, 2000.0])
P_fake = np.array([0.0, 0.0, 0.0])
P_target = np.array([0.0, 200.0, 0.0])
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])

# -----------------------------
# 导弹速度向量
# -----------------------------
def compute_missile_velocity(P0, P_fake, speed):
    u = P_fake - P0
    u /= np.linalg.norm(u)
    return speed * u

v_M = compute_missile_velocity(P_M0, P_fake, 300.0)

# -----------------------------
# 烟幕轨迹
# -----------------------------
def smoke_trajectory(theta, v_UAV, t_drop, t_explode):
    dir_xy = np.array([np.cos(theta), np.sin(theta), 0.0])
    P_drop = P_FY1_0 + v_UAV * t_drop * dir_xy
    V_drop = v_UAV * dir_xy
    P_expl = P_drop + V_drop * t_explode + np.array([0.0, 0.0, -0.5*g*t_explode**2])
    return lambda t_cloud: P_expl + np.array([0.0, 0.0, -cloud_sinking*t_cloud])

# -----------------------------
# 点到线段距离
# -----------------------------
def seg_point_distance(A, B, P):
    u = B - A
    uu = np.dot(u, u)
    if uu < 1e-12:
        return np.linalg.norm(P - A)
    lam = np.clip(np.dot(u, P - A)/uu, 0.0, 1.0)
    closest = A + lam*u
    return np.linalg.norm(P - closest)

def is_blocked(P_cloud, P_missile, P_target):
    return seg_point_distance(P_missile, P_target, P_cloud) <= R_cloud

# -----------------------------
# 提取时间遮盖区间
# -----------------------------
def extract_cover_intervals(covered_flags, time_steps):
    """从布尔数组提取连续的时间遮盖区间"""
    intervals = []
    if not np.any(covered_flags):
        return intervals
    
    # 找到所有True的起始和结束位置
    diff = np.diff(np.concatenate(([False], covered_flags, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        intervals.append([time_steps[start], time_steps[end]])
    
    return intervals

# -----------------------------
# 多枚烟幕遮蔽时间计算（循环 + 向量化）
# -----------------------------
def compute_cover_time_multi(params_list):
    t_end = np.linalg.norm(P_fake - P_M0)/np.linalg.norm(v_M)
    time_steps = np.arange(0, t_end+dt, dt)
    n_clouds = len(params_list)

    covered_total = np.zeros(len(time_steps), dtype=bool)
    covered_each = np.zeros((n_clouds, len(time_steps)), dtype=bool)

    # --- 计算每个时间步的导弹位置 ---
    missile_positions = np.zeros((len(time_steps), 3))
    for i, t in enumerate(time_steps):
        missile_positions[i] = P_M0 + v_M * t

    # --- 遍历每枚烟幕 ---
    for idx, p in enumerate(params_list):
        theta, v_UAV, t_drop, t_explode = p
        cloud_pos_fn = smoke_trajectory(theta, v_UAV, t_drop, t_explode)

        # 烟幕相对时间
        t_cloud = time_steps - (t_drop + t_explode)
        valid_idx = (t_cloud >= 0) & (t_cloud <= T_cloud)
        if not np.any(valid_idx):
            continue

        # 计算烟幕位置
        Pc = np.array([cloud_pos_fn(tc) for tc in t_cloud[valid_idx]])

        # 判断遮蔽
        for i, pos in zip(np.where(valid_idx)[0], Pc):
            if is_blocked(pos, missile_positions[i], P_target):
                covered_total[i] = True
                covered_each[idx,i] = True

    total_cover_time = np.sum(covered_total) * dt
    each_cover_time = np.sum(covered_each, axis=1) * dt
    return total_cover_time, each_cover_time, time_steps, covered_each

# -----------------------------
# 打印每个烟幕的时间遮盖区间
# -----------------------------
def print_cover_intervals(params_list, covered_each, time_steps):
    """打印每个烟幕的详细时间遮盖区间信息"""
    print("\n" + "="*80)
    print("每个烟幕的时间遮盖区间详情")
    print("="*80)
    
    for idx, p in enumerate(params_list):
        theta, v_UAV, t_drop, t_explode = p
        print(f"\n烟幕 {idx+1}:")
        print(f"  参数: θ={theta:.4f} rad, v_UAV={v_UAV:.2f} m/s, t_drop={t_drop:.4f} s, t_explode={t_explode:.4f} s")
        print(f"  起爆时间: {t_drop + t_explode:.4f} s")
        
        # 提取该烟幕的遮盖区间
        intervals = extract_cover_intervals(covered_each[idx], time_steps)
        
        if intervals:
            print(f"  遮盖区间数量: {len(intervals)}")
            total_duration = 0
            for i, interval in enumerate(intervals):
                start_time, end_time = interval
                duration = end_time - start_time
                total_duration += duration
                print(f"    区间 {i+1}: [{start_time:.3f}s, {end_time:.3f}s] 持续 {duration:.3f}s")
            print(f"  总遮盖时长: {total_duration:.3f}s")
        else:
            print("  无有效遮盖区间")
    
    # 计算并打印并集遮盖区间
    print(f"\n并集遮盖区间:")
    union_intervals = extract_cover_intervals(np.any(covered_each, axis=0), time_steps)
    if union_intervals:
        print(f"  并集区间数量: {len(union_intervals)}")
        total_union_duration = 0
        for i, interval in enumerate(union_intervals):
            start_time, end_time = interval
            duration = end_time - start_time
            total_union_duration += duration
            print(f"    区间 {i+1}: [{start_time:.3f}s, {end_time:.3f}s] 持续 {duration:.3f}s")
        print(f"  并集总时长: {total_union_duration:.3f}s")
    else:
        print("  无有效并集遮盖区间")
    
    print("="*80)

# -----------------------------
# 保存结果到 Excel
# -----------------------------
def save_to_excel(params_list, total_time, each_times, filename="result.xlsx"):
    df = pd.DataFrame(params_list, columns=["theta(rad)","v_UAV(m/s)","t_drop(s)","t_explode(s)"])
    df["独立遮蔽时间(s)"] = each_times
    # 添加总有效遮蔽时间
    df.loc[len(df)] = [np.nan]*4 + [total_time]
    df.to_excel(filename, index=False)
    print(f"结果已保存到 {filename}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    smoke_params = np.array([
        [3.1329, 126.446, 3.87537403, 5.061331],
        [3.1329, 126.446, 0.62864592, 3.8478435],
        [3.1329, 126.446, 9.27616791, 3.96646362]
    ])

    total_time, each_times, time_steps, covered_each = compute_cover_time_multi(smoke_params)

    print("三枚烟幕干扰弹参数(theta, v_UAV, t_drop, t_explode):")
    for i, p in enumerate(smoke_params):
        print(f"弹{i+1}: {p}, 独立遮蔽时间: {each_times[i]:.2f} 秒")
    print("总有效遮蔽时间:", total_time, "秒")

    # 打印每个烟幕的详细时间遮盖区间
    print_cover_intervals(smoke_params, covered_each, time_steps)

    save_to_excel(smoke_params, total_time, each_times)
