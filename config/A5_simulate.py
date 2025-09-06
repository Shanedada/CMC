import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------------
# 常量
# -------------------------
TRUE_TARGET = np.array([0.0, 200.0, 0.0])
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
MISSILE_SPEED = 300.0
G = 9.81
R_CLOUD = 10.0
CLOUD_SINKING = 3.0
T_CLOUD = 20.0

# -------------------------
# 工具函数
# -------------------------
def line_point_distance(A, B, P):
    """点P到线段AB的最短距离"""
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = np.clip(t, 0, 1)
    closest = A + t * AB
    return np.linalg.norm(P - closest)

def missile_pos(missile, t):
    init_pos = np.array(missile["init_pos"], dtype=float)
    target = FAKE_TARGET
    v = missile.get("v_M", MISSILE_SPEED)
    direction = target - init_pos
    dist = np.linalg.norm(direction)
    if dist == 0:
        return init_pos
    direction /= dist
    traveled = v * t
    if traveled >= dist:
        return target
    return init_pos + direction * traveled

def uav_pos(uav, t):
    init_pos = np.array(uav["init_pos"], dtype=float)
    v = uav["v_FY"]
    theta = uav["theta"]
    dx = v * np.cos(theta) * t
    dy = v * np.sin(theta) * t
    return init_pos + np.array([dx, dy, 0.0])

def bomb_pos(uav, bomb, t):
    """返回时刻t下炸弹的烟雾球心坐标，若无效则返回None"""
    t_release = bomb["t_release"]
    t_delay = bomb["t_delay"]

    # 未到引爆时间
    if t < t_release + t_delay:
        return None

    # 爆炸点：UAV在释放时刻的位置 + 延迟期间的位移（近似抛体）
    release_pos = uav_pos(uav, t_release)
    v = uav["v_FY"]
    theta = uav["theta"]
    flight_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    # 引爆点（包含水平位移+竖直下沉）
    expl_pos = release_pos + v * t_delay * flight_dir - np.array([0.0, 0.0, 0.5 * G * t_delay ** 2])

    # 引爆后烟雾存在时间
    t_since = t - (t_release + t_delay)
    if t_since > T_CLOUD:
        return None

    cloud_pos = expl_pos.copy()
    cloud_pos[2] -= CLOUD_SINKING * t_since
    return cloud_pos

# -------------------------
# 遮蔽区间计算
# -------------------------
def is_blocked(missile, t, uavs):
    """判断时刻t下导弹是否被任意烟雾遮蔽"""
    M = missile_pos(missile, t)
    T = TRUE_TARGET
    for fy in uavs:
        for bomb in fy["bombs"]:
            c = bomb_pos(fy, bomb, t)
            if c is None:
                continue
            d = line_point_distance(M, T, c)
            if d <= R_CLOUD:
                return True
    return False

def find_cover_intervals(f, a, b, threshold=True, step=0.01, min_len=0.02, gap_tol=0.03):
    """
    在区间[a,b]内采样，返回所有f(t)=True的时间区间
    参数:
        f: 判断函数 f(t)->bool
        a,b: 时间范围
        step: 采样步长
        min_len: 丢弃短于该值的区间
        gap_tol: 合并相邻区间的间隔阈值
    返回: [[t_start, t_end], ...]
    """
    ts = np.arange(a, b + step, step)
    vals = np.array([f(t) for t in ts])

    intervals = []
    in_seg = False
    seg_start = None

    for i in range(len(ts)):
        if vals[i] and not in_seg:
            seg_start = ts[i]
            in_seg = True
        elif not vals[i] and in_seg:
            seg_end = ts[i]
            intervals.append([seg_start, seg_end])
            in_seg = False
    if in_seg:
        intervals.append([seg_start, ts[-1]])

    # 合并间隙很小的区间
    merged = []
    for seg in intervals:
        if not merged:
            merged.append(seg)
        else:
            if seg[0] - merged[-1][1] < gap_tol:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)

    # 丢弃过短的区间
    merged = [seg for seg in merged if seg[1] - seg[0] >= min_len]
    return merged

def compute_cover_intervals(missile, uavs, T_max=30.0, step=0.01):
    """计算某颗导弹在[0, T_max]内的遮蔽区间"""
    f = lambda t: is_blocked(missile, t, uavs)
    return find_cover_intervals(f, 0, T_max, step=step)


# ------------------------- #
# 可视化函数
# ------------------------- #
def plot_cover_segments(segments, uavs, missile_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：单个导弹遮蔽区间
    ax1.set_title(f'导弹 {missile_id} 遮蔽区间时间轴', fontsize=16, fontweight='bold')
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('遮蔽状态', fontsize=12)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['', '遮蔽区间', ''])

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']

    for i, segment in enumerate(segments):
        if segment and segment[1] > segment[0]:
            color = colors[i % len(colors)]
            ax1.plot([segment[0], segment[1]], [1, 1],
                     color=color, linewidth=8, solid_capstyle='round',
                     label=f'遮蔽区间{i+1}')
            ax1.text(segment[0], 1.2, f'{segment[0]:.2f}s',
                     ha='center', va='bottom', fontsize=10, color=color)
            ax1.text(segment[1], 1.2, f'{segment[1]:.2f}s',
                     ha='center', va='bottom', fontsize=10, color=color)
            duration = segment[1] - segment[0]
            mid_time = (segment[0] + segment[1]) / 2
            ax1.text(mid_time, 0.7, f'{duration:.2f}s',
                     ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 下图：无人机参数对比表
    ax2.axis('off')
    ax2.set_title('无人机参数对比表', fontsize=16, fontweight='bold', pad=20)

    headers = ['参数'] + [u["id"] for u in uavs]
    param_names = ['航向角θ (rad)', '速度v (m/s)', '初始位置 (x,y,z)']

    table_data = []
    for name in param_names:
        row = [name]
        for u in uavs:
            if name.startswith("航向角"):
                row.append(f'{u.get("theta", 0.0):.4f}')
            elif name.startswith("速度"):
                row.append(f'{u.get("v_FY", 0.0):.1f}')
            else:  # 位置
                pos = u["init_pos"]
                row.append(f'({pos[0]}, {pos[1]}, {pos[2]})')
        table_data.append(row)

    table = ax2.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.2] + [0.8 / len(uavs)] * len(uavs)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else '#FFFFFF')

    plt.tight_layout()
    plt.show()


def plot_all_missiles_segments(all_segments, missiles, T_max):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('所有导弹遮蔽区间对比总览', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('导弹编号', fontsize=12)

    # 每个导弹一条线
    y_ticks = []
    y_labels = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']

    for i, missile in enumerate(missiles):
        y = i + 1
        y_ticks.append(y)
        y_labels.append(missile["id"])
        segments = all_segments.get(missile["id"], [])
        for j, (start, end) in enumerate(segments):
            if end > start:
                color = colors[j % len(colors)]
                ax.plot([start, end], [y, y],
                        color=color, linewidth=8, solid_capstyle='round')
                duration = end - start
                mid_time = (start + end) / 2
                ax.text(mid_time, y + 0.15, f'{duration:.2f}s',
                        ha='center', va='bottom', fontsize=10, color=color)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, T_max)
    ax.set_ylim(0, len(missiles) + 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "a.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    missiles = config["missiles"]
    uavs = config["uavs"]

    # 2. 模拟，得到所有导弹的遮蔽区间
    all_segments = {}
    T_max = 15.0  # 仿真时间
    for missile in missiles:
        missile_id = missile["id"]
        segments = compute_cover_intervals(missile, uavs, T_max=30.0, step=0.01)
        all_segments[missile_id] = segments

        # 单个导弹绘图
        plot_cover_segments(segments, uavs, missile_id)

    # 3. 总览图
    plot_all_missiles_segments(all_segments, missiles, T_max)
