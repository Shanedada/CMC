import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 初始数据 (x, y, z)
M1_pos = np.array([20000.0, 0.0, 2000.0])   # (x,z) 导弹位置
false_target = np.array([0.0, 0.0, 0.0])  # 假目标
true_target = np.array([0.0, 200.0, 0.0])  # 真目标

v_M = 300.0  # m/s

theta, v_FY, t_release, t_explode, FY1_pos = None, None, None, None, None

Bomb_pos = None

g = 9.8  # 重力加速度
sin_alpha = 1 / np.sqrt(100 + 1) # 导弹运动方向与x轴负方向的夹角
cos_alpha = 10 / np.sqrt(100 + 1)

# 导弹运动方程计算 M1_pos -> M_pos
def missile_trajectory(t):
    # 从 M1_pos 到 target 的直线运动
    M_pos = M1_pos - v_M * t * np.array([cos_alpha, 0, sin_alpha])
    return M_pos

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

def find_roots(f, a, b, threshold=10, step=0.01, tol=1e-8):
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


# 角度，速度，释放时间长度，延迟引爆时间，无人机初始位置    
                                        
param1 = [0.112785, 140.000000, 0.000000, 0.639469, np.array([17800.0, 0.0, 1800.0])]
param2 = [4.0165, 140.000000, 4.592267, 7.884428,  np.array([12000.0 ,1400.0 ,1400.0])]
param3 = [1.774928, 121.359148, 20.502107, 5.638041, np.array([6000, -3000, 700])]
params = [param1, param2, param3]
params_num = len(params)

segments = []
for param in params:
    theta, v_FY, t_release, t_explode, FY1_pos = param
    t_explode = t_explode + t_release
    print("="*100)
    print("theta:", theta, "v_FY:", v_FY, "t_release:", t_release, "t_explode:", t_explode)
    
    # ========  0 - t_release 无人机平飞 =========
    # 无人机飞行方向向量（与x轴负方向成theta角度）
    flight_direction = np.array([np.cos(theta), np.sin(theta), 0])

    FY1_pos = FY1_pos + v_FY * t_release * flight_direction

    print("无人机投放时位置：", FY1_pos)
    # ========  t_release - t_explode 干扰弹平抛运动 ======== 
    Bomb_pos = FY1_pos

    Bomb_pos = Bomb_pos + v_FY * (t_explode - t_release) * flight_direction - 0.5 * g * (t_explode - t_release) ** 2 * np.array([0, 0, 1])

    print("干扰弹起爆时位置：", Bomb_pos)

    roots = find_roots(dis_func, a=t_explode - 1, b=t_explode + 20, threshold=10, step=0.01)
    if(len(roots) > 0):
        roots[0] = max(roots[0], t_explode)
    print("交点:", roots)
    if len(roots) == 2:
        print("有效遮蔽时长 =", roots[1] - roots[0], "秒")
        segments.append([roots[0], roots[1]])
    else:
        segments.append([0, 0])

# 创建美观的遮蔽区间可视化
def plot_cover_segments(segments, params):
    """绘制美观的遮蔽区间图"""
    params_num = len(params)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12 + params_num * 2, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 上图：时间轴上的遮蔽区间
    ax1.set_title('遮蔽弹遮蔽区间时间轴', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('遮蔽状态', fontsize=12)
    
    # 设置y轴范围
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['', '遮蔽区间', ''])
    
    # 绘制每个遮蔽区间
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']
    for i, segment in enumerate(segments):
        if segment and segment[1] > segment[0]:  # 如果有有效区间且不是[0,0]
            color = colors[i % len(colors)]
            # 绘制区间线段
            ax1.plot([segment[0], segment[1]], [1, 1], 
                    color=color, linewidth=8, solid_capstyle='round', 
                    label=f'遮蔽弹{i+1}')
            # 添加时间标签
            ax1.text(segment[0], 1.2, f'{segment[0]:.2f}s', 
                    ha='center', va='bottom', fontsize=10, color=color)
            ax1.text(segment[1], 1.2, f'{segment[1]:.2f}s', 
                    ha='center', va='bottom', fontsize=10, color=color)
            # 添加持续时间标签
            duration = segment[1] - segment[0]
            mid_time = (segment[0] + segment[1]) / 2
            ax1.text(mid_time, 0.7, f'{duration:.2f}s', 
                    ha='center', va='center', fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # 设置网格和样式
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 下图：参数对比表
    ax2.axis('off')
    ax2.set_title('参数对比表', fontsize=16, fontweight='bold', pad=20)
    
    # 创建表格数据
    table_data = []
    headers = ['参数'] + [f'遮蔽弹{i+1}' for i in range(params_num)]
    
    param_names = ['航向角θ (rad)', '速度v (m/s)', '投放时间 (s)', '延迟时间 (s)']
    for i, name in enumerate(param_names):
        row = [name]
        for param in params:
            if i < 4:  # 前4个参数
                row.append(f'{param[i]:.4f}')
            else:  # 位置参数
                pos = param[4]
                row.append(f'({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')
        table_data.append(row)
    
    # 添加遮蔽时长行
    cover_times = []
    for i, segment in enumerate(segments):
        if segment and segment[1] > segment[0]:  # 有效遮蔽区间
            cover_times.append(f'{segment[1] - segment[0]:.3f}s')
        else:  # 无遮蔽
            cover_times.append('无遮蔽')
    
    table_data.append(['遮蔽时长'] + cover_times)
    
    # 计算列宽
    col_widths = [0.2] + [0.8 / params_num] * params_num
    
    # 绘制表格
    table = ax2.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=col_widths)
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加总标题
    fig.suptitle(f'遮蔽弹仿真结果可视化 ({params_num}枚弹)', fontsize=18, fontweight='bold', y=0.95)
    
    plt.show()

# 调用改进的绘制函数
plot_cover_segments(segments, params)

