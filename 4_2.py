import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.optimize import linprog

# ==========================================
# 1. 核心参数配置
# ==========================================
M_TOTAL = 1.0e8           # 总任务: 1亿吨
CAP_R = 1.5e5             # 火箭运力: 15万吨/年
CAP_E = 37451.0           # 电梯运力: 37,451 吨/年

# 成本参数 (USD)
COST_R_YEAR = 1.55e12     # 火箭成本
COST_E_YEAR = 2.848e9     # 电梯成本

# --- 污染参数 (5996 * 10^7 吨 CO2) ---
TOTAL_ROCKET_POLLUTION_SCENARIO = 5996 * 10**7 
POLL_R = TOTAL_ROCKET_POLLUTION_SCENARIO / M_TOTAL  # 约 599.6
POLL_E = 0                

# 字体设置 (大字体)
LABEL_SIZE = 16   
TICK_SIZE = 13    
LABEL_PAD = 18    

# ==========================================
# 2. 求解器
# ==========================================
def solve_scenario(limit_time, limit_budget):
    c = [POLL_R, POLL_E, 0]
    A_ub = [
        [1, 0, -CAP_R],               
        [0, 1, -CAP_E],               
        [0, 0, 1],                    
        [COST_R_YEAR/CAP_R, COST_E_YEAR/CAP_E, 0] 
    ]
    b_ub = [0, 0, limit_time, limit_budget]
    A_eq = [[1, 1, 0]]
    b_eq = [M_TOTAL]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    
    if res.success:
        return res.fun / 1e9 
    else:
        return 0.0

def polygon_under_graph(x, y):
    y = np.nan_to_num(y)
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

# ==========================================
# 3. 绘图并保存 PDF - 图表 1
# ==========================================
def plot_and_save_figure_1():
    print("正在生成图表 1 (Time)...")
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    budget_levels = [10e12, 50e12, 200e12, 500e12, 800e12, 1050e12]
    budget_labels = [f"${int(b/1e12)}T" for b in budget_levels]
    t_vals = np.linspace(600, 3000, 80)
    
    verts = []
    colors = []
    cmap = plt.get_cmap('viridis') 
    
    for i, b_lim in enumerate(budget_levels):
        z_vals = []
        for t in t_vals:
            z_vals.append(solve_scenario(t, b_lim))
        z_vals = np.array(z_vals)
        verts.append(polygon_under_graph(t_vals, z_vals))
        ax.plot(t_vals, z_vals, zs=i, zdir='y', color='k', alpha=0.3, linewidth=1)
        colors.append(cmap(i / len(budget_levels)))

    poly = PolyCollection(verts, facecolors=colors, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax.add_collection3d(poly, zs=range(len(budget_levels)), zdir='y')

    ax.set_xlabel('Time Limit (Years)', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_xlim(600, 3000)
    
    ax.set_ylabel('Budget Constraint', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_yticks(range(len(budget_levels)))
    ax.set_yticklabels(budget_labels, rotation=-15, va='center', ha='left', fontsize=TICK_SIZE)
    
    ax.set_zlabel('Pollution (Gt CO2)', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_zlim(0, 70) 
    
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='z', labelsize=TICK_SIZE)
    
    ax.view_init(elev=30, azim=-60)
    
    # 保存为 PDF
    filename = "Chart1_Pollution_vs_Time.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"已保存: {filename}")

# ==========================================
# 4. 绘图并保存 PDF - 图表 2
# ==========================================
def plot_and_save_figure_2():
    print("正在生成图表 2 (Budget)...")
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    time_levels = [700, 800, 1000, 1500, 2000, 2800]
    time_labels = [f"{t}y" for t in time_levels]
    b_vals = np.linspace(10e12, 1100e12, 60)
    
    verts = []
    colors = []
    cmap = plt.get_cmap('magma_r')
    
    for i, t_lim in enumerate(time_levels):
        z_vals = []
        for b in b_vals:
            z_vals.append(solve_scenario(t_lim, b))
        z_vals = np.array(z_vals)
        verts.append(polygon_under_graph(b_vals/1e12, z_vals)) 
        ax.plot(b_vals/1e12, z_vals, zs=i, zdir='y', color='k', alpha=0.3, linewidth=1)
        colors.append(cmap(i / len(time_levels)))

    poly = PolyCollection(verts, facecolors=colors, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax.add_collection3d(poly, zs=range(len(time_levels)), zdir='y')

    ax.set_xlabel('Budget ($ Trillions)', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_xlim(0, 1100)
    
    ax.set_ylabel('Time Limit', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_yticks(range(len(time_levels)))
    ax.set_yticklabels(time_labels, rotation=-15, va='center', ha='left', fontsize=TICK_SIZE)
    
    ax.set_zlabel('Pollution (Gt CO2)', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_zlim(0, 70)

    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='z', labelsize=TICK_SIZE)
    
    ax.view_init(elev=30, azim=-50)

    # 保存为 PDF
    filename = "Chart2_Pollution_vs_Budget.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"已保存: {filename}")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    plot_and_save_figure_1()
    plot_and_save_figure_2()
    print("所有图表生成完毕，请查看当前目录下的 PDF 文件。")
    # 如果您也想在屏幕上看到结果，取消下面这行的注释
    # plt.show()
