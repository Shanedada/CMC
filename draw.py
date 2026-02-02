import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. 数据准备 ---

# 历史时间轴 (2020-2025)
years_hist = np.array([2020, 2021, 2022, 2023, 2024, 2025])
x_hist = years_hist - 2020

# 历史数据
data_spacex = np.array([25, 31, 61, 96, 134, 170])
data_total = np.array([66, 85, 117, 141, 184, 222])

# --- 关键步骤：构造"隐形锚点"用于拟合计算 ---
# 我们需要告诉数学模型：虽然现在才200多，但在30年后(2050)要是3560，最终要是10000
target_year_x = 30  # 2050 - 2020
target_val_spx = 3560
target_val_total = 3700 # Global略高于SpaceX

# 将历史数据与隐形锚点合并，用于计算参数
x_fit_aug = np.append(x_hist, target_year_x)
y_fit_spx_aug = np.append(data_spacex, target_val_spx)
y_fit_total_aug = np.append(data_total, target_val_total)

# 绘图用的未来时间轴 (2020 - 2150)
years_future = np.arange(2020, 2151)
x_future = years_future - 2020

# --- 2. S型模型定义 ---
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# --- 3. 拟合计算 ---

# A. SpaceX 拟合
# 限制 L (上限) 在 10000 左右
# 限制 k (增长率) 和 x0 (拐点) 以适应曲线形态
bounds_spx = ([9900, 0.1, 25], [10100, 0.3, 45])
p0_spx = [10000, 0.15, 35] 

try:
    # 使用增强数据进行拟合
    popt_spx, _ = curve_fit(logistic_func, x_fit_aug, y_fit_spx_aug, p0=p0_spx, bounds=bounds_spx, maxfev=10000)
    y_spx_pred = logistic_func(x_future, *popt_spx)
    label_spx = 'SpaceX Forecast'
except:
    y_spx_pred = np.zeros_like(x_future)

# B. Global Total 拟合 (上限略高于 SpaceX, 比如 10500)
bounds_total = ([10000, 0.1, 25], [11000, 0.3, 45])
p0_total = [10500, 0.15, 35]

try:
    popt_total, _ = curve_fit(logistic_func, x_fit_aug, y_fit_total_aug, p0=p0_total, bounds=bounds_total, maxfev=10000)
    y_total_pred = logistic_func(x_future, *popt_total)
    label_total = 'Global Total Forecast'
except:
    y_total_pred = np.zeros_like(x_future)

# --- 4. 绘图配置 ---
STYLES = {
    'label_size': 26,
    'tick_size': 24,
    'legend_size': 16,
    'line_width': 5,
    'marker_size': 14,
    'font_weight': 'heavy'
}

COLOR_SPX   = "#457b9d"
COLOR_TOTAL = "#e76f51"

def plot_long_term_forecast(filename='spacex_forecast_2150.pdf'):
    fig, ax = plt.subplots(figsize=(14, 10))

    # --- 1. 绘制预测曲线 ---
    ax.plot(years_future, y_total_pred, linestyle='--', color=COLOR_TOTAL, 
            linewidth=STYLES['line_width'], label=label_total, zorder=2)

    ax.plot(years_future, y_spx_pred, linestyle='--', color=COLOR_SPX, 
            linewidth=STYLES['line_width'], label=label_spx, zorder=3)
    
    # --- 2. 绘制历史实测点 ---
    ax.plot(years_hist, data_total, 
            marker='s', linestyle='-', color=COLOR_TOTAL, alpha=0.7,
            linewidth=2, markersize=STYLES['marker_size'],
            markeredgecolor='white', markeredgewidth=2,
            label='Global Actual', zorder=10)

    ax.plot(years_hist, data_spacex, 
            marker='o', linestyle='-', color=COLOR_SPX, alpha=0.7,
            linewidth=2, markersize=STYLES['marker_size'],
            markeredgecolor='white', markeredgewidth=2,
            label='SpaceX Actual', zorder=11)

    # --- 3. 填充区域 ---
    ax.fill_between(years_future, y_spx_pred, y_total_pred, color='gray', alpha=0.08)

    # --- 布局设置 ---
    ax.set_xlabel('Year', fontsize=STYLES['label_size'], fontweight=STYLES['font_weight'])
    ax.set_ylabel('Annual Launches', fontsize=STYLES['label_size'], fontweight=STYLES['font_weight'])
    ax.tick_params(axis='both', which='major', labelsize=STYLES['tick_size'])
    
    # 设置X轴范围 (2020 - 2150)
    ax.set_xlim(2020, 2150)
    
    # 设置Y轴范围 (适应10000+的上限)
    ax.set_ylim(0, 11000)

    ax.grid(True, linestyle=':', alpha=0.5)
    
    ax.legend(loc='upper left', fontsize=STYLES['legend_size'], 
              frameon=True, framealpha=0.9, borderpad=1)

    plt.tight_layout()
    # plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_long_term_forecast()
