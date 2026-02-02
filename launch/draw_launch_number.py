import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# ----------------------------
# 1. 历史数据 (2006-2025)
# ----------------------------
years_hist = np.arange(2006, 2026)
x_hist = years_hist - 2020  # 以2020为基准

data_spacex = np.array([1,0,1,0,2,2,2,3,6,6,8,18,21,13,25,31,61,96,134,170])
data_total = np.array([2,1,2,1,5,6,8,10,15,18,20,30,35,28,66,85,117,141,184,222])

# ----------------------------
# 2. 远期锚点 (2050 和 2150)
# ----------------------------
# 2050 中期目标，2150 上限接近10000
x_fit_aug = np.append(x_hist, [30, 130])  # 2050, 2150
y_fit_spx_aug = np.append(data_spacex, [1500, 10000])   # SpaceX
y_fit_total_aug = np.append(data_total, [1600, 10000])  # Global

# ----------------------------
# 3. 未来时间轴 (2006-2150)
# ----------------------------
years_future = np.arange(2006, 2151)
x_future = years_future - 2020

# ----------------------------
# 4. Logistic S型函数
# ----------------------------
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k*(x - x0)))

# ----------------------------
# 5. 拟合参数设置
# ----------------------------
# SpaceX
p0_spx = [10000, 0.03, 100]   # 初值
bounds_spx = ([9900, 0.01, 50], [10100, 0.1, 150])
popt_spx, _ = curve_fit(logistic_func, x_fit_aug, y_fit_spx_aug, p0=p0_spx, bounds=bounds_spx, maxfev=20000)
y_spx_pred = logistic_func(x_future, *popt_spx)

# Global
p0_total = [10000, 0.03, 100]
bounds_total = ([9900, 0.01, 50], [10100, 0.1, 150])
popt_total, _ = curve_fit(logistic_func, x_fit_aug, y_fit_total_aug, p0=p0_total, bounds=bounds_total, maxfev=20000)
y_total_pred = logistic_func(x_future, *popt_total)

# ----------------------------
# 6. 样式设置
# ----------------------------
STYLES = {'label_size':26,'tick_size':24,'legend_size':16,
          'line_width':5,'marker_size':14,'font_weight':'heavy'}
COLOR_SPX = "#457b9d"
COLOR_TOTAL = "#e76f51"

# ----------------------------
# 7. 绘图函数
# ----------------------------
def plot_long_term_forecast(filename=None):
    fig, ax = plt.subplots(figsize=(14,10))

    # 预测曲线
    ax.plot(years_future, y_total_pred, linestyle='--', color=COLOR_TOTAL,
            linewidth=STYLES['line_width'], label='Global Total Forecast', zorder=2)
    ax.plot(years_future, y_spx_pred, linestyle='--', color=COLOR_SPX,
            linewidth=STYLES['line_width'], label='SpaceX Forecast', zorder=3)

    # 历史点，早期 marker 小
    for i, year in enumerate(years_hist):
        ms = 6 if year <= 2015 else STYLES['marker_size']
        ax.plot(year, data_total[i], marker='s', linestyle='-', color=COLOR_TOTAL,
                alpha=0.7, linewidth=2, markersize=ms, markeredgecolor='white',
                markeredgewidth=2, zorder=10 if year>2015 else 5)
        ax.plot(year, data_spacex[i], marker='o', linestyle='-', color=COLOR_SPX,
                alpha=0.7, linewidth=2, markersize=ms, markeredgecolor='white',
                markeredgewidth=2, zorder=11 if year>2015 else 6)

    # 填充预测区间
    ax.fill_between(years_future, y_spx_pred, y_total_pred, color='gray', alpha=0.08)

    # 标签与刻度
    ax.set_xlabel('Year', fontsize=STYLES['label_size'], fontweight=STYLES['font_weight'])
    ax.set_ylabel('Annual Launches', fontsize=STYLES['label_size'], fontweight=STYLES['font_weight'])
    ax.tick_params(axis='both', which='major', labelsize=STYLES['tick_size'])
    ax.set_xlim(2006, 2150)
    ax.set_ylim(-200, 11000)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=STYLES['legend_size'], frameon=True, framealpha=0.9, borderpad=1)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

# ----------------------------
# 8. CSV 导出函数
# ----------------------------
def export_future_launch_csv(filename='spacex_future_2050_2150.csv', start_year=2050, end_year=2150):
    years_csv = np.arange(start_year, end_year+1)
    x_csv = years_csv - 2020

    spx_pred_csv = logistic_func(x_csv, *popt_spx)
    total_pred_csv = logistic_func(x_csv, *popt_total)

    df = pd.DataFrame({
        'year': years_csv,
        'spacex_launches': spx_pred_csv.astype(int),
        'global_launches': total_pred_csv.astype(int)
    })

    df.to_csv(filename, index=False)
    print(f"CSV saved as {filename} ({len(years_csv)} rows).")
    return df

# ----------------------------
# 9. 运行
# ----------------------------
if __name__ == "__main__":
    plot_long_term_forecast()
    df_future = export_future_launch_csv()
    print(df_future.head())
    print(df_future.tail())
