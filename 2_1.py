import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

# ==========================================
# 1. Parameters
# ==========================================
N_req = 666700         
p_f = 0.0323           
p_s = 1 - p_f          
Cost_unit = 1.0        
Time_per_launch = 1    

# Statistical Calculations
mean = N_req / p_s
sigma = math.sqrt(N_req * p_f) / p_s

# ==========================================
# 2. Data Preparation
# ==========================================
x_min = mean - 3.2 * sigma
# 【修改点1】右边界再次收缩，因为方框贴近了，不需要那么多空白
x_max = mean + 5.2 * sigma 

x = np.linspace(x_min, x_max, 1000)
y = norm.pdf(x, mean, sigma)

confidences = [0.50, 0.80, 0.95, 0.99]
z_scores = [norm.ppf(c) for c in confidences]
x_cuts = [mean + z * sigma for z in z_scores]

# ==========================================
# 3. Plotting Setup
# ==========================================
plt.style.use('default') 
# 保持大尺寸画布
fig, ax = plt.figure(figsize=(16, 10), dpi=120), plt.gca()

# --- A. Main Curve ---
ax.plot(x, y, color='#2c3e50', lw=4.0, alpha=0.9) # 曲线加粗到4.0适配大字体

# --- B. Shaded Areas ---
ax.fill_between(np.linspace(x_min, mean, 500), norm.pdf(np.linspace(x_min, mean, 500), mean, sigma), color='#e3f2fd', alpha=0.9)
ax.fill_between(np.linspace(mean, x_cuts[1], 500), norm.pdf(np.linspace(mean, x_cuts[1], 500), mean, sigma), color='#90caf9', alpha=0.8)
ax.fill_between(np.linspace(x_cuts[1], x_cuts[2], 500), norm.pdf(np.linspace(x_cuts[1], x_cuts[2], 500), mean, sigma), color='#ffcc80', alpha=0.8)
ax.fill_between(np.linspace(x_cuts[2], x_cuts[3], 500), norm.pdf(np.linspace(x_cuts[2], x_cuts[3], 500), mean, sigma), color='#ef9a9a', alpha=0.8)
ax.fill_between(np.linspace(x_cuts[3], x_max, 500), norm.pdf(np.linspace(x_cuts[3], x_max, 500), mean, sigma), color='#c62828', alpha=0.8)

# --- C. Annotations (Huge Font & Tight Layout) ---
colors = ['#1565c0', '#1565c0', '#ef6c00', '#c62828']
box_titles = ['Mean (50%)', '80% Conf.', '95% Conf.', '99% Conf.']

y_peak = np.max(y)

# 【修改点2】X轴偏移量大幅减小
# 之前是 [1.3, 2.4, 3.5, 4.6]，现在让它们紧贴曲线
box_x_offsets = [0.7, 1.7, 2.8, 3.8]

# Y轴位置
box_y_levels = [0.85, 0.63, 0.41, 0.19]

for i, (x_val, title) in enumerate(zip(x_cuts, box_titles)):
    total_launches = x_val
    extra_launches = total_launches - N_req
    extra_cost = extra_launches * Cost_unit
    extra_time = (extra_launches * Time_per_launch) / 365.0
    
    plt.vlines(x_val, 0, norm.pdf(x_val, mean, sigma), colors='k', linestyles=':', alpha=0.4, lw=2.5)
    
    info_text = (
        f"[{title}]\n"
        f"Cost: {extra_cost:,.0f}M\n" 
        f"Time: {extra_time:.1f} Years"
    )
    
    text_y_pos = y_peak * box_y_levels[i]
    text_x_pos = mean + box_x_offsets[i] * sigma
    
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=1.0, edgecolor=colors[i], linewidth=3.0)
    
    ax.annotate(info_text, 
                xy=(x_val, norm.pdf(x_val, mean, sigma)), 
                xytext=(text_x_pos, text_y_pos), 
                # 【修改点3】箭头弧度变直 (rad -0.1)，因为距离很近了，太弯很难看
                arrowprops=dict(arrowstyle='->', connectionstyle=f'arc3,rad={-0.1}', color='#444', lw=2.5),
                bbox=props,
                fontsize=20,          # 【修改点4】字体加大到20
                family='monospace',
                fontweight='bold')

# --- D. Aesthetics ---
ax.autoscale(enable=True, axis='x', tight=True) 
ax.set_ylim(bottom=0)                           

# 坐标轴和图例全部配套加大
ax.set_xlabel('Total Launch Attempts Required', fontsize=22, fontweight='bold', labelpad=15)
ax.set_ylabel('Probability Density', fontsize=22, labelpad=15)
ax.tick_params(axis='x', labelsize=18)
ax.set_yticks([]) 

legend_elements = [
    Patch(facecolor='#e3f2fd', label='<50%: Under Budget'),
    Patch(facecolor='#90caf9', label='50%-80%: Normal'),
    Patch(facecolor='#ffcc80', label='80%-95%: Risk'),
    Patch(facecolor='#ef9a9a', label='95%-99%: High Cost'),
    Patch(facecolor='#c62828', label='>99%: Extreme'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=16, title="Confidence Zones", title_fontsize=18, framealpha=0.95)

plt.tight_layout()

# ==========================================
# 4. Save as PDF
# ==========================================
plt.savefig('launch_distribution.pdf', format='pdf', bbox_inches='tight')
print("PDF generated successfully.")

plt.show()
