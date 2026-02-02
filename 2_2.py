import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 基础参数
# ==========================================
M_GOAL = 100_000_000      # 目标 1亿吨
CAP_BASE = 537_000        # 三个港口总年产能
COST_FIXED = 0            # 初始建设费
COST_BASE_OPS = 4e9       # 基础运营费 40亿/年

# ==========================================
# 定义策略列表
# ==========================================
scenarios = [
    # 0. 绝对理想对照组
    {
        "name": "Ideal Case (Perfect World)",
        "cost_maint": 0,
        "lambda": 0.0,
        "beta": 1.0,
        "color": "black",
        "style": "--"
    },
    
    # 1. 策略 A: 几乎不维护
    {
        "name": "A: Minimal Maint. (Run-to-Failure)",
        "cost_maint": 0,
        "lambda": 0.02,
        "beta": 0.95,     
        "color": "#d62728", # 红色
        "style": "-"
    },
    
    # 2. 策略 B: 标准维护
    {
        "name": "B: Standard Maint. (Balanced)",
        "cost_maint": 1e9,
        "lambda": 0.005,
        "beta": 0.95,
        "color": "#ff7f0e", # 橙色
        "style": "-"
    },
    
    # 3. 策略 C: 高投入维护 (亮黄色)
    {
        "name": "C: Proactive Maint. (High-Rel)",
        "cost_maint": 3e9,
        "lambda": 0.0005,
        "beta": 0.95,     
        "color": "#FFC300", # 亮黄色
        "style": "-"
    },
]

# ==========================================
# 仿真引擎
# ==========================================
def simulate_strategy(strategy_dict):
    current_year = 2050
    mass_transported = 0
    total_cost = COST_FIXED
    
    years = []
    mass_history = []
    
    lam = strategy_dict["lambda"]
    beta = strategy_dict["beta"]
    annual_spend = COST_BASE_OPS + strategy_dict["cost_maint"]
    
    # 依然计算到 3500年，保证数据完整
    while mass_transported < M_GOAL and current_year < 3500:
        t = current_year - 2050
        
        current_cap = CAP_BASE * beta * np.exp(-lam * t)
        mass_transported += current_cap
        total_cost += annual_spend
        
        years.append(current_year)
        mass_history.append(mass_transported / 1e6) 
        
        current_year += 1
        
        if current_cap < 100:
            break
            
    return years, mass_history, total_cost, mass_transported

# ==========================================
# 绘图与输出
# ==========================================
plt.figure(figsize=(14, 8))

print(f"{'Scenario':<35} | {'Duration':<10} | {'Total Cost':<15} | {'% Done':<8} | {'Status'}")
print("-" * 95)

for sc in scenarios:
    years, mass, cost, final_mass = simulate_strategy(sc)
    
    plt.plot(years, mass, label=f"{sc['name']}", 
             color=sc['color'], linestyle=sc['style'], linewidth=3.0)
    
    percent_done = (final_mass / M_GOAL) * 100
    
    if final_mass >= M_GOAL * 0.99: 
        duration = f"{years[-1] - 2050} yrs"
        status = "Success"
        cost_str = f"${cost/1e9:.1f} B"
    else:
        duration = f"{years[-1] - 2050} yrs"
        status = "FAILED" 
        cost_str = f"${cost/1e9:.1f} B (Wasted)"
        
    print(f"{sc['name']:<35} | {duration:<10} | {cost_str:<15} | {percent_done:5.1f}%   | {status}")

# 装饰图表
plt.axhline(y=100, color='blue', linestyle=':', linewidth=2, label="Target: 100M Tons")
plt.xlabel("Year", fontsize=16)
plt.ylabel("Cumulative Mass Delivered (Million Tons)", fontsize=16)

# 坐标轴刻度字体变大
plt.tick_params(axis='both', which='major', labelsize=14)

# 图例位置微调
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.9), fontsize=13, framealpha=0.9)

plt.grid(True, alpha=0.3)

# 横坐标只显示到 2450
plt.xlim(2050, 2450) 
plt.ylim(0, 110)

plt.tight_layout()

# ==========================================
# 保存为 PDF
# ==========================================
# bbox_inches='tight' 确保标签不会被裁剪掉
plt.savefig("mars_transport_scenarios.pdf", format='pdf', bbox_inches='tight')
print("\nPlot saved as 'mars_transport_scenarios.pdf'")

plt.show()
