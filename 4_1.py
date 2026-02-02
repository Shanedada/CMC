import numpy as np
import matplotlib.pyplot as plt

G0 = 9.81  
PAYLOAD_TARGET_KG = 150 * 1000  # 150吨
LAUNCHES_PER_YEAR = 1000

# === 用户指定参数 ===
STRUCTURAL_COEFF = 0.12  # 结构系数 (0.12 代表结构很重，技术水平较低)
# ===================

# 发动机参数
ISP_AVERAGE = 450      # 第一级平均比冲 (秒)

# 排放因子 (Emission Factors, EF) - 每燃烧1kg燃料产生的污染物kg数
EF_CO2 = 0.6
EF_BC = 0.035
EF_NOX = 0.014  

# 环境危害权重 (将污染物换算成等效CO2的倍率)
# 平流层黑炭(BC)吸热能力极强
WEIGHT_BC_STRATOSPHERE = 500  
# 平流层氮氧化物(NOx)破坏臭氧层，也是间接温室气体，假设权重为50 (约为CO2的50倍)
WEIGHT_NOX_STRATOSPHERE = 50  

# ==========================================
# 2. 齐奥尔科夫斯基火箭方程 (核心物理引擎)
# ==========================================
def calculate_fuel_required(payload_mass, delta_v, isp):
    """
    根据载荷、所需速度增量和比冲，反推需要多少燃料。
    考虑了结构系数带来的死重。
    """
    exhaust_velocity = isp * G0
    # 质量比 R = exp(DeltaV / Ve)
    mass_ratio = np.exp(delta_v / exhaust_velocity)
    
    # 核心难点：如果结构系数太大，分母会趋近0或负数，代表物理上无法实现
    denominator = 1 - STRUCTURAL_COEFF * (mass_ratio - 1)
    
    if denominator <= 0.001:
        return None, None 
        
    fuel_mass = payload_mass * (mass_ratio - 1) / denominator
    struct_mass = fuel_mass * STRUCTURAL_COEFF
    total_stage_mass = payload_mass + fuel_mass + struct_mass
    
    return fuel_mass, total_stage_mass

# ==========================================
# 3. 运行模拟
# ==========================================
def run_simulation():
    # --- Stage 2 (太空段) ---
    # 假设第二级使用高比冲引擎 (360s)，负责最后的 6000 m/s
    fuel_s2, mass_s2_total = calculate_fuel_required(PAYLOAD_TARGET_KG, 6000, 360)
    if fuel_s2 is None: 
        print("Error: Stage 2 结构过重，无法完成任务。")
        return None

    # --- Stage 1 (大气段) ---
    # 第一级负责将第二级推到一定速度 (3400 m/s)
    # 这里的 payload 就是 Stage 2 的总重
    fuel_s1, mass_s1_total = calculate_fuel_required(mass_s2_total, 3400, ISP_AVERAGE)
    
    if fuel_s1 is None: 
        print("Error: Stage 1 结构过重，无法完成任务。")
        return None
    
    # --- 污染物计算 ---
    # 假设主要污染来自第一级 (在大气层内燃烧)
    polluting_fuel_per_launch = fuel_s1 
    
    # 计算完成 100 Megatons (1亿吨) 运输所需的总次数
    total_cargo_needed = 100 * 10**6 * 1000  # kg
    total_launches = np.ceil(total_cargo_needed / PAYLOAD_TARGET_KG)
    
    total_polluting_fuel = total_launches * polluting_fuel_per_launch
    
    # 原始排放质量 (吨)
    emissions_mass = {
        "CO2": (total_polluting_fuel * EF_CO2) / 1000,
        "BC": (total_polluting_fuel * EF_BC) / 1000,
        "NOx": (total_polluting_fuel * EF_NOX) / 1000  
    }
    
    # ===【修正点】环境影响评分计算 ===
    # 将所有污染物按权重换算成“等效 CO2”
    impact_components = {
        "CO2_Base": emissions_mass["CO2"] * 1,
        "BC_Weighted": emissions_mass["BC"] * WEIGHT_BC_STRATOSPHERE,
        "NOx_Weighted": emissions_mass["NOx"] * WEIGHT_NOX_STRATOSPHERE # 修正：纳入 NOx
    }
    
    impact_score = sum(impact_components.values())
    
    return {
        "launches": total_launches,
        "emissions_mass": emissions_mass,
        "impact_components": impact_components,
        "impact_score": impact_score
    }

res = run_simulation()

# ==========================================
# 4. 绘图与分析
# ==========================================
if res:
    print(f"=== 环境影响评估报告 ===")
    print(f"物理约束: 结构系数 = {STRUCTURAL_COEFF} (较重)")
    print(f"总发射次数: {res['launches']:,.0f}")
    print("-" * 30)
    print(f"1. CO2 物理质量: {res['emissions_mass']['CO2']:,.0e} 吨")
    print(f"2. BC  物理质量: {res['emissions_mass']['BC']:,.0e} 吨")
    print(f"3. NOx 物理质量: {res['emissions_mass']['NOx']:,.0e} 吨")
    print("-" * 30)
    print(f"总环境影响评分 (等效CO2): {res['impact_score']:,.0e} 吨")

    plt.figure(figsize=(10, 6))
    
    # 绘图数据
    categories = [
        'CO2 (Mass)', 
        'NOx (Mass)', 
        'Black Carbon (Mass)', 
        'Total Equivalent\nImpact (Weighted)'
    ]
    
    values = [
        res['emissions_mass']['CO2'], 
        res['emissions_mass']['NOx'], 
        res['emissions_mass']['BC'], 
        res['impact_score']  # 修正：显示总加权影响
    ]
    
    colors = ['#7f7f7f', '#9467bd', 'black', '#d62728'] 
    
    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    
    plt.yscale('log') # 使用对数坐标，因为数值差异巨大
    plt.ylabel('Metric Tons (Log Scale)')
    plt.title(f'Environmental Impact Analysis\n(Including Weighted NOx & BC)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 在柱状图上方标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                 f'{height:.1e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("计算失败：火箭结构太重，无法满足入轨速度要求。")
