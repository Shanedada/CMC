import scipy.integrate as integrate

def calculate_global_average_cost(mass_kg, height_km, efficiency=0.8):
    
    # --- 1. 定义各国电价 (美元 USD / kWh) ---
    # 数据参考 2023-2024 年工业/民用混合估算值
    # 包含了高电价区(欧洲)和低电价区(亚洲/中东)
    electricity_prices_usd = {
        "USA (美国)": 0.19,      # 区域差异大，取平均
        "Germany (德国)": 0.458,  # 能源价格较高
        "India (印度)": 0.071,    # 相对便宜
        "Japan (日本)": 0.236,    # 能源依赖进口
        "UK (英国)": 0.415
    }
    
    # 计算平均电价
    avg_price_usd = sum(electricity_prices_usd.values()) / len(electricity_prices_usd)
    
    # --- 2. 物理常数 ---
    G = 6.67430e-11   # 万有引力常数
    M_EARTH = 5.972e24   # 地球质量 (kg)
    R_EARTH = 6_371_000  # 地球半径 (m)
    
    # --- 3. 积分计算重力势能 (Work) ---
    # 目标半径 = 地球半径 + 高度
    r_initial = R_EARTH
    r_final = R_EARTH + (height_km * 1000)
    
    # 定义引力函数 F(r) = GMm / r^2
    def gravitational_force(r):
        return (G * M_EARTH * mass_kg) / (r**2)
    
    # 积分：从地面积到目标高度
    energy_joules, error = integrate.quad(gravitational_force, r_initial, r_final)
    
    # --- 4. 能量与费用转换 ---
    # 焦耳 -> 千瓦时
    kwh_theoretical = energy_joules / 3_600_000
    
    # 考虑系统效率 (如 80%)
    kwh_actual = kwh_theoretical / efficiency
    
    # 计算总价
    total_cost_usd = kwh_actual * avg_price_usd

    # --- 5. 打印详细报告 ---
    print(f"{'='*50}")
    print(f"🌍 全球平均电价版：太空运输成本计算器")
    print(f"{'='*50}")
    
    print(f"📦 运输质量: {mass_kg:,.0f} kg")
    print(f"🚀 目标高度: {height_km:,.0f} km")
    print(f"⚡ 系统效率: {efficiency*100}%")
    print(f"-"*50)
    
    print("💰 选取的电价参考 (USD/kWh):")
    for country, price in electricity_prices_usd.items():
        print(f"   - {country:<15}: ${price:.2f}")
    print(f"   -------------------------")
    print(f"   📊 平均电价: ${avg_price_usd:.3f} / kWh")
    print(f"-"*50)
    
    print(f"🔬 物理计算结果 (积分法):")
    print(f"   ΔEp (重力势能): {energy_joules:.4e} Joules")
    print(f"   实际耗电量:     {kwh_actual:,.2f} kWh")
    print(f"-"*50)
    
    print(f"💵 最终电费账单 (USD):")
    print(f"   ${total_cost_usd:,.2f}")
    print(f"   (约合 {total_cost_usd/1_000_000:.2f} 百万美元)")
    print(f"{'='*50}")

# --- 执行 ---
# 1.79亿公斤, 10万公里
calculate_global_average_cost(179_000_000, 100_000)
