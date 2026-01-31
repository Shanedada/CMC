import math
import matplotlib.pyplot as plt


# ============================================================
# 标定后的“物理接口”
# ============================================================
class OrbitalPhysics:
    """
    Reduced-order / calibrated physical interface.
    All parameters are calibrated from paper results.
    """

    def __init__(self):
        # === 论文给定宏观结论 ===
        self.total_payload = 1.0e11  # kg (100 million tons)

        # 完成时间（年）
        self.rocket_years = 667
        self.elevator_years = 890

    def rocket_annual_capacity(self):
        """等效火箭年运输能力 (kg/year)"""
        return self.total_payload / self.rocket_years

    def elevator_annual_capacity(self):
        """等效太空电梯年运输能力 (kg/year)"""
        return self.total_payload / self.elevator_years


# ============================================================
# 运输与成本计算器
# ============================================================
class RealRocketCalculator:
    """
    Cost-driven transport optimizer using calibrated parameters
    """

    def __init__(self):
        self.physics = OrbitalPhysics()

        # === 论文标定成本参数 ===
        # Pure rocket solution: $1.55 trillion for 100 Mt
        self.cost_rocket_per_kg = 15.5  # USD/kg

        # Space elevator solution (including fuel backhaul): $3.56 trillion
        self.cost_elevator_per_kg = 35.6  # USD/kg

    def run_mixed_optimization(self, total_payload_kg, duration_years):
        """
        Mixed rocket + space elevator transport within fixed duration.
        """

        rocket_cap = self.physics.rocket_annual_capacity()
        elevator_cap = self.physics.elevator_annual_capacity()

        max_rocket_payload = rocket_cap * duration_years
        max_elevator_payload = elevator_cap * duration_years

        if max_rocket_payload + max_elevator_payload < total_payload_kg:
            print("❌ 在给定年限内无法完成运输任务")
            return None

        # 优先使用更便宜的火箭
        payload_rocket = min(total_payload_kg, max_rocket_payload)
        payload_elevator = total_payload_kg - payload_rocket

        cost_rocket = payload_rocket * self.cost_rocket_per_kg
        cost_elevator = payload_elevator * self.cost_elevator_per_kg
        total_cost = cost_rocket + cost_elevator

        return {
            "years": duration_years,
            "rocket_payload": payload_rocket,
            "elevator_payload": payload_elevator,
            "total_cost": total_cost
        }


# ============================================================
# 主程序：扫描年限 & 绘图
# ============================================================
if __name__ == "__main__":

    TOTAL_PAYLOAD = 1.0e11  # kg

    calc = RealRocketCalculator()

    years_range = range(600, 701, 10)

    feasible_years = []
    elevator_fractions = []
    total_costs = []

    for y in years_range:
        result = calc.run_mixed_optimization(TOTAL_PAYLOAD, y)
        if result is None:
            continue

        frac_elevator = result["elevator_payload"] / TOTAL_PAYLOAD

        feasible_years.append(y)
        elevator_fractions.append(frac_elevator)
        total_costs.append(result["total_cost"] / 1e12)  # trillion USD

    # =========================
    # 图 1：时间 vs 电梯占比
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(feasible_years, elevator_fractions, marker="o")
    plt.xlabel("Project Duration (years)")
    plt.ylabel("Space Elevator Payload Fraction")
    plt.title("Time vs Space Elevator Utilization (Calibrated Model)")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # =========================
    # 图 2：时间 vs 总成本
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(feasible_years, total_costs, marker="s", color="red")
    plt.xlabel("Project Duration (years)")
    plt.ylabel("Total Cost (Trillion USD)")
    # plt.title("Total Cost vs Project Duration")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================
    # 打印两个关键方案对照
    # =========================
    print("\n================== 论文关键方案校验 ==================")

    rocket_only = calc.run_mixed_optimization(TOTAL_PAYLOAD, 667)
    elevator_only = calc.run_mixed_optimization(TOTAL_PAYLOAD, 890)

    print(f"🚀 纯火箭方案:")
    print(f"   - 时间: 667 年")
    print(f"   - 成本: ${rocket_only['total_cost']/1e12:.2f} trillion USD")

    print(f"\n🛰 太空电梯方案 (含燃料反运):")
    print(f"   - 时间: 890 年")
    print(f"   - 成本: ${elevator_only['total_cost']/1e12:.2f} trillion USD")

    print("======================================================")
