import math

# ==============================
# 轨道物理计算
# ==============================
class OrbitalPhysics:
    def __init__(self):
        self.G = 6.67430e-11
        self.M_earth = 5.972e24
        self.R_earth = 6371000
        self.M_moon = 7.348e22
        self.R_moon = 1737100
        self.D_earth_moon = 384404000  # m
        self.omega_earth = 2 * math.pi / 86164  # rad/s

    def get_orbit_velocity(self, mass, radius):
        return math.sqrt(self.G * mass / radius)

    def get_hohmann_transfer_dv(self, r1, r2, mass_central):
        a_trans = (r1 + r2) / 2
        v1 = math.sqrt(self.G * mass_central / r1)
        v_trans = math.sqrt(self.G * mass_central * (2 / r1 - 1 / a_trans))
        return v_trans - v1

    def calculate_dv(self, leo_alt_km=300, llo_alt_km=100):
        r_leo = self.R_earth + leo_alt_km * 1000
        r_llo = self.R_moon + llo_alt_km * 1000
        # 地月转移 Δv
        dv_tli = self.get_hohmann_transfer_dv(r_leo, self.D_earth_moon, self.M_earth)
        # 月球捕获 Δv
        v_moon_orbital = 1022  # m/s
        a_trans = (r_leo + self.D_earth_moon) / 2
        v_apogee = math.sqrt(self.G * self.M_earth * (2 / self.D_earth_moon - 1 / a_trans))
        v_inf = abs(v_apogee - v_moon_orbital)
        v_flyby = math.sqrt(v_inf ** 2 + 2 * self.G * self.M_moon / r_llo)
        dv_loi = v_flyby - self.get_orbit_velocity(self.M_moon, r_llo)
        return dv_tli, dv_loi

# ==============================
# 火箭逆向计算
# ==============================
class RocketCalculator:
    def __init__(self, g0=9.80665):
        self.g0 = g0
        self.physics = OrbitalPhysics()

    def solve_stage(self, dv, isp, payload, struct_ratio):
        ve = isp * self.g0
        mass_ratio = math.exp(dv / ve)
        denominator = 1 - mass_ratio * struct_ratio
        if denominator <= 0.001:
            raise ValueError(f"任务不可行: dv={dv}, 结构比={struct_ratio}")
        stage_total = payload * (mass_ratio - 1) / denominator
        return stage_total  # kg

    def compute_rocket_mass(self, payload_kg, hardware):
        dv_tli, dv_loi = self.physics.calculate_dv()
        # LOI级
        mass_loi = self.solve_stage(dv_loi * 1.02, hardware['isp_loi'], payload_kg, hardware['struct_loi'])
        # TLI级
        mass_tli = self.solve_stage(dv_tli * 1.02, hardware['isp_tli'], payload_kg + mass_loi, hardware['struct_tli'])
        # 一级二级发射级
        # 假设发射 Δv = 9.5 km/s
        dv_launch_total = 9500
        dv_s2 = dv_launch_total * 0.4
        dv_s1 = dv_launch_total - dv_s2
        mass_s2 = self.solve_stage(dv_s2, hardware['isp_s2'], payload_kg + mass_loi + mass_tli, hardware['struct_s2'])
        mass_s1 = self.solve_stage(dv_s1, hardware['isp_s1'], payload_kg + mass_loi + mass_tli + mass_s2, hardware['struct_s1'])
        total_mass = payload_kg + mass_loi + mass_tli + mass_s2 + mass_s1
        fuel_mass = total_mass - payload_kg
        return total_mass, fuel_mass

# ==============================
# 成本计算与优化
# ==============================
class MixedTransport:
    def __init__(self, total_material_ton=1e8):
        self.total_material = total_material_ton  # 吨
        # Space Elevator参数
        self.elevator_capacity_per_year = 179000 * 3  # 三个Harbor
        self.elevator_cost_per_ton = 1000  # 美元/吨电费+维护（假设）
        # Rocket参数
        self.rocket_payload = 145  # 吨
        self.rocket_fixed_cost = 900e3  # 美元/次
        self.rocket_fuel_price_per_kg = 1.2  # USD/kg
        self.rocket_hardware = {
            'isp_loi': 320, 'struct_loi': 0.15,
            'isp_tli': 450, 'struct_tli': 0.12,
            'isp_s2': 450, 'struct_s2': 0.10,
            'isp_s1': 365, 'struct_s1': 0.09
        }
        self.rocket_calc = RocketCalculator()

    def compute_cost_time(self, elevator_fraction):
        """ elevator_fraction: 0~1 """
        material_elevator = self.total_material * elevator_fraction
        material_rocket = self.total_material * (1 - elevator_fraction)

        # Space Elevator
        years_elevator = material_elevator / self.elevator_capacity_per_year
        cost_elevator = material_elevator * self.elevator_cost_per_ton

        # Rockets
        n_rockets = math.ceil(material_rocket / self.rocket_payload)
        fuel_mass_total = 0
        for _ in range(n_rockets):
            _, fuel_mass = self.rocket_calc.compute_rocket_mass(self.rocket_payload * 1000, self.rocket_hardware)
            fuel_mass_total += fuel_mass
        cost_rocket = n_rockets * self.rocket_fixed_cost + fuel_mass_total * self.rocket_fuel_price_per_kg

        # 总成本
        total_cost = cost_elevator + cost_rocket
        # 总时间（取耗时最长）
        total_time_years = max(years_elevator, n_rockets / 10)  # 假设每年每个基地10次发射能力
        return total_cost, total_time_years

    def optimize_fraction(self):
        best_cost = float('inf')
        best_frac = 0
        for frac in [i/100 for i in range(0, 101)]:
            cost, t = self.compute_cost_time(frac)
            if cost < best_cost:
                best_cost = cost
                best_frac = frac
        final_cost, final_time = self.compute_cost_time(best_frac)
        return best_frac, final_cost, final_time

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    mt = MixedTransport()
    best_frac, total_cost, total_time = mt.optimize_fraction()
    print("最优Space Elevator比例:", best_frac)
    print("总成本 (USD):", f"{total_cost/1e9:.2f} 亿")
    print("总工期 (年):", f"{total_time:.1f}")
