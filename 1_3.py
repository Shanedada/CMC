import math

class OrbitalPhysics:
    def __init__(self):
        self.G = 6.67430e-11
        self.M_earth = 5.972e24       
        self.R_earth = 6371000        
        self.M_moon = 7.348e22        
        self.R_moon = 1737100         
        self.D_earth_moon = 384400000 
        
        # 地球自转角速度 (rad/s)
        self.omega_earth = 2 * math.pi / 86164 

    def get_orbit_velocity(self, central_mass, radius):
        return math.sqrt(self.G * central_mass / radius)

    def calculate_tether_mission(self, tether_alt_km, llo_alt_km, payload_kg):
        """
        电梯发射任务：计算v、dv和电力消耗
        """
        r_start = self.R_earth + tether_alt_km * 1000
        r_llo = self.R_moon + llo_alt_km * 1000

        # 初始速度 v = ω * r
        v_initial = self.omega_earth * r_start

        # 能量守恒到月球距离
        energy_initial = (v_initial**2)/2 - (self.G * self.M_earth / r_start)
        # 如果能量为负（未超过逃逸速度），直接设为零以避免math domain error
        energy_initial = max(energy_initial, 0)
        v_arrival_sq = 2 * (energy_initial + self.G * self.M_earth / self.D_earth_moon)
        v_arrival_earth_frame = math.sqrt(v_arrival_sq)

        # 月球捕获
        v_moon_orbital = 1022
        v_inf_moon = abs(v_arrival_earth_frame - v_moon_orbital)
        v_flyby_perigee = math.sqrt(v_inf_moon**2 + 2 * self.G * self.M_moon / r_llo)
        v_llo_circular = self.get_orbit_velocity(self.M_moon, r_llo)
        dv_loi = v_flyby_perigee - v_llo_circular

        # 电力消耗 (焦耳 -> kWh)
        delta_U = self.G * self.M_earth * payload_kg * (1/self.R_earth - 1/r_start)
        elec_energy_kwh = delta_U / 3.6e6

        return {
            'v_init': v_initial,
            'v_arrival': v_arrival_earth_frame,
            'dv_tli': 0,
            'dv_loi': dv_loi,
            'elec_energy_kwh': elec_energy_kwh
        }

class RealRocketCalculator:
    def __init__(self):
        self.physics = OrbitalPhysics()

    def run_mixed_optimization(self, total_payload_kg, duration_years):
        """
        在固定 duration 内，混合电梯+火箭运输，总量为 total_payload_kg。
        输出混合方案下各自有效荷载、燃料和电量消耗。
        """
        # 比例关系
        elevator_fuel_ratio = 4.155  # 每 1 kg 有效荷载需要 4.155 kg 燃料
        rocket_fuel_ratio = 7.2      # 每 1 kg 有效荷载需要 7.2 kg 燃料

        # 假设每年可以发射量（单位: kg）
        elevator_payload_per_year = 50_000_000  # 电梯每年有效荷载
        rocket_payload_per_year = 20_000_000    # 火箭每年有效荷载

        # 最大可运输总量
        max_elevator_payload = elevator_payload_per_year * duration_years
        max_rocket_payload = rocket_payload_per_year * duration_years

        # 判断是否可完成任务
        if max_elevator_payload + max_rocket_payload < total_payload_kg:
            print("❌ 在给定年限内无法完成任务！")
            return

        # 优化方案：尽量使用低燃料比方案（电梯）优先
        payload_elevator = min(total_payload_kg, max_elevator_payload)
        remaining_payload = total_payload_kg - payload_elevator
        payload_rocket = min(remaining_payload, max_rocket_payload)

        # 电梯燃料
        elevator_fuel = payload_elevator * elevator_fuel_ratio
        # 电梯电力
        elevator_phys = self.physics.calculate_tether_mission(100_000, 100, payload_elevator)
        elevator_energy_kwh = elevator_phys['elec_energy_kwh']

        # 火箭燃料
        rocket_fuel = payload_rocket * rocket_fuel_ratio

        total_fuel = elevator_fuel + rocket_fuel
        total_energy = elevator_energy_kwh  # 只有电梯消耗电

        print(f"\n✅ 混合方案最优分配 (duration={duration_years} 年):")
        print(f"   - 电梯有效荷载: {payload_elevator:,.0f} kg, 燃料: {elevator_fuel:,.0f} kg, 电力: {elevator_energy_kwh:,.0f} kWh")
        print(f"   - 火箭有效荷载: {payload_rocket:,.0f} kg, 燃料: {rocket_fuel:,.0f} kg")
        print(f"   - 总有效荷载: {payload_elevator + payload_rocket:,.0f} kg")
        print(f"   - 总燃料消耗: {total_fuel:,.0f} kg, 总电力消耗: {total_energy:,.0f} kWh")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    total_payload = 100_000_000_000  # 100 million tons -> kg

    years_range = range(1429, 3000)  # 1429 - 1500
    elevator_ratios = []  # 电梯方案有效荷载占比
    feasible_years = []

    calc = RealRocketCalculator()

    for y in years_range:
        try:
            # 计算每年的最大可用荷载
            elevator_payload_per_year = 50_000_000  # 假设电梯每年可送50M kg
            rocket_payload_per_year = 20_000_000    # 火箭每年可送20M kg

            max_elevator_payload = elevator_payload_per_year * y
            max_rocket_payload = rocket_payload_per_year * y

            if max_elevator_payload + max_rocket_payload < total_payload:
                continue  # 年限不够，跳过

            payload_elevator = min(total_payload, max_elevator_payload)
            remaining_payload = total_payload - payload_elevator
            payload_rocket = min(remaining_payload, max_rocket_payload)

            # 电梯占比
            ratio = payload_elevator / total_payload
            elevator_ratios.append(ratio)
            feasible_years.append(y)

        except Exception as e:
            continue

    # 绘制图像
    plt.figure(figsize=(10,6))
    plt.plot(feasible_years, elevator_ratios, marker='o', color='blue')
    plt.xlabel("Duration (years)")
    plt.ylabel("Elevator payload fraction")
    plt.title("Time vs Elevator Payload Fraction for Moon Colony Transport")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.show()
