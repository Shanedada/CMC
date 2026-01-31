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
        # 恒星日 = 86164秒
        self.omega_earth = 2 * math.pi / 86164 

    def get_orbit_velocity(self, central_mass, radius):
        return math.sqrt(self.G * central_mass / radius)

    def calculate_tether_mission(self, tether_alt_km, llo_alt_km):
        """
        计算从旋转系绳释放的任务需求
        """
        r_start = self.R_earth + tether_alt_km * 1000
        r_llo = self.R_moon + llo_alt_km * 1000

        # 1. 初始状态：被系绳甩出去的速度
        # v = ω * r
        v_initial = self.omega_earth * r_start
        
        # 2. 地月转移 (TLI) - 不需要动力
        # 此时的能量 E = v^2/2 - GM/r
        # 如果 E > 0，说明是双曲线轨道（飞离地球）
        energy_initial = (v_initial**2)/2 - (self.G * self.M_earth / r_start)
        
        # 3. 计算到达月球距离时的速度
        # 能量守恒: E_start = E_arrival
        # v_arrival^2 / 2 - GM_earth / r_moon = E_start
        # 这里的 potential energy 近似取月球轨道处的地球势能
        v_arrival_sq = 2 * (energy_initial + self.G * self.M_earth / self.D_earth_moon)
        v_arrival_earth_frame = math.sqrt(v_arrival_sq)
        
        # 4. 月球捕获 (LOI) - 重点在这里
        # 我们假设“顺向”到达（追着月球跑），相对速度最小
        v_moon_orbital = 1022 # 月球公转速度
        
        # 相对速度 (V_infinity)
        v_inf_moon = abs(v_arrival_earth_frame - v_moon_orbital)
        
        # 在月球近地点 (100km高度) 的飞掠速度
        # 能量守恒: v_peri^2 = v_inf^2 + 2*GM_moon/r_llo
        v_flyby_perigee = math.sqrt(v_inf_moon**2 + 2 * self.G * self.M_moon / r_llo)
        
        # 我们想要的环月圆轨道速度
        v_llo_circular = self.get_orbit_velocity(self.M_moon, r_llo)
        
        # 刹车 Delta V
        dv_loi = v_flyby_perigee - v_llo_circular

        return {
            'v_init': v_initial,
            'v_arrival': v_arrival_earth_frame,
            'dv_tli': 0,           # 不需要加速！
            'dv_loi': dv_loi       # 需要巨大刹车
        }

class RealRocketCalculator:
    def __init__(self):
        self.physics = OrbitalPhysics()
        self.g0 = 9.80665

    def solve_stage(self, name, dv, isp, payload, struct_ratio):
        """ 火箭方程计算 """
        ve = isp * self.g0
        mass_ratio = math.exp(dv / ve)
        denominator = 1 - mass_ratio * struct_ratio
        
        if denominator <= 0.001:
            raise ValueError(f"❌ {name} 任务不可行！需要 {dv:.0f} m/s，但结构系数限制了上限。")
            
        stage_total = payload * (mass_ratio - 1) / denominator
        return {
            "name": name,
            "dv": dv,
            "payload": payload,
            "stage_total": stage_total,
            "total_initial": payload + stage_total
        }

    def run(self, satellite_mass, hardware_specs):
        print(f"{'='*70}")
        print(f"🚀 太空电梯/系绳弹射任务 (无动力发射版)")
        print(f"{'='*70}")
        
        start_h = 100000 
        llo_h = 100
        
        # 计算物理参数
        phys = self.physics.calculate_tether_mission(start_h, llo_h)
        
        print(f"📊 物理情景分析:")
        print(f"   1. 出发: 在 {start_h} km 高度被甩出")
        print(f"      - 初始速度: {phys['v_init']:.0f} m/s (远超逃逸速度 2700 m/s)")
        print(f"      - 结论: 不需要点火，直接起飞！")
        
        print(f"   2. 到达: 飞抵月球附近")
        print(f"      - 地心系速度: {phys['v_arrival']:.0f} m/s")
        print(f"      - 相对月球速度: {abs(phys['v_arrival'] - 1022):.0f} m/s (非常快)")
        
        # 增加 5% 刹车余量
        dv_loi_req = phys['dv_loi'] * 1.05
        
        print(f"   3. 刹车: 必须减速才能入轨")
        print(f"      - 刹车需求 (LOI): {dv_loi_req:.0f} m/s")
        print(f"      (对比: 普通阿波罗任务只需要约 900 m/s 刹车)")

        # ==========================================
        # 设计火箭
        # ==========================================
        # 既然没有发射级，也没有转移级，那我们只需要一个巨大的刹车级
        # 为了保险，我们加一个微小的“中途修正级” (TCM)
        
        # 1. 刹车级 (任务最重)
        stage_brake = self.solve_stage(
            "1. 月球急刹车级", 
            dv_loi_req, 
            hardware_specs['isp_loi'], 
            satellite_mass, 
            hardware_specs['struct_loi']
        )
        
        # 2. 轨道修正级 (TCM) - 仅做微调，防止撞歪
        stage_tcm = self.solve_stage(
            "2. 中途修正模块", 
            50, # 象征性 50 m/s
            hardware_specs['isp_tcm'], 
            stage_brake['total_initial'], 
            hardware_specs['struct_tcm']
        )
        
        self.print_report([stage_brake, stage_tcm], satellite_mass)

    def print_report(self, stages, payload):
        print(f"\n{'='*80}")
        print(f"{'阶段':<15} | {'任务 dV':<12} | {'总重 (kg)':>12} | {'说明':<20}")
        print("-" * 80)
        for s in stages:
            print(f"{s['name']:<15} | "
                  f"{s['dv']:<12.0f} | "
                  f"{s['total_initial']:12.1f} | "
                  f"{'利用惯性飞行' if s['dv'] < 100 else '消耗大量燃料'}")
        
        print(f"{'='*80}")
        print(f"🔥 系统总重: {stages[-1]['total_initial']/1000:.2f} 吨")
        print(f"📦 有效载荷: {payload/1000:.2f} 吨")
        print(f"💡 总结: 虽然省去了发射和转移燃料，但为了在月球停下来，\n"
              f"        这一级火箭的重量依然很可观（主要是刹车燃料）。")

if __name__ == "__main__":
    target_satellite = 145000 # kg
    
    hardware_specs = {
        # 刹车级参数 (建议用高性能发动机，因为刹车量太大了)
        'isp_loi': 450,     # 氢氧发动机
        'struct_loi': 0.12, # 结构系数
        
        # 修正级参数
        'isp_tcm': 320,     # 普通毒燃料
        'struct_tcm': 0.10,
    }

    calc = RealRocketCalculator()
    calc.run(target_satellite, hardware_specs)
