import math

class OrbitalPhysics:
    """
    天体物理引擎
    负责根据万有引力定律计算理论速度
    """
    def __init__(self):
        # 基础常数
        self.G = 6.67430e-11
        
        # 地球参数
        self.M_earth = 5.972e24       # kg
        self.R_earth = 6371000        # m (平均半径)
        
        # 月球参数
        self.M_moon = 7.348e22        # kg
        self.R_moon = 1737100         # m
        self.D_earth_moon = 384400000 # 地月平均距离 (m)

    def get_orbit_velocity(self, central_mass, radius):
        """ 第一宇宙速度公式: v = sqrt(GM / r) """
        return math.sqrt(self.G * central_mass / radius)

    def get_hohmann_transfer_dv(self, r1, r2, central_mass):
        """
        霍曼转移计算 (从圆轨道 r1 变轨去 r2)
        返回: 在 r1 处需要的瞬间加速 (Delta V)
        """
        # 目标转移轨道的半长轴
        a_transfer = (r1 + r2) / 2
        
        # 1. 当前圆轨道速度
        v1 = math.sqrt(self.G * central_mass / r1)
        
        # 2. 转移轨道在近地点的速度 (活力公式 Vis-viva equation)
        # v = sqrt(GM * (2/r - 1/a))
        v_transfer = math.sqrt(self.G * central_mass * (2/r1 - 1/a_transfer))
        
        # 需要的加速量
        return v_transfer - v1

    def calculate_mission_dv(self, leo_alt_km, llo_alt_km):
        """
        自动计算任务所需的物理 Delta V
        """
        r_leo = self.R_earth + leo_alt_km * 1000
        r_llo = self.R_moon + llo_alt_km * 1000
        
        # --- 1. 计算地月转移 (TLI) ---
        # 这是一个从 LEO (200km) 到 月球高度 (38万km) 的霍曼转移
        # 我们计算在地球这边需要加速多少
        dv_tli_theoretical = self.get_hohmann_transfer_dv(r_leo, self.D_earth_moon, self.M_earth)
        
        # --- 2. 计算月球捕获 (LOI) ---
        # 这是一个相对复杂的近似：
        # 飞船到达月球时，速度不仅有转移速度，还要考虑月球引力井的加速。
        # 这里使用一种简化的"补丁圆锥法"估算：
        
        # 飞船相对于月球的"无穷远来流速度" (V_inf)
        # 大约等于：转移轨道远地点速度 - 月球公转速度 (约1022 m/s)
        v_moon_orbit = 1022 
        
        # 转移轨道在远地点(月球位置)的速度
        a_trans = (r_leo + self.D_earth_moon)/2
        v_apogee = math.sqrt(self.G * self.M_earth * (2/self.D_earth_moon - 1/a_trans))
        
        v_inf = abs(v_moon_orbit - v_apogee) # 相对速度
        
        # 在月球近圆轨道(LLO)处的双曲线飞掠速度
        # 能量守恒: V_perigee^2 = V_inf^2 + 2*GM_moon/r_llo
        v_flyby_perigee = math.sqrt(v_inf**2 + 2 * self.G * self.M_moon / r_llo)
        
        # 我们想要的环月轨道速度
        v_llo_circular = self.get_orbit_velocity(self.M_moon, r_llo)
        
        # 刹车所需的 dV
        dv_loi_theoretical = v_flyby_perigee - v_llo_circular

        return {
            'dv_tli': dv_tli_theoretical,
            'dv_loi': dv_loi_theoretical,
            'v_leo': self.get_orbit_velocity(self.M_earth, r_leo) # 环地速度
        }

class RealRocketCalculator:
    def __init__(self):
        self.physics = OrbitalPhysics()
        self.g0 = 9.80665

    def solve_stage(self, name, dv, isp, payload, struct_ratio):
        """ 逆向火箭方程 (同前，不做修改) """
        ve = isp * self.g0
        mass_ratio = math.exp(dv / ve)
        denominator = 1 - mass_ratio * struct_ratio
        
        if denominator <= 0.001:
            raise ValueError(f"❌ {name} 任务不可行！结构系数 {struct_ratio} 太重了。")
            
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
        print(f"🚀 物理驱动的航天任务计算器")
        print(f"{'='*70}")
        
        # ==========================================
        # 第一步：计算物理需求 (这是算出来的！)
        # ==========================================
        leo_h = 300 # 地球停泊轨道高度 km
        llo_h = 100 # 月球环绕轨道高度 km (比如嫦娥一号)
        
        print(f"📡 轨道参数初始化:")
        print(f"   - 地球停泊高度: {leo_h} km")
        print(f"   - 月球目标高度: {llo_h} km")
        
        physics_data = self.physics.calculate_mission_dv(leo_h, llo_h)
        
        # 增加一点点余量 (Margin)，应对变轨误差，通常 +2%
        dv_loi_req = physics_data['dv_loi'] * 1.02
        dv_tli_req = physics_data['dv_tli'] * 1.02
        
        print(f"\n📐 轨道力学计算结果 (含2%余量):")
        print(f"   - 月球捕获 (LOI): {dv_loi_req:.0f} m/s (理论值: {physics_data['dv_loi']:.0f})")
        print(f"   - 地月转移 (TLI): {dv_tli_req:.0f} m/s (理论值: {physics_data['dv_tli']:.0f})")
        
        # ==========================================
        # 第二步：逆向设计火箭
        # ==========================================
        
        # 1. 月球捕获级
        stage_loi = self.solve_stage(
            "1. 月球捕获级", 
            dv_loi_req, 
            hardware_specs['isp_loi'], 
            satellite_mass, 
            hardware_specs['struct_loi']
        )
        
        # 2. 地月转移级
        stage_tli = self.solve_stage(
            "2. 地月转移级", 
            dv_tli_req, 
            hardware_specs['isp_tli'], 
            stage_loi['total_initial'], 
            hardware_specs['struct_tli']
        )
        
        # 3. 地面发射 (Launch)
        # 这里需要特别说明：
        # 理论速度 = 7.8 km/s
        # 实际需要 = 理论 - 自转 + 损耗
        lat = hardware_specs['launch_lat']
        v_rot = 465 * math.cos(math.radians(lat)) # 简化的自转计算
        
        # 损耗 (Gravity Loss + Drag Loss)
        # 这个是没法简单算出来的，必须用经验值。
        # 好的液体火箭通常是 1.2 ~ 1.5 km/s
        gravity_drag_loss = 1400 
        
        dv_launch_total = physics_data['v_leo'] - v_rot + gravity_drag_loss
        
        print(f"\n🚀 发射阶段需求:")
        print(f"   - 环绕速度: {physics_data['v_leo']:.0f} m/s")
        print(f"   - 自转借力: {v_rot:.0f} m/s (纬度 {lat})")
        print(f"   - 重力/风阻损耗: {gravity_drag_loss} m/s (经验值)")
        print(f"   - 总计入轨 dV: {dv_launch_total:.0f} m/s")

        # 分配发射级 (40% 二级, 60% 一级)
        dv_s2 = dv_launch_total * 0.4
        dv_s1 = dv_launch_total - dv_s2

        stage_s2 = self.solve_stage("3. 发射二级", dv_s2, hardware_specs['isp_s2'], stage_tli['total_initial'], hardware_specs['struct_s2'])
        stage_s1 = self.solve_stage("4. 发射一级", dv_s1, hardware_specs['isp_s1'], stage_s2['total_initial'], hardware_specs['struct_s1'])

        # 输出
        self.print_report([stage_loi, stage_tli, stage_s2, stage_s1], satellite_mass)

    def print_report(self, stages, payload):
        print(f"\n{'='*80}")
        print(f"{'阶段':<15} | {'任务 dV (m/s)':<12} | {'总重 (t)':>10} | {'发动机 Isp':>10}")
        print("-" * 80)
        for s in stages:
            print(f"{s['name']:<15} | "
                  f"{s['dv']:<12.0f} | "
                  f"{s['total_initial']/1000:10.1f} | "
                  f"{hardware_specs['isp_' + ('loi' if '捕获' in s['name'] else 'tli' if '转移' in s['name'] else 's2' if '二级' in s['name'] else 's1')]:10.0f}") # 这里偷懒匹配了一下key
        
        print(f"{'='*80}")
        print(f"🔥 起飞总重: {stages[-1]['total_initial']/1000:.1f} 吨")


# ==========================================
# 用户输入区：只填“硬件参数”，不填“物理结果”
# ==========================================
if __name__ == "__main__":
    
    # 你的卫星有多重？
    target_satellite = 14500 # kg
    
    # 这里全是【硬件规格】，不是计算结果
    # 你不能“计算”出比冲，就像你不能计算出法拉利发动机的马力一样，这得查说明书。
    hardware_specs = {
        'launch_lat': 28.5, # 卡纳维拉尔角
        
        # --- 发动机 ---
        'isp_loi': 320,     # 卫星仍然建议用常规毒燃料 (体积小，可长期储存)
        'isp_tli': 450,     # 上面级：真空氢氧 (RL-10 级别，非常优秀)
        'isp_s2':  450,     # 二级：真空氢氧
        'isp_s1':  365,     # 一级：海平面氢氧 (受大气压削弱，不能填440!)
        
        # --- 结构 ---
        'struct_loi': 0.15, # 卫星结构通常较重（带太阳能板、仪器等）
        'struct_tli': 0.12, # 氢氧上面级，储箱大，保温层重
        'struct_s2':  0.10, # 氢氧二级
        'struct_s1':  0.09  # 氢氧一级 (比煤油火箭的0.05要“劣”很多，因为罐子巨大)
    }


    calc = RealRocketCalculator()
    calc.run(target_satellite, hardware_specs)
