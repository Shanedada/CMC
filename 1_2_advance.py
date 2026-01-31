import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Earth, Moon
from poliastro.twobody import Orbit

# ==============================
# 创建 LEO -> Moon 转移轨迹
# ==============================
def create_earth_to_moon_transfer(leo_alt_km=300, llo_alt_km=100):
    # 1. LEO圆轨道
    leo = Orbit.circular(Earth, alt=leo_alt_km * u.km)

    # 2. Hohmann转移到月球平均轨道
    r_apogee = 384400 * u.km
    # 提取标量
    r_leo_val = leo.r.to(u.km).value.item()
    r_apogee_val = r_apogee.to(u.km).value.item()
    a_trans_val = (r_leo_val + r_apogee_val)/2 * u.km
    ecc_val = (r_apogee_val - r_leo_val) / (r_apogee_val + r_leo_val)

    # 构造转移轨道
    trans_orbit = Orbit.from_classical(
        attractor=Earth,
        a=a_trans_val,
        ecc=ecc_val,
        inc=0 * u.deg,
        raan=0 * u.deg,
        argp=0 * u.deg,
        nu=0 * u.deg
    )
    return leo, trans_orbit



# ==============================
# 主程序
# ==============================
leo, trans_orbit = create_earth_to_moon_transfer()

# 使用 matplotlib 3D 绘图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 获取轨道点
leo_xyz = leo.sample(500).get_xyz().to(u.km)
trans_xyz = trans_orbit.sample(500).get_xyz().to(u.km)

# 绘制轨道
ax.plot(leo_xyz[0], leo_xyz[1], leo_xyz[2], 'b', label="LEO Orbit")
ax.plot(trans_xyz[0], trans_xyz[1], trans_xyz[2], 'r', label="Earth-Moon Transfer")

# 绘制地球和月球
ax.scatter([0], [0], [0], color='cyan', s=500, label='Earth')
ax.scatter([384400], [0], [0], color='gray', s=100, label='Moon (avg)')

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Earth → Moon Transfer Trajectory (Falcon Heavy 100–150 t payload)')
ax.legend()
plt.show()
