# 导弹-干扰弹-目标系统参数配置

import numpy as np

# 导弹参数
MISSILE_SPEED = 300.0  # m/s

# 导弹初始位置 (x, y, z)
MISSILE_POSITIONS = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}

# 无人机参数
DRONE_SPEED_RANGE = (70.0, 140.0)  # m/s 速度范围

# 无人机初始位置 (x, y, z)
DRONE_POSITIONS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}

# 目标参数
FALSE_TARGET = np.array([0.0, 0.0, 0.0])  # 假目标位置
TRUE_TARGET_CENTER = np.array([0.0, 200.0, 0.0])  # 真目标下底面圆心

# 圆柱形目标参数
CYLINDER_RADIUS = 7.0  # m
CYLINDER_HEIGHT = 10.0  # m

# 物理常数
GRAVITY = 9.80665  # m/s² 重力加速度

# 导弹飞行方向参数
SIN_THETA = 1 / np.sqrt(100 + 1)
COS_THETA = 10 / np.sqrt(100 + 1)

# 干扰弹参数
BOMB_FALL_SPEED = 3.0  # m/s 干扰弹下落速度

# 时间参数
T_RELEASE = 1.5  # s 投放时间
T_EXPLODE = T_RELEASE + 3.6  # s 起爆时间

# 距离阈值
DISTANCE_THRESHOLD = 10.0  # m 距离阈值

# 求解参数
ROOT_FINDING_STEP = 0.1  # 根查找步长
ROOT_FINDING_TOL = 1e-5  # 根查找容差
TIME_RANGE = 20.0  # s 时间搜索范围

# 圆柱体表面采样参数
CYLINDER_SAMPLES = 10000  # 圆柱体表面采样点数量
