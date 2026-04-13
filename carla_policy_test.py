#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import subprocess
import numpy as np
import carla
from stable_baselines3 import PPO

# ===================================================
# 1. 配置参数 (需与训练时完全一致)
# ===================================================
MODEL_PATH = "ppo_carla_final_policy.zip"
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
SUMO_PORT = 8813
DT = 0.05

# 基础生成设置
NUM_VEHICLES = 20
NUM_PEDESTRIANS = 30

# ===================================================
# 2. 辅助计算函数 (复用自 demo_v3_1 逻辑)
# ===================================================

def _get_dist(l1, l2):
    return math.sqrt((l1.x - l2.x)**2 + (l1.y - l2.y)**2 + (l1.z - l2.z)**2)

def _nearest_entity(ego_loc, entity_list, exclude_id=None):
    min_dist = 999.0
    speed = 0.0
    target = None
    for entity in entity_list:
        if exclude_id and entity.id == exclude_id: continue
        d = _get_dist(ego_loc, entity.get_location())
        if d < min_dist:
            min_dist = d
            v = entity.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            target = entity
    return target, min_dist, speed

def _count_within_radius(ego_loc, entity_list, radius=10.0, exclude_id=None):
    count = 0
    for entity in entity_list:
        if exclude_id and entity.id == exclude_id: continue
        if _get_dist(ego_loc, entity.get_location()) <= radius:
            count += 1
    return count

def get_trained_observation(vehicle, world):
    """
    【核心】12维特征提取器，必须与训练时完全对齐
    """
    # 获取实时 Actor 列表
    all_actors = world.get_actors()
    vehicles = all_actors.filter('vehicle.*')
    walkers = all_actors.filter('walker.pedestrian.*')
    t_lights = all_actors.filter('traffic.traffic_light*')

    # 1. 基础物理状态
    ego_loc = vehicle.get_location()
    v = vehicle.get_velocity()
    speed_mps = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    limit_mps = vehicle.get_speed_limit() / 3.6 if vehicle.get_speed_limit() > 0 else 10.0
    
    # 2. 周边障碍物探测
    _, min_ped_dist, min_ped_speed = _nearest_entity(ego_loc, walkers)
    _, min_veh_dist, min_veh_speed = _nearest_entity(ego_loc, vehicles, exclude_id=vehicle.id)
    
    # 3. 交通灯状态 (简化逻辑)
    tl_state_val = 1.0 # 默认绿灯
    tl_dist = 99.0
    if vehicle.is_at_traffic_light():
        tl = vehicle.get_traffic_light()
        if tl.get_state() == carla.TrafficLightState.Red: tl_state_val = 0.0
        elif tl.get_state() == carla.TrafficLightState.Yellow: tl_state_val = 0.5
        tl_dist = 0.0 # 已经在灯下

    # 4. 密度计算
    num_ped_r10 = _count_within_radius(ego_loc, walkers, radius=10.0)
    num_veh_r15 = _count_within_radius(ego_loc, vehicles, radius=15.0, exclude_id=vehicle.id)

    # 构建 12 维向量 (Normalization)
    obs = np.array([
        speed_mps / limit_mps,                      # 0: 速度比例
        1.0 / (min_veh_dist + 1.0),                 # 1: 最近车倒数 (近似前车)
        1.0 / (min_ped_dist + 1.0),                 # 2: 最近行人倒数
        min_ped_speed / 5.0,                        # 3: 行人速度
        1.0 / (min_veh_dist + 1.0),                 # 4: 周边车倒数
        min_veh_speed / 15.0,                       # 5: 周边车速度
        tl_state_val,                               # 6: 灯光状态
        1.0 / (tl_dist + 1.0),                      # 7: 灯光距离倒数
        num_ped_r10 / 10.0,                         # 8: 行人密度
        num_veh_r15 / 10.0,                         # 9: 车辆密度
        vehicle.get_angular_velocity().z / 1.0,     # 10: 航向角速度
        limit_mps / 30.0                            # 11: 限速背景
    ], dtype=np.float32)
    
    return obs

# ===================================================
# 3. 主测试程序
# ===================================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}"); return

    # 加载 PPO 模型
    print(f"[TEST] 正在加载策略模型...")
    model = PPO.load(MODEL_PATH)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    
    sumo_proc = None
    ego_vehicle = None
    actors_to_clean = []

    try:
        world = client.get_world()
        
        # 1. 开启 SUMO 桥接 (如果环境需要)
        print("[TEST] 启动 SUMO 协同仿真...")
        sumo_script = os.path.join(os.environ.get('CARLA_ROOT', ''), "Co-Simulation/Sumo/run_synchronization.py")
        sumo_proc = subprocess.Popen(["python", sumo_script, "Town10HD.sumocfg", "--sumo-port", str(SUMO_PORT)])
        time.sleep(3.0)

        # 2. 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        # 3. 生成主车 (Ego Vehicle)
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        actors_to_clean.append(ego_vehicle)
        
        spectator = world.get_spectator()

        print("[START] 策略接管成功。观察 AI 驾驶行为...")

        while True:
            # A. 提取 12 维特征
            obs = get_trained_observation(ego_vehicle, world)
            
            # B. 模型推理 (使用确定性策略)
            action, _ = model.predict(obs, deterministic=True)
            
            # C. 应用动作
            control = carla.VehicleControl()
            # 动作映射逻辑与训练时保持一致
            control.throttle = float(np.clip(action[0], 0.0, 1.0))
            control.brake = float(np.clip(-action[0], 0.0, 1.0))
            control.steer = float(action[1])
            ego_vehicle.apply_control(control)

            # D. 更新视角
            ego_tf = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                ego_tf.location + carla.Location(z=2.8) - ego_tf.get_forward_vector() * 6,
                ego_tf.rotation
            ))

            world.tick()

    except KeyboardInterrupt:
        print("\n[STOP] 测试手动停止。")
    finally:
        # 清理环境
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actors_to_clean:
            if actor.is_alive: actor.destroy()
            
        if sumo_proc:
            sumo_proc.terminate()
        print("[DONE] 资源已释放。")

if __name__ == "__main__":
    main()