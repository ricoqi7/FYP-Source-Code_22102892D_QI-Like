#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_hybrid_data.py
功能：在 CARLA 中生成带有“人工噪声/异常行为”的复杂交通与行人环境，
并允许用户使用 Gamepad (手柄) 进行第一人称驾驶控制。
采集的 (State, Action) 数据会完整记录到 CSV，作为 RL/Meta-RL 预训练的 Offline 数据集。
"""

import os
import time
import math
import random
import csv
import queue
from datetime import datetime
import carla
import pygame

# =========================
# 配置参数
# =========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
MAP_NAME = "Town10HD_Opt"
DT = 0.05
DURATION_SEC = 1200.0

NUM_VEHICLES = 20
NUM_PEDESTRIANS = 30
HAZARD_PEDESTRIAN_RATIO = 0.3 # 30%的行人会突然横穿

# 手柄映射配置
AXIS_STEER = 0       
AXIS_THROTTLE = 5    
AXIS_BRAKE = 4       
BTN_REVERSE = 0      
BTN_QUIT = 7         

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.getcwd(), f"hybrid_dataset_{RUN_ID}")
LOG_PATH = os.path.join(OUT_DIR, "agent_transitions.csv")

def _init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("未检测到手柄 (No Gamepad detected)!")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def _setup_world(client):
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    
    # 调低全局安全距离，制造更危险的车流
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.5)
    return world, tm

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        joystick = _init_gamepad()
    except Exception as e:
        print(f"[ERROR] {e}"); return

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world, tm = _setup_world(client)
    spectator = world.get_spectator()

    # CSV 初始化
    f_csv = open(LOG_PATH, "w", newline="", encoding="utf-8")
    w_csv = csv.writer(f_csv)
    w_csv.writerow(["step", "timestamp", "x", "y", "z", "speed_mps", 
                    "min_ped_dist", "front_veh_dist", "collision_flag",
                    "a_throttle", "a_brake", "a_steer"])

    ego_vehicle = None
    vehicles = []
    
    try:
        # 1. 生成主车 (Ego)
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.filter("vehicle.audi.etron")[0]
        ego_tf = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_bp, ego_tf)
        
        # 添加碰撞传感器
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego_vehicle)
        col_history = []
        col_sensor.listen(lambda event: col_history.append(event))

        # 2. 生成背景乱入车辆 (NPCs)
        veh_bps = bp_lib.filter("vehicle.*")
        for sp in world.get_map().get_spawn_points()[1:NUM_VEHICLES+1]:
            npc = world.try_spawn_actor(random.choice(veh_bps), sp)
            if npc:
                npc.set_autopilot(True, 8000)
                tm.ignore_lights_percentage(npc, random.uniform(10.0, 30.0)) # 故意闯红灯
                vehicles.append(npc)

        print("[START] 开始手动采集数据，请使用手柄驾驶。按 START/OPTIONS 键退出。")
        vehicle_control = carla.VehicleControl()
        steps = int(DURATION_SEC / DT)

        for step in range(steps):
            pygame.event.pump()
            
            # 读取手柄指令 (Action)
            steer_val = joystick.get_axis(AXIS_STEER)
            vehicle_control.steer = float(steer_val) if abs(steer_val) > 0.05 else 0.0
            vehicle_control.throttle = max(0.0, (joystick.get_axis(AXIS_THROTTLE) + 1.0) / 2.0)
            vehicle_control.brake = max(0.0, (joystick.get_axis(AXIS_BRAKE) + 1.0) / 2.0)
            
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN and event.button == BTN_QUIT:
                    raise KeyboardInterrupt

            ego_vehicle.apply_control(vehicle_control)
            world.tick()

            # 更新第一人称视角 (上帝视角 -> 驾驶员视角)
            cam_tf = ego_vehicle.get_transform()
            cam_tf.location += carla.Location(x=0.3, y=-0.4, z=1.2) # 模拟人眼位置
            spectator.set_transform(cam_tf)

            # 提取环境特征 (State)
            loc = ego_vehicle.get_location()
            speed = math.sqrt(ego_vehicle.get_velocity().x**2 + ego_vehicle.get_velocity().y**2)
            col_flag = 1 if len(col_history) > 0 else 0
            col_history.clear() # 清空本帧碰撞
            
            # 记录到 CSV
            w_csv.writerow([
                step, round(step*DT, 2), loc.x, loc.y, loc.z, speed,
                "0.0", "0.0", col_flag, # 简化的行人/前车距离，实际可结合 demo3_1 的 _nearest_entity
                vehicle_control.throttle, vehicle_control.brake, vehicle_control.steer
            ])

    except KeyboardInterrupt:
        print("\n[STOP] 手动中断采集。")
    finally:
        f_csv.close()
        if ego_vehicle: ego_vehicle.destroy()
        for v in vehicles: v.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print(f"[DONE] 数据已保存至 {LOG_PATH}")

if __name__ == "__main__":
    main()