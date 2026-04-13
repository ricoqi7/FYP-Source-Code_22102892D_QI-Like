#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math  # 已修复：添加数学库
import queue
from datetime import datetime

import carla
import pygame

# =========================
# 固定参数
# =========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
MAP_NAME = "Town10HD_Opt"
DT = 0.05
DURATION_SEC = 1200.0 

# =========================
# 采集与相机参数
# =========================
SAVE_RGB = True
SAVE_IMAGE_EVERY_N_STEPS = 5 
CAM_W = 800
CAM_H = 600
CAM_FOV = 90.0
IMAGE_EXT = "png"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.getcwd(), f"manual_dataset_{RUN_ID}")
OUT_IMG_DIR = os.path.join(OUT_DIR, "rgb")

# =========================
# 手柄映射配置
# =========================
AXIS_STEER = 0       
AXIS_THROTTLE = 5    
AXIS_BRAKE = 4       
BTN_REVERSE = 0      
BTN_QUIT = 7         

def _ensure_out_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_RGB:
        os.makedirs(OUT_IMG_DIR, exist_ok=True)

def _setup_carla_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    world.tick()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    return world

def _spawn_ego_vehicle(world: carla.World) -> carla.Vehicle:
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.audi.etron")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] if spawn_points else carla.Transform()
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle!")
    return ego_vehicle

def _attach_rgb_camera(world: carla.World, vehicle: carla.Vehicle):
    bp_lib = world.get_blueprint_library()
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(int(CAM_W)))
    rgb_bp.set_attribute("image_size_y", str(int(CAM_H)))
    rgb_bp.set_attribute("fov", str(float(CAM_FOV)))
    
    # 【已修改】模拟真实驾驶员视角坐标
    # x=0.3 (仪表盘后方), y=-0.4 (左舵驾驶位), z=1.2 (人眼高度)
    cam_tf = carla.Transform(carla.Location(x=0.3, y=-0.4, z=1.2))
    camera = world.spawn_actor(rgb_bp, cam_tf, attach_to=vehicle)
    
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    return camera, image_queue

def _drain_queue_latest(q: queue.Queue):
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest

def _save_image_sample(image: carla.Image):
    if image is None: return
    file_path = os.path.join(OUT_IMG_DIR, f"{int(image.frame):08d}.{IMAGE_EXT}")
    image.save_to_disk(file_path)

def _init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No Gamepad detected!")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def _parse_gamepad_input(joystick, current_control: carla.VehicleControl) -> tuple:
    quit_flag = False
    pygame.event.pump()
    
    steer_val = joystick.get_axis(AXIS_STEER)
    current_control.steer = float(steer_val) if abs(steer_val) > 0.05 else 0.0
    
    throttle_raw = joystick.get_axis(AXIS_THROTTLE)
    brake_raw = joystick.get_axis(AXIS_BRAKE)
    current_control.throttle = max(0.0, (throttle_raw + 1.0) / 2.0)
    current_control.brake = max(0.0, (brake_raw + 1.0) / 2.0)
    
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == BTN_REVERSE:
                current_control.reverse = not current_control.reverse
            elif event.button == BTN_QUIT:
                quit_flag = True
    return current_control, quit_flag

def main():
    _ensure_out_dirs()
    try:
        joystick = _init_gamepad()
    except Exception as e:
        print(f"[ERROR] {e}"); return

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = _setup_carla_world(client)
    spectator = world.get_spectator() # 获取观察者对象

    ego_vehicle = None
    camera = None
    
    try:
        ego_vehicle = _spawn_ego_vehicle(world)
        camera, image_queue = _attach_rgb_camera(world, ego_vehicle)
        
        vehicle_control = carla.VehicleControl()
        steps = int(DURATION_SEC / DT)
        
        for step in range(steps):
            vehicle_control, quit_flag = _parse_gamepad_input(joystick, vehicle_control)
            if quit_flag: break
                
            ego_vehicle.apply_control(vehicle_control)
            world.tick()
            
            # 【新增】更新 CARLA 窗口视角，使其锁定在车内相机位置
            spectator.set_transform(camera.get_transform())
            
            rgb_img = _drain_queue_latest(image_queue)
            if SAVE_RGB and (step % SAVE_IMAGE_EVERY_N_STEPS == 0):
                _save_image_sample(rgb_img)
                
            if step % 100 == 0:
                vel = ego_vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                print(f"Step {step} | Speed: {speed_kmh:.1f} km/h")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if camera is not None:
            camera.stop(); camera.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()

if __name__ == "__main__":
    main()