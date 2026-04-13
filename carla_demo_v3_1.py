#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
carla_demo_v2_0.py（在 carla_demo.py 基础上“尽量少改动”的可采集数据版本，已按你的新需求微调）

你提出的改动点（本版本已处理）：
1) 将车辆/行人数量降低为：NUM_VEHICLES=20, NUM_PEDESTRIANS=30（减轻运算与日志体积）。
2) depth/semantic 文件夹为空：
   - 原因：默认 SAVE_DEPTH=False、SAVE_SEMANTIC=False，因此脚本不会生成这些传感器数据，但旧版仍会创建文件夹。
   - 处理：本版本仅在对应开关为 True 时才创建 depth/semantic 子目录，并在启动时打印开关状态。
3) 不再保存每一帧相机图片：
   - 处理：新增 SAVE_IMAGE_EVERY_N_STEPS（默认每 20 ticks 保存一帧=1Hz）
   - 传感器读取改为“非阻塞 drain 队列”，避免每步阻塞导致不稳定/变慢。
4) 强化 logs/agent_transitions.csv，使其更适合作为后续 RL 训练的数据来源：
   - 在 (x,y,z,yaw,speed,control,collision,reward,done) 基础上增加低维上下文特征（最近行人/车辆/前车/密度/红绿灯等）

运行方式（Windows PowerShell）：
cd C:\\FYP\\CARLA_0.9.16
.\\carla_env312\\Scripts\\activate
.\\CarlaUE4.exe
# 等 UE4 启动并完成加载后，在另一个 PowerShell 窗口运行：
python carla_demo_v2_0.py

注意：
- 仍然保持 SUMO-GUI “并行展示”的策略：不启 TraCI 同步，不做 SUMO→CARLA actor 映射（与原 demo 保持一致）。
"""

import os
import sys
import time
import math
import random
import subprocess
import csv
import queue
from datetime import datetime

import carla

# =========================
# 固定参数（与原 demo 一致，只调整数量）
# =========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
MAP_NAME = "Town10HD_Opt"

NUM_VEHICLES = 30
NUM_PEDESTRIANS = 50

DURATION_SEC = 1200.0
DT = 0.05

# SUMO 文件输出目录（保持一致）
SUMO_OUT_DIR = os.path.join(os.getcwd(), "sumo_demo_Town10HD_Opt")
XODR_PATH = os.path.join(SUMO_OUT_DIR, f"{MAP_NAME}.xodr")
NET_PATH  = os.path.join(SUMO_OUT_DIR, f"{MAP_NAME}.net.xml")
ROU_PATH  = os.path.join(SUMO_OUT_DIR, f"{MAP_NAME}.rou.xml")

# =========================
# 采集相关参数
# =========================
AGENT_INDEX = 0  # vehicles 列表中的第几个作为 agent（0 = 第一辆）

# =========================
# 风险数据采集参数（新增）
# =========================
# 目的：故意生成更危险的初始数据，让后续 RL 能学到“避让异常行人”
AGGRESSIVE_VEHICLE_RATIO = 0.45   # 约 45% NPC 更激进
TM_GLOBAL_MIN_GAP = 0.8           # 全局最小跟车距离（越小越危险）
TM_GLOBAL_SPEED_DIFF = -20.0      # 负值=比限速更快，增加风险
TM_IGNORE_LIGHTS_PCT = 35.0       # 一部分车忽略红灯
TM_IGNORE_SIGNS_PCT = 20.0        # 一部分车忽略路牌
TM_RANDOM_LANECHANGE_PCT = 35.0   # 随机换道概率

# agent 也故意略激进，避免采到全是“安全巡航”数据
AGENT_EXTRA_AGGRESSIVE = True

# 异常行人：不走 navmesh 规则路线，而是“手动横穿”
HAZARD_PEDESTRIAN_RATIO = 0.25    # 25% 行人作为 hazard walkers
HAZARD_TRIGGER_MIN_SEC = 2.0
HAZARD_TRIGGER_MAX_SEC = 5.0
HAZARD_AHEAD_MIN = 8.0            # 在 agent 前方 8~16m 触发横穿
HAZARD_AHEAD_MAX = 16.0
HAZARD_CROSS_SPEED_MIN = 2.0
HAZARD_CROSS_SPEED_MAX = 4.2
HAZARD_ACTIVE_MIN_SEC = 2.5
HAZARD_ACTIVE_MAX_SEC = 4.5

# 传感器开关：RGB 默认仅用于“验证第一人称管线”，后续训练主要用 CSV 特征
SAVE_RGB = True
SAVE_DEPTH = True #默认是False，手动调节
SAVE_SEMANTIC = True #同上

# 摄像头参数
CAM_W = 800
CAM_H = 600
CAM_FOV = 90.0
IMAGE_EXT = "png"

# 图像保存频率：不再每帧保存（默认每 50 ticks 保存 1 帧 = 1Hz）
SAVE_IMAGE_EVERY_N_STEPS = 50
SAVE_IMAGES = True  # 关闭则完全不落盘图像（仅保留 CSV 特征）

# 全局状态采样频率：1 = 每 tick 写入；想降文件体积可设 5/10
LOG_GLOBAL_EVERY_N_STEPS = 1

# 行人“卡住”检测：降低重置频率，减少 NAV 警告刷屏
STUCK_CHECK_INTERVAL_SEC = 2.0
STUCK_DIST_EPS = 0.25
STUCK_REPLAN_AFTER_N_CHECKS = 3  # 连续 N 次检测到几乎没移动才重规划（默认 3*2s=6s）

# 输出目录：每次运行一个新文件夹，避免覆盖
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.getcwd(), f"demo_dataset_v3_1_{RUN_ID}")
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_LOG_DIR = os.path.join(OUT_DIR, "logs")


def _find_sumo_bin(exe_name: str) -> str:
    """尽量找到 SUMO 可执行文件；优先 SUMO_HOME，否则依赖 PATH。"""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = os.path.join(sumo_home, "bin", exe_name)
        if os.path.exists(candidate):
            return candidate
    return exe_name


def _export_xodr_from_carla(world: carla.World, out_path: str):
    """从 CARLA 当前地图导出 OpenDRIVE(.xodr)。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return
    print(f"[SUMO] Export OpenDRIVE: {out_path}")
    xodr = world.get_map().to_opendrive()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xodr)


def _netconvert_xodr_to_net(xodr_path: str, net_path: str):
    """用 netconvert 将 xodr 转成 net.xml（最小参数，避免版本不兼容）。"""
    if os.path.exists(net_path) and os.path.getsize(net_path) > 1000:
        return

    os.makedirs(os.path.dirname(net_path), exist_ok=True)
    netconvert = _find_sumo_bin("netconvert.exe")
    if netconvert == "netconvert.exe":
        netconvert = _find_sumo_bin("netconvert")

    cmd = [
        netconvert,
        "--opendrive-files", xodr_path,
        "--output-file", net_path,
        "--ignore-errors",
    ]
    print("[SUMO] netconvert -> net.xml")
    print("[netconvert]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[SUMO] net.xml ready: {net_path}")


def _generate_routes_with_randomTrips(net_path: str, rou_path: str, num_veh: int, sim_time: float):
    """用 SUMO tools/randomTrips.py 生成 rou.xml。"""
    if os.path.exists(rou_path) and os.path.getsize(rou_path) > 200:
        return

    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise RuntimeError("未设置 SUMO_HOME，无法定位 randomTrips.py。")

    randomTrips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(randomTrips):
        raise RuntimeError(f"找不到 randomTrips.py: {randomTrips}")

    period = max(0.5, float(sim_time) / max(1, int(num_veh)))

    cmd = [
        sys.executable, randomTrips,
        "-n", net_path,
        "-e", str(float(sim_time)),
        "-p", str(period),
        "--route-file", rou_path,
        "--seed", "42",
        "--validate",
    ]
    print("[SUMO] randomTrips -> rou.xml")
    print("[randomTrips]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[SUMO] rou.xml ready: {rou_path}")


def _start_sumo_gui(net_path: str, rou_path: str, dt: float, sim_time: float) -> subprocess.Popen:
    """只负责弹出 SUMO-GUI 并跑起来（不启 TraCI，同原 demo 一致）。"""
    sumo_gui = _find_sumo_bin("sumo-gui.exe")
    if sumo_gui == "sumo-gui.exe":
        sumo_gui = _find_sumo_bin("sumo-gui")

    cmd = [
        sumo_gui,
        "-n", net_path,
        "-r", rou_path,
        "--step-length", str(float(dt)),
        "--start",
        "--begin", "0",
        "--end", str(float(sim_time)),
        "--no-step-log", "true",
        "--quit-on-end",
    ]
    print("[SUMO] Starting:", " ".join(cmd))
    return subprocess.Popen(cmd)


def _setup_carla_world(client: carla.Client) -> carla.World:
    """加载地图并设置同步模式（更稳定）。"""
    world = client.load_world(MAP_NAME)
    world.tick()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.global_percentage_speed_difference(0.0)

    print(f"[CARLA] World: {world.get_map().name.split('/')[-1]}")
    return world


def _spawn_vehicles(world: carla.World, client: carla.Client, num: int):
    """用 CARLA Traffic Manager 生成并驱动车辆（不依赖 SUMO）。"""
    blueprints = world.get_blueprint_library()
    vehicle_bps = blueprints.filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()

    vehicles = []
    count = 0
    for sp in spawn_points:
        if count >= num:
            break
        bp = random.choice(vehicle_bps)

        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels").as_int()) < 4:
                    continue
            except Exception:
                pass

        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue

        actor.set_autopilot(True, tm_port)
        vehicles.append(actor)
        count += 1

    print(f"[CARLA] Spawned vehicles: {len(vehicles)}/{num}")
    return vehicles


def _configure_tm_for_risky_data(client: carla.Client, vehicles, agent_vehicle=None):
    """
    相比 v3 原版，保留“更危险的数据分布”，但降低到“可长时间稳定运行”的激进程度。
    重点是产生 near-miss / 少量碰撞，而不是把 TM 调到容易把仿真拖崩。
    """
    tm = client.get_trafficmanager(8000)

    # 全局只做温和扰动，不再使用过激参数
    tm.set_global_distance_to_leading_vehicle(max(1.0, TM_GLOBAL_MIN_GAP + 0.4))
    tm.global_percentage_speed_difference(max(TM_GLOBAL_SPEED_DIFF, -10.0))

    for v in vehicles:
        try:
            tm.auto_lane_change(v, True)
            if random.random() < AGGRESSIVE_VEHICLE_RATIO:
                tm.vehicle_percentage_speed_difference(v, random.uniform(-15.0, -5.0))
                tm.distance_to_leading_vehicle(v, random.uniform(0.8, 1.5))
                tm.ignore_lights_percentage(v, random.uniform(5.0, 25.0))
                tm.ignore_signs_percentage(v, random.uniform(0.0, 15.0))
                tm.random_left_lanechange_percentage(v, random.uniform(10.0, 30.0))
                tm.random_right_lanechange_percentage(v, random.uniform(10.0, 30.0))
        except Exception:
            pass

    if agent_vehicle is not None and AGENT_EXTRA_AGGRESSIVE:
        try:
            tm.vehicle_percentage_speed_difference(agent_vehicle, random.uniform(-10.0, -3.0))
            tm.distance_to_leading_vehicle(agent_vehicle, random.uniform(0.8, 1.4))
            # agent 仍然可更激进，但不再高概率闯灯，避免仿真中途失稳
            tm.ignore_lights_percentage(agent_vehicle, random.uniform(0.0, 8.0))
        except Exception:
            pass

def _build_nav_location_pool(world: carla.World, target_count: int, min_dist: float = 8.0, max_tries: int = 5000):
    """构建“可行走位置池”，强制去重并过滤原点附近，避免行人集中到(0,0)。"""
    pool = []
    tries = 0
    while len(pool) < target_count and tries < max_tries:
        tries += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if abs(loc.x) < 1.0 and abs(loc.y) < 1.0:
            continue
        ok = True
        for p in pool:
            dx = loc.x - p.x
            dy = loc.y - p.y
            if (dx * dx + dy * dy) < (min_dist * min_dist):
                ok = False
                break
        if ok:
            pool.append(loc)
    return pool


def _spawn_pedestrians(world: carla.World, num: int):
    """生成行人 + AI controller，并下发初始目的地与速度。"""
    blueprints = world.get_blueprint_library()
    walker_bps = blueprints.filter("walker.pedestrian.*")
    controller_bp = blueprints.find("controller.ai.walker")

    world.tick()

    spawn_pool = _build_nav_location_pool(world, target_count=num * 2, min_dist=10.0)
    dest_pool  = _build_nav_location_pool(world, target_count=num * 3, min_dist=12.0)

    random.shuffle(spawn_pool)
    random.shuffle(dest_pool)

    walkers = []
    controllers = []

    for _ in range(num):
        if not spawn_pool:
            break

        loc = spawn_pool.pop()
        loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.2)
        tf = carla.Transform(loc, carla.Rotation(yaw=random.uniform(0, 360)))

        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, tf)
        if walker is None:
            continue

        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if ctrl is None:
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(ctrl)

    world.tick()

    for idx, ctrl in enumerate(controllers):
        ctrl.start()
        ctrl.set_max_speed(random.uniform(1.2, 2.0))

        wloc = walkers[idx].get_location()
        chosen = None
        for _ in range(15):
            if not dest_pool:
                break
            cand = dest_pool.pop()
            if (cand.x - wloc.x) ** 2 + (cand.y - wloc.y) ** 2 > (25.0 ** 2):
                chosen = cand
                break
        if chosen is None:
            chosen = world.get_random_location_from_navigation()

        if chosen is not None:
            ctrl.go_to_location(chosen)

    print(f"[CARLA] Spawned pedestrians: {len(walkers)}/{num}")
    return walkers, controllers

def _spawn_hazard_walkers(world: carla.World, num: int): #新增，行人混乱度
    """
    生成一批“异常行人”：
    - 不使用 WalkerAI controller
    - 平时静止
    - 在主循环里被随机触发，从 agent 前方横穿道路
    """
    blueprints = world.get_blueprint_library()
    walker_bps = blueprints.filter("walker.pedestrian.*")
    spawn_points = world.get_map().get_spawn_points()

    hazard_walkers = []

    for _ in range(num):
        if not spawn_points:
            break

        base_tf = random.choice(spawn_points)
        wp = world.get_map().get_waypoint(
            base_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue

        right = wp.transform.get_right_vector()
        side = random.choice([-1.0, 1.0])
        lateral = max(2.0, wp.lane_width * 0.9)

        spawn_loc = carla.Location(
            x=wp.transform.location.x + right.x * side * lateral,
            y=wp.transform.location.y + right.y * side * lateral,
            z=wp.transform.location.z + 0.2
        )

        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(
            walker_bp,
            carla.Transform(spawn_loc, wp.transform.rotation)
        )
        if walker is None:
            continue

        # 初始静止
        walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0))
        hazard_walkers.append({
            "actor": walker,
            "active": False,
            "ttl": 0.0,
            "status": "hazard_idle"
        })

    print(f"[CARLA] Spawned hazard pedestrians: {len(hazard_walkers)}/{num}")
    return hazard_walkers

def _trigger_hazard_crossing(world: carla.World, agent_vehicle: carla.Vehicle, hazard_record: dict):
    """
    让 hazard walker 横穿，但避免过近瞬移与无效 actor 导致的中途异常。
    相比 v3 原版，这里做了两件事：
    1) 只在更远的前方触发；
    2) 先检查 actor 是否有效、目标点是否合理。
    """
    actor = hazard_record.get("actor", None)
    if not _actor_is_alive(actor) or (agent_vehicle is None):
        return False

    try:
        ego_tf = agent_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        ahead = random.uniform(max(12.0, HAZARD_AHEAD_MIN), max(18.0, HAZARD_AHEAD_MAX))
        side = random.choice([-1.0, 1.0])
        lateral = random.uniform(2.8, 4.2)

        spawn_loc = carla.Location(
            x=ego_loc.x + fwd.x * ahead + right.x * side * lateral,
            y=ego_loc.y + fwd.y * ahead + right.y * side * lateral,
            z=ego_loc.z + 0.3
        )

        # 横穿方向：从一侧走到另一侧
        cross_dir = carla.Vector3D(x=-right.x * side, y=-right.y * side, z=0.0)
        norm = math.sqrt(cross_dir.x * cross_dir.x + cross_dir.y * cross_dir.y)
        if norm < 1e-6:
            return False
        cross_dir.x /= norm
        cross_dir.y /= norm

        actor.set_transform(carla.Transform(spawn_loc, ego_tf.rotation))
        actor.apply_control(
            carla.WalkerControl(
                direction=cross_dir,
                speed=random.uniform(HAZARD_CROSS_SPEED_MIN, min(3.2, HAZARD_CROSS_SPEED_MAX)),
                jump=False
            )
        )
        hazard_record["active"] = True
        hazard_record["ttl"] = random.uniform(HAZARD_ACTIVE_MIN_SEC, min(3.5, HAZARD_ACTIVE_MAX_SEC))
        hazard_record["status"] = "hazard_crossing"
        return True
    except Exception:
        hazard_record["active"] = False
        hazard_record["status"] = "hazard_idle"
        return False


def _update_hazard_walkers(hazard_walkers, dt: float):
    """
    hazard walker 触发后持续横穿几秒，然后停下等待下一次事件。
    对失效 actor 做保护，避免因为 actor 生命周期异常导致主循环中断。
    """
    for rec in hazard_walkers:
        actor = rec.get("actor", None)
        if not _actor_is_alive(actor):
            rec["active"] = False
            rec["status"] = "hazard_invalid"
            continue

        if rec.get("active", False):
            rec["ttl"] -= dt
            if rec["ttl"] <= 0.0:
                try:
                    actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0))
                except Exception:
                    pass
                rec["active"] = False
                rec["status"] = "hazard_idle"


# =========================
# 输出目录与低层工具
# =========================
def _ensure_out_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LOG_DIR, exist_ok=True)

    # 仅在开启“保存图像”且对应传感器开启时创建目录，避免出现空文件夹引起误判
    if SAVE_IMAGES and SAVE_RGB:
        os.makedirs(os.path.join(OUT_IMG_DIR, "rgb"), exist_ok=True)
    if SAVE_IMAGES and SAVE_DEPTH:
        os.makedirs(os.path.join(OUT_IMG_DIR, "depth"), exist_ok=True)
    if SAVE_IMAGES and SAVE_SEMANTIC:
        os.makedirs(os.path.join(OUT_IMG_DIR, "semantic"), exist_ok=True)


def _vec3d_norm(v: carla.Vector3D) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _actor_is_alive(actor) -> bool:
    try:
        return actor is not None and actor.is_alive
    except Exception:
        return False


def _safe_actor_location(actor):
    try:
        return actor.get_location()
    except Exception:
        return None


def _safe_actor_speed(actor) -> float:
    try:
        return _vec3d_norm(actor.get_velocity())
    except Exception:
        return 0.0


def _get_live_pedestrians(walkers, hazard_walkers):
    peds = []
    for p in walkers:
        if _actor_is_alive(p):
            peds.append(p)
    for rec in hazard_walkers:
        actor = rec.get("actor", None)
        if _actor_is_alive(actor):
            peds.append(actor)
    return peds


def _drain_queue_latest(q: "queue.Queue"):
    """非阻塞地把队列全部取空，只保留最新一条（避免积压导致内存增长）。"""
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


def _save_image_sample(image: carla.Image, subdir: str, color_converter=None):
    """保存图像到磁盘，并返回 file_path。"""
    if image is None:
        return ""
    img_dir = os.path.join(OUT_IMG_DIR, subdir)
    file_path = os.path.join(img_dir, f"{int(image.frame):08d}.{IMAGE_EXT}")
    if color_converter is None:
        image.save_to_disk(file_path)
    else:
        image.save_to_disk(file_path, color_converter)
    return file_path


# =========================
# Agent 传感器 + 训练特征
# =========================
def _attach_agent_sensors(world: carla.World, agent_vehicle: carla.Vehicle):
    """给 agent_vehicle 挂载 RGB/Depth/Semantic 相机（可选）+ Collision sensor（必）。"""
    bp_lib = world.get_blueprint_library()

    sensors = {}
    qmap = {}

    cam_tf = carla.Transform(carla.Location(x=0.8, z=1.7))

    if SAVE_RGB:
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(int(CAM_W)))
        rgb_bp.set_attribute("image_size_y", str(int(CAM_H)))
        rgb_bp.set_attribute("fov", str(float(CAM_FOV)))
        rgb_bp.set_attribute("sensor_tick", str(float(DT)))
        rgb = world.spawn_actor(rgb_bp, cam_tf, attach_to=agent_vehicle)
        q_rgb = queue.Queue()
        rgb.listen(q_rgb.put)
        sensors["rgb"] = rgb
        qmap["rgb"] = q_rgb

    if SAVE_DEPTH:
        dep_bp = bp_lib.find("sensor.camera.depth")
        dep_bp.set_attribute("image_size_x", str(int(CAM_W)))
        dep_bp.set_attribute("image_size_y", str(int(CAM_H)))
        dep_bp.set_attribute("fov", str(float(CAM_FOV)))
        dep_bp.set_attribute("sensor_tick", str(float(DT)))
        dep = world.spawn_actor(dep_bp, cam_tf, attach_to=agent_vehicle)
        q_dep = queue.Queue()
        dep.listen(q_dep.put)
        sensors["depth"] = dep
        qmap["depth"] = q_dep

    if SAVE_SEMANTIC:
        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(int(CAM_W)))
        sem_bp.set_attribute("image_size_y", str(int(CAM_H)))
        sem_bp.set_attribute("fov", str(float(CAM_FOV)))
        sem_bp.set_attribute("sensor_tick", str(float(DT)))
        sem = world.spawn_actor(sem_bp, cam_tf, attach_to=agent_vehicle)
        q_sem = queue.Queue()
        sem.listen(q_sem.put)
        sensors["semantic"] = sem
        qmap["semantic"] = q_sem

    col_bp = bp_lib.find("sensor.other.collision")
    col = world.spawn_actor(col_bp, carla.Transform(), attach_to=agent_vehicle)
    collision_state = {
        "last_frame": -1,
        "last_other_id": -1,
        "last_other_type": "",
        "count": 0,
    }

    def _on_collision(event: carla.CollisionEvent):
        collision_state["last_frame"] = int(event.frame)
        try:
            collision_state["last_other_id"] = int(event.other_actor.id)
            collision_state["last_other_type"] = str(event.other_actor.type_id)
        except Exception:
            collision_state["last_other_id"] = -1
            collision_state["last_other_type"] = ""
        collision_state["count"] += 1

    col.listen(_on_collision)
    sensors["collision"] = col

    return sensors, qmap, collision_state


def _traffic_light_state_str(state) -> str:
    try:
        return str(state).split(".")[-1]
    except Exception:
        return str(state)


def _nearest_entity(ego_loc: carla.Location, entities, exclude_id: int = -1):
    """返回 (min_id, min_dist, min_speed)。entities 可为车辆或行人 actor 列表。"""
    min_dist = float("inf")
    min_id = -1
    min_speed = 0.0
    for a in entities:
        try:
            if exclude_id != -1 and a.id == exclude_id:
                continue
            loc = a.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            dz = loc.z - ego_loc.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < min_dist:
                min_dist = dist
                min_id = int(a.id)
                min_speed = _vec3d_norm(a.get_velocity())
        except Exception:
            continue
    if min_id == -1:
        return -1, float("inf"), 0.0
    return -1 if min_id is None else min_id, float(min_dist), float(min_speed)


def _nearest_front_vehicle(agent_vehicle: carla.Vehicle, vehicles, max_lat: float = 3.0):
    """粗略取“前方同向车辆”距离。"""
    ego_tf = agent_vehicle.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()

    best_dist = float("inf")
    best_id = -1

    for v in vehicles:
        if v.id == agent_vehicle.id:
            continue
        loc = v.get_location()
        rx = loc.x - ego_loc.x
        ry = loc.y - ego_loc.y
        rz = loc.z - ego_loc.z
        dot = rx*fwd.x + ry*fwd.y + rz*fwd.z
        if dot <= 0:
            continue
        dist = math.sqrt(rx*rx + ry*ry + rz*rz)
        lateral_sq = max(dist*dist - dot*dot, 0.0)
        lateral = math.sqrt(lateral_sq)
        if lateral > max_lat:
            continue
        if dist < best_dist:
            best_dist = dist
            best_id = int(v.id)

    if best_id == -1:
        return -1, float("inf")
    return best_id, float(best_dist)


def _count_within_radius(ego_loc: carla.Location, entities, radius: float, exclude_id: int = -1) -> int:
    r2 = radius * radius
    c = 0
    for a in entities:
        try:
            if exclude_id != -1 and a.id == exclude_id:
                continue
            loc = a.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            dz = loc.z - ego_loc.z
            if (dx*dx + dy*dy + dz*dz) <= r2:
                c += 1
        except Exception:
            continue
    return int(c)


def _nearest_traffic_light(ego_loc: carla.Location, tls) -> tuple:
    """返回 (tl_id, tl_state, tl_dist)。"""
    best_dist = float("inf")
    best_id = -1
    best_state = "None"
    for tl in tls:
        try:
            loc = tl.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            dz = loc.z - ego_loc.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < best_dist:
                best_dist = dist
                best_id = int(tl.id)
                best_state = _traffic_light_state_str(tl.get_state())
        except Exception:
            continue
    if best_id == -1:
        return -1, "None", float("inf")
    return best_id, best_state, float(best_dist)


def _compute_reward(speed_mps: float, control: carla.VehicleControl, collision_flag: int, min_ped_dist: float) -> float:
    """最小可用 reward + 轻量近行人惩罚（更贴近 safety）。"""
    r_collision = -100.0 * float(collision_flag)
    r_near_ped = -10.0 * (2.0 - float(min_ped_dist)) if min_ped_dist < 2.0 else 0.0
    r_speed = 0.05 * float(speed_mps)
    r_comfort = -0.5 * abs(float(control.steer)) - 0.2 * float(control.brake)
    return r_collision + r_near_ped + r_speed + r_comfort


def main():
    _ensure_out_dirs()

    print(f"[CFG] NUM_VEHICLES={NUM_VEHICLES}, NUM_PEDESTRIANS={NUM_PEDESTRIANS}, DT={DT}, DURATION_SEC={DURATION_SEC}")
    print(f"[CFG] SAVE_RGB={SAVE_RGB}, SAVE_DEPTH={SAVE_DEPTH}, SAVE_SEMANTIC={SAVE_SEMANTIC}")
    print(f"[CFG] SAVE_IMAGES={SAVE_IMAGES}, SAVE_IMAGE_EVERY_N_STEPS={SAVE_IMAGE_EVERY_N_STEPS}")

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = _setup_carla_world(client)

    # SUMO-GUI（独立展示用）
    try:
        _export_xodr_from_carla(world, XODR_PATH)
        _netconvert_xodr_to_net(XODR_PATH, NET_PATH)
        _generate_routes_with_randomTrips(NET_PATH, ROU_PATH, NUM_VEHICLES, DURATION_SEC)
        sumo_proc = _start_sumo_gui(NET_PATH, ROU_PATH, DT, DURATION_SEC)
    except Exception as e:
        sumo_proc = None
        print("[WARN] SUMO 启动失败，但 CARLA demo 继续运行。原因：", e)

    #vehicles = _spawn_vehicles(world, client, NUM_VEHICLES)
    #walkers, controllers = _spawn_pedestrians(world, NUM_PEDESTRIANS)


    # 全新的车辆、行人逻辑
    # 2) 车辆 
    vehicles = _spawn_vehicles(world, client, NUM_VEHICLES)

    # 3) 先选 agent
    agent_vehicle = None
    if vehicles:
        idx = min(max(0, int(AGENT_INDEX)), len(vehicles) - 1)
        agent_vehicle = vehicles[idx]
        print(f"[AGENT] Using vehicle id={agent_vehicle.id}, type={agent_vehicle.type_id}")
    else:
        print("[WARN] 没有成功生成车辆，agent 为空。")

    # 4) 把 Traffic Manager 调激进
    _configure_tm_for_risky_data(client, vehicles, agent_vehicle)

    # 5) 行人拆成两部分：
    #    - 正常行人：继续走 navmesh
    #    - 异常行人：由脚本手动横穿
    num_hazard = max(2, int(NUM_PEDESTRIANS * HAZARD_PEDESTRIAN_RATIO))
    num_normal = max(0, NUM_PEDESTRIANS - num_hazard)

    walkers, controllers = _spawn_pedestrians(world, num_normal)
    hazard_walkers = _spawn_hazard_walkers(world, num_hazard)

    all_walkers = walkers + [rec["actor"] for rec in hazard_walkers if rec["actor"] is not None]


    # 选定 agent (旧版本，删去)
    '''agent_vehicle = None
    if vehicles:
        idx = min(max(0, int(AGENT_INDEX)), len(vehicles) - 1)
        agent_vehicle = vehicles[idx]
        print(f"[AGENT] Using vehicle id={agent_vehicle.id}, type={agent_vehicle.type_id}")
    else:
        print("[WARN] 没有成功生成车辆，agent 为空。将只运行行人 + SUMO 展示。")'''

    agent_sensors = {}
    agent_qmap = {}
    collision_state = None
    if agent_vehicle is not None:
        agent_sensors, agent_qmap, collision_state = _attach_agent_sensors(world, agent_vehicle)

    # CSV 输出
    trans_path = os.path.join(OUT_LOG_DIR, "agent_transitions.csv")
    f_trans = open(trans_path, "w", newline="", encoding="utf-8")
    w_trans = csv.writer(f_trans)
    w_trans.writerow([
        "step", "frame", "timestamp", "vehicle_id",
        "o_rgb", "o_depth", "o_semantic",
        "x", "y", "z", "yaw", "pitch", "roll", "speed_mps",
        "min_ped_id", "min_ped_dist", "min_ped_speed",
        "min_veh_id", "min_veh_dist", "min_veh_speed",
        "front_veh_id", "front_veh_dist",
        "num_ped_r10", "num_veh_r15",
        "nearest_tl_id", "nearest_tl_state", "nearest_tl_dist",
        "at_tl_flag", "at_tl_id", "at_tl_state",
        "speed_limit",
        "a_throttle", "a_brake", "a_steer",
        "collision_flag", "collision_other_id", "collision_other_type",
        "reward", "done",
        "op_rgb", "op_depth", "op_semantic",
        "xp", "yp", "zp", "yawp", "pitchp", "rollp", "speedp_mps",
        "min_ped_id_p", "min_ped_dist_p", "min_ped_speed_p",
        "min_veh_id_p", "min_veh_dist_p", "min_veh_speed_p",
        "front_veh_id_p", "front_veh_dist_p",
        "num_ped_r10_p", "num_veh_r15_p",
        "nearest_tl_id_p", "nearest_tl_state_p", "nearest_tl_dist_p",
        "at_tl_flag_p", "at_tl_id_p", "at_tl_state_p",
        "speed_limit_p",
    ])

    veh_path = os.path.join(OUT_LOG_DIR, "global_vehicles.csv")
    f_veh = open(veh_path, "w", newline="", encoding="utf-8")
    w_veh = csv.writer(f_veh)
    w_veh.writerow(["step", "frame", "timestamp", "vehicle_id", "x", "y", "z", "yaw", "pitch", "roll", "speed_mps"])

    ped_path = os.path.join(OUT_LOG_DIR, "global_pedestrians.csv")
    f_ped = open(ped_path, "w", newline="", encoding="utf-8")
    w_ped = csv.writer(f_ped)
    w_ped.writerow(["step", "frame", "timestamp", "pedestrian_id", "x", "y", "z", "speed_mps", "status"])

    tl_path = os.path.join(OUT_LOG_DIR, "traffic_lights.csv")
    f_tl = open(tl_path, "w", newline="", encoding="utf-8")
    w_tl = csv.writer(f_tl)
    w_tl.writerow(["step", "frame", "timestamp", "traffic_light_id", "state"])

    print(f"[DATA] Output directory: {OUT_DIR}")
    print(f"[DATA] agent_transitions.csv -> {trans_path}")
    print(f"[DATA] global_vehicles.csv   -> {veh_path}")
    print(f"[DATA] global_pedestrians.csv-> {ped_path}")
    print(f"[DATA] traffic_lights.csv    -> {tl_path}")

    steps = int(DURATION_SEC / DT)
    check_every = max(1, int(STUCK_CHECK_INTERVAL_SEC / DT))
    last_pos = {w.id: w.get_location() for w in walkers}
    stuck_counts = {w.id: 0 for w in walkers}

    prev = None
    flush_every = 200

    # hazard 触发步数不要在循环体内临时创建，避免后续修改时引入未定义/状态漂移问题
    next_hazard_step = random.randint(
        int(HAZARD_TRIGGER_MIN_SEC / DT),
        int(HAZARD_TRIGGER_MAX_SEC / DT)
    )

    print("[DEMO] Running... (CARLA vehicles moving; pedestrians should keep walking; SUMO-GUI shown if started)")
    try:
        for step in range(steps):
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = float(snapshot.timestamp.elapsed_seconds)
            frame = int(snapshot.frame)


            # ====== hazard 行人事件触发（稳定版） ======
            if agent_vehicle is not None and hazard_walkers:
                active_count = sum(1 for rec in hazard_walkers if rec.get("active", False) and _actor_is_alive(rec.get("actor", None)))
                if step >= next_hazard_step and active_count < 1:
                    inactive = [
                        rec for rec in hazard_walkers
                        if (not rec.get("active", False)) and _actor_is_alive(rec.get("actor", None))
                    ]
                    if inactive:
                        chosen = random.choice(inactive)
                        _trigger_hazard_crossing(world, agent_vehicle, chosen)

                    next_hazard_step = step + random.randint(
                        int(HAZARD_TRIGGER_MIN_SEC / DT),
                        int(HAZARD_TRIGGER_MAX_SEC / DT)
                    )

                _update_hazard_walkers(hazard_walkers, DT)

            tls_all = None

            # 全局日志
            if (step % LOG_GLOBAL_EVERY_N_STEPS) == 0:
                for v in vehicles:
                    tf = v.get_transform()
                    loc = tf.location
                    rot = tf.rotation
                    spd = _vec3d_norm(v.get_velocity())
                    w_veh.writerow([step, frame, sim_time, v.id, loc.x, loc.y, loc.z, rot.yaw, rot.pitch, rot.roll, spd])

                #旧行人日志，删去
                '''for p in walkers: 
                    loc = p.get_location()
                    spd = _vec3d_norm(p.get_velocity())
                    w_ped.writerow([step, frame, sim_time, p.id, loc.x, loc.y, loc.z, spd, "normal"])'''
                
                #新行人日志
                # normal walkers
                for p in walkers:
                    loc = p.get_location()
                    spd = _vec3d_norm(p.get_velocity())
                    w_ped.writerow([step, frame, sim_time, p.id, loc.x, loc.y, loc.z, spd, "normal"])

                # hazard walkers（加入失效保护）
                for rec in hazard_walkers:
                    p = rec.get("actor", None)
                    if not _actor_is_alive(p):
                        continue
                    loc = _safe_actor_location(p)
                    if loc is None:
                        continue
                    spd = _safe_actor_speed(p)
                    w_ped.writerow([step, frame, sim_time, p.id, loc.x, loc.y, loc.z, spd, rec.get("status", "hazard")])

                tls_all = world.get_actors().filter("traffic.traffic_light*")
                for tl in tls_all:
                    w_tl.writerow([step, frame, sim_time, tl.id, _traffic_light_state_str(tl.get_state())])

            # agent transition
            cur = None
            if agent_vehicle is not None:
                tf = agent_vehicle.get_transform()
                loc = tf.location
                rot = tf.rotation
                speed = _vec3d_norm(agent_vehicle.get_velocity())
                ctrl = agent_vehicle.get_control()

                # 非阻塞传感器读取
                rgb_img = dep_img = sem_img = None
                if SAVE_RGB:
                    rgb_img = _drain_queue_latest(agent_qmap["rgb"])
                if SAVE_DEPTH:
                    dep_img = _drain_queue_latest(agent_qmap["depth"])
                if SAVE_SEMANTIC:
                    sem_img = _drain_queue_latest(agent_qmap["semantic"])

                # 仅按频率保存图像（用于验证，不用于训练）
                save_this_step = bool(SAVE_IMAGES and (step % max(1, int(SAVE_IMAGE_EVERY_N_STEPS)) == 0))
                o_rgb = o_dep = o_sem = ""
                if save_this_step:
                    if SAVE_RGB:
                        o_rgb = _save_image_sample(rgb_img, "rgb", None)
                    if SAVE_DEPTH:
                        o_dep = _save_image_sample(dep_img, "depth", carla.ColorConverter.LogarithmicDepth)
                    if SAVE_SEMANTIC:
                        o_sem = _save_image_sample(sem_img, "semantic", carla.ColorConverter.CityScapesPalette)

                # collision
                col_flag = 0
                col_other_id = -1
                col_other_type = ""
                if collision_state is not None and int(collision_state.get("last_frame", -1)) == frame:
                    col_flag = 1
                    col_other_id = int(collision_state.get("last_other_id", -1))
                    col_other_type = str(collision_state.get("last_other_type", ""))

                # context features
                pedestrians_all = _get_live_pedestrians(walkers, hazard_walkers)
                min_ped_id, min_ped_dist, min_ped_speed = _nearest_entity(loc, pedestrians_all)
                min_veh_id, min_veh_dist, min_veh_speed = _nearest_entity(loc, vehicles, exclude_id=agent_vehicle.id)
                front_veh_id, front_veh_dist = _nearest_front_vehicle(agent_vehicle, vehicles)
                num_ped_r10 = _count_within_radius(loc, pedestrians_all, radius=10.0)
                num_veh_r15 = _count_within_radius(loc, vehicles, radius=15.0, exclude_id=agent_vehicle.id)

                # traffic light info
                at_tl_flag = 0
                at_tl_id = -1
                at_tl_state = "None"
                try:
                    if agent_vehicle.is_at_traffic_light():
                        at_tl_flag = 1
                        tl = agent_vehicle.get_traffic_light()
                        if tl is not None:
                            at_tl_id = int(tl.id)
                            at_tl_state = _traffic_light_state_str(tl.get_state())
                except Exception:
                    pass

                if tls_all is None:
                    tls_all = world.get_actors().filter("traffic.traffic_light*")
                nearest_tl_id, nearest_tl_state, nearest_tl_dist = _nearest_traffic_light(loc, tls_all)

                try:
                    speed_limit = float(agent_vehicle.get_speed_limit())
                except Exception:
                    speed_limit = -1.0

                done = 1 if (col_flag == 1 or step == steps - 1) else 0
                reward = _compute_reward(speed, ctrl, col_flag, min_ped_dist)

                cur = {
                    "step": step, "frame": frame, "timestamp": sim_time, "vehicle_id": agent_vehicle.id,
                    "o_rgb": o_rgb, "o_depth": o_dep, "o_sem": o_sem,
                    "x": loc.x, "y": loc.y, "z": loc.z,
                    "yaw": rot.yaw, "pitch": rot.pitch, "roll": rot.roll,
                    "speed": speed,
                    "min_ped_id": min_ped_id, "min_ped_dist": min_ped_dist, "min_ped_speed": min_ped_speed,
                    "min_veh_id": min_veh_id, "min_veh_dist": min_veh_dist, "min_veh_speed": min_veh_speed,
                    "front_veh_id": front_veh_id, "front_veh_dist": front_veh_dist,
                    "num_ped_r10": num_ped_r10, "num_veh_r15": num_veh_r15,
                    "nearest_tl_id": nearest_tl_id, "nearest_tl_state": nearest_tl_state, "nearest_tl_dist": nearest_tl_dist,
                    "at_tl_flag": at_tl_flag, "at_tl_id": at_tl_id, "at_tl_state": at_tl_state,
                    "speed_limit": speed_limit,
                    "a_throttle": float(ctrl.throttle),
                    "a_brake": float(ctrl.brake),
                    "a_steer": float(ctrl.steer),
                    "collision_flag": col_flag,
                    "collision_other_id": col_other_id,
                    "collision_other_type": col_other_type,
                    "reward": float(reward),
                    "done": int(done),
                }

                if prev is not None:
                    w_trans.writerow([
                        prev["step"], prev["frame"], prev["timestamp"], prev["vehicle_id"],
                        prev["o_rgb"], prev["o_depth"], prev["o_sem"],
                        prev["x"], prev["y"], prev["z"], prev["yaw"], prev["pitch"], prev["roll"], prev["speed"],
                        prev["min_ped_id"], prev["min_ped_dist"], prev["min_ped_speed"],
                        prev["min_veh_id"], prev["min_veh_dist"], prev["min_veh_speed"],
                        prev["front_veh_id"], prev["front_veh_dist"],
                        prev["num_ped_r10"], prev["num_veh_r15"],
                        prev["nearest_tl_id"], prev["nearest_tl_state"], prev["nearest_tl_dist"],
                        prev["at_tl_flag"], prev["at_tl_id"], prev["at_tl_state"],
                        prev["speed_limit"],
                        prev["a_throttle"], prev["a_brake"], prev["a_steer"],
                        cur["collision_flag"], cur["collision_other_id"], cur["collision_other_type"],
                        cur["reward"], cur["done"],
                        cur["o_rgb"], cur["o_depth"], cur["o_sem"],
                        cur["x"], cur["y"], cur["z"], cur["yaw"], cur["pitch"], cur["roll"], cur["speed"],
                        cur["min_ped_id"], cur["min_ped_dist"], cur["min_ped_speed"],
                        cur["min_veh_id"], cur["min_veh_dist"], cur["min_veh_speed"],
                        cur["front_veh_id"], cur["front_veh_dist"],
                        cur["num_ped_r10"], cur["num_veh_r15"],
                        cur["nearest_tl_id"], cur["nearest_tl_state"], cur["nearest_tl_dist"],
                        cur["at_tl_flag"], cur["at_tl_id"], cur["at_tl_state"],
                        cur["speed_limit"],
                    ])

                prev = cur

            # 行人卡住处理：多次确认后才重规划
            if walkers and (step % check_every == 0):
                for w, ctrl in zip(walkers, controllers):
                    try:
                        cur_loc = w.get_location()
                        prev_loc = last_pos.get(w.id, cur_loc)
                        dist = math.sqrt((cur_loc.x - prev_loc.x) ** 2 + (cur_loc.y - prev_loc.y) ** 2)
                        last_pos[w.id] = cur_loc

                        if dist < STUCK_DIST_EPS:
                            stuck_counts[w.id] = stuck_counts.get(w.id, 0) + 1
                        else:
                            stuck_counts[w.id] = 0

                        if stuck_counts[w.id] >= STUCK_REPLAN_AFTER_N_CHECKS:
                            wloc = cur_loc
                            dest = None
                            for _ in range(10):
                                cand = world.get_random_location_from_navigation()
                                if cand is None:
                                    continue
                                if (cand.x - wloc.x)**2 + (cand.y - wloc.y)**2 > (15.0**2):
                                    dest = cand
                                    break
                            if dest is None:
                                dest = world.get_random_location_from_navigation()
                            if dest is not None:
                                ctrl.go_to_location(dest)
                                ctrl.set_max_speed(random.uniform(1.2, 2.2))
                            stuck_counts[w.id] = 0
                    except Exception:
                        continue

            if (step % flush_every) == 0:
                try:
                    f_trans.flush()
                    f_veh.flush()
                    f_ped.flush()
                    f_tl.flush()
                except Exception:
                    pass

            # 如果你想更快生成数据，可把下面这行注释掉（同步模式下 world.tick 已经控制仿真推进）
            time.sleep(DT)

    except KeyboardInterrupt:
        print("[DEMO] Interrupted by user.")

    finally:
        try:
            f_trans.close()
            f_veh.close()
            f_ped.close()
            f_tl.close()
        except Exception:
            pass

        try:
            for s in agent_sensors.values():
                try:
                    s.stop()
                except Exception:
                    pass
                try:
                    s.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            for ctrl in controllers:
                try:
                    ctrl.stop()
                except Exception:
                    pass
                try:
                    ctrl.destroy()
                except Exception:
                    pass

            for w in walkers:
                try:
                    w.destroy()
                except Exception:
                    pass

            # 清理 hazard walkers
            for rec in hazard_walkers:
                try:
                    actor = rec.get("actor", None)
                    if actor is not None:
                        actor.destroy()
                except Exception:
                    pass

            for v in vehicles:
                try:
                    v.destroy()
                except Exception:
                    pass

            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        except Exception:
            pass

        if sumo_proc is not None:
            try:
                time.sleep(1.0)
                if sumo_proc.poll() is None:
                    sumo_proc.terminate()
            except Exception:
                pass

        print("[DEMO] Done. Cleaned up.")
        print(f"[DATA] Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()