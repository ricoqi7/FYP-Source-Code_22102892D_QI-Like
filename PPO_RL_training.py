import os
import time
import math
import subprocess
import numpy as np
import carla
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ===================================================
# 1. 全局配置 (同步自 carla_demo_v3_1.py)
# ===================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
SUMO_PORT = 8813
DT = 0.05  # 仿真步长
NUM_VEHICLES = 20
NUM_PEDESTRIANS = 30

# 你提供的最优权重结果
W1_SAFETY = 0.567
W2_EFFICIENCY = 0.275
W3_COMFORT = 0.158

# ===================================================
# 2. 自定义 CARLA-SUMO 环境
# ===================================================
class CarlaSumoPpoEnv(gym.Env):
    def __init__(self):
        super(CarlaSumoPpoEnv, self).__init__()
        
        # 动作空间: [加减速(-1到1), 转向(-1到1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观测空间: 12维低维特征 (速度, 航向角速度, 最近行人距离, 前车距离, 侧方距离等)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # 连接 CARLA
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        self.sumo_proc = None
        self.ego_vehicle = None
        self.collision_sensor = None
        self.collision_flag = False
        self.actors = []
        
        # 启动 SUMO 桥接
        self._start_sumo_bridge()

    def _start_sumo_bridge(self):
        """调用你 demo 中的启动逻辑"""
        print("[ENV] 正在激活 SUMO-Bridge...")
        sumo_script = os.path.join(os.environ.get('CARLA_ROOT', ''), "Co-Simulation/Sumo/run_synchronization.py")
        if os.path.exists(sumo_script):
            self.sumo_proc = subprocess.Popen([
                "python", sumo_script, 
                "Town10HD.sumocfg", 
                "--sumo-port", str(SUMO_PORT)
            ])
            time.sleep(2.0)

    def _get_obs(self):
        """
        特征提取逻辑 (对应你 report 中的第一人称观察)
        """
        v = self.ego_vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        ang_v = self.ego_vehicle.get_angular_velocity().z
        
        # 简化版最近物体检测 (可以复用 v3_1 中的 _nearest_entity 函数)
        # 这里预留 12 维特征
        obs = np.array([
            kmh / 30.0,      # 归一化速度
            ang_v,           # 角速度
            1.0, 1.0, 1.0,   # 这里的占位符应替换为传感器探测的距离
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        
        # 重新生成 Ego
        bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn_point = self.map.get_spawn_points()[0]
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actors.append(self.ego_vehicle)
        
        # 碰撞传感器
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actors.append(self.collision_sensor)
        
        self.collision_flag = False
        self.world.tick()
        return self._get_obs(), {}

    def _on_collision(self, event):
        self.collision_flag = True

    def step(self, action):
        # 执行动作
        control = carla.VehicleControl()
        control.throttle = float(np.clip(action[0], 0.0, 1.0))
        control.brake = float(np.clip(-action[0], 0.0, 1.0))
        control.steer = float(action[1])
        self.ego_vehicle.apply_control(control)
        
        self.world.tick()
        
        obs = self._get_obs()
        
        # 1. 安全得分 (基于 w1)
        r_safety = -100.0 if self.collision_flag else 1.0
        
        # 2. 效率得分 (基于 w2, 目标 20km/h)
        speed_kmh = obs[0] * 30.0
        r_efficiency = 1.0 - abs(speed_kmh - 20.0) / 20.0
        
        # 3. 舒适得分 (基于 w3)
        r_comfort = -abs(action[1]) # 惩罚大幅度转向
        
        # 最终奖励计算
        reward = (W1_SAFETY * r_safety) + (W2_EFFICIENCY * r_efficiency) + (W3_COMFORT * r_comfort)
        
        done = self.collision_flag
        truncated = False
        
        return obs, reward, done, truncated, {}

    def _cleanup(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors = []

    def close(self):
        self._cleanup()
        if self.sumo_proc:
            self.sumo_proc.terminate()

# ===================================================
# 3. 训练启动脚本
# ===================================================
def run_training():
    env = CarlaSumoPpoEnv()
    
    # 定义 100 个迭代。
    # 在 PPO 中，每个迭代由 n_steps 决定。
    # 如果 n_steps=2048，那么 100 次迭代总步数就是 204,800。
    iters = 100
    steps_per_iter = 2048
    total_timesteps = iters * steps_per_iter

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=steps_per_iter, 
        batch_size=64,
        device="cuda" # 如果有显卡建议开启
    )

    # 保存中间模型的回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./logs/',
        name_prefix='ppo_carla_model'
    )

    print(f"--- 启动 PPO 训练 (总计 {iters} 个迭代) ---")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    model.save("ppo_carla_final_policy")
    env.close()

if __name__ == "__main__":
    run_training()