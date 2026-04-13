# FYP-Source-Code_22102892D_QI-Like

## 🛠️ Code Structure & Responsibilities

1. **Environment Setup (`carla_demo_v3_1.py`)**
   - Builds the base simulation world.
   - Manages SUMO co-simulation, 20 background vehicles, and 30 pedestrians with chaotic logic.

2. **Data Collection (`carla_manual_gamepad_record.py`)**
   - Collects expert driving data in chaotic environments via gamepad.
   - Generates offline datasets for reward weight optimization.

3. **Weight Optimization (`optimize_reward_weights.py`)**
   - Performs meta-optimization to determine optimal reward weights:
     - Safety ($w_1$): 0.567
     - Efficiency ($w_2$): 0.275
     - Comfort ($w_3$): 0.158

4. **RL Training (`ppo_complete_train.py`)**
   - Executes PPO algorithm for 100 iterations of online fine-tuning.
   - Trains the driving policy using the optimized weights within the `v3_1` environment.

5. **Policy Testing (`carla_policy_test.py`)**
   - Performs inference tests to demonstrate AI performance in edge cases (e.g., hazard pedestrian crossings).

## 🛠️ 代码结构与职责

1. **环境准备 (`carla_demo_v3_1.py`)**
   - 负责构建基础仿真世界。
   - 实现 SUMO 协同仿真、20 辆背景车及 30 名行人的复杂逻辑。

2. **数据采集 (`carla_manual_gamepad_record.py`)**
   - 负责通过手柄采集专家在混乱环境下的操作数据。
   - 生成用于权重优化的离线数据集。

3. **权重优化 (`optimize_reward_weights.py`)**
   - 负责预训练过程，确定最优奖励权重：
     - Safety ($w_1$): 0.567
     - Efficiency ($w_2$): 0.275
     - Comfort ($w_3$): 0.158

4. **RL 训练 (`ppo_complete_train.py`)**
   - 负责执行 PPO 算法进行 100 次迭代微调。
   - 在 `v3_1` 环境基础上，利用上述权重训练自动驾驶 Policy。

5. **模型测试 (`carla_policy_test.py`)**
   - 负责推理测试，展示 AI 策略在处理异常行人横穿等场景下的表现。
