# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO  # 导入LeggedRobotCfg和LeggedRobotCfgPPO类

class H1MimicCfg(LeggedRobotCfg):  # 定义H1MimicCfg类，继承自LeggedRobotCfg
    class env(LeggedRobotCfg.env):  # 定义env子类，继承自LeggedRobotCfg.env
        num_envs = 6144  # 环境数量

        n_demo_steps = 2  # 演示步数
        n_demo = 9 + 3 + 3 + 3 + 6 * 3  # 演示数量，观察高度
        interval_demo_steps = 0.1  # 演示步数间隔

        n_scan = 0  # 扫描数量（注释掉的值为132）
        n_priv = 3  # 私有状态数量
        n_priv_latent = 4 + 1 + 19 * 2  # 私有潜在状态数量
        n_proprio = 3 + 2 + 2 + 19 * 3 + 2  # 本体感受数量（one hot编码）
        history_len = 10  # 历史长度

        prop_hist_len = 4  # 本体感受历史长度
        n_feature = prop_hist_len * n_proprio  # 特征数量

        num_observations = n_feature + n_proprio + n_demo + n_scan + history_len * n_proprio + n_priv_latent + n_priv  # 观测数量

        episode_length_s = 50  # 回合长度（秒）

        num_policy_actions = 19  # 策略动作数量

    class motion:  # 定义motion子类
        motion_curriculum = True  # 是否使用运动课程
        motion_type = "yaml"  # 运动类型
        motion_name = "motions_autogen_all_no_run_jump.yaml"  # 运动名称

        global_keybody = False  # 是否使用全局关键身体
        global_keybody_reset_time = 2  # 全局关键身体重置时间

        num_envs_as_motions = False  # 是否将环境数量作为运动数量

        no_keybody = False  # 是否没有关键身体
        regen_pkl = False  # 是否重新生成pkl文件

        step_inplace_prob = 0.05  # 原地步行概率
        resample_step_inplace_interval_s = 10  # 重新采样原地步行间隔（秒）

    class terrain(LeggedRobotCfg.terrain):  # 定义terrain子类，继承自LeggedRobotCfg.terrain
        horizontal_scale = 0.1  # 水平尺度（米），影响计算时间

        height = [0., 0.04]  # 地形高度范围

    class init_state(LeggedRobotCfg.init_state):  # 定义init_state子类，继承自LeggedRobotCfg.init_state
        pos = [0.0, 0.0, 1.1]  # 初始位置（x, y, z）（米）
        default_joint_angles = {  # 默认关节角度（弧度），当动作为0.0时的目标角度
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.4,
            'left_knee_joint': 0.8,
            'left_ankle_joint': -0.4,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.4,
            'right_knee_joint': 0.8,
            'right_ankle_joint': -0.4,
            'torso_joint': 0.,
            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_joint': 0.,
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_joint': 0.,
        }

    class control(LeggedRobotCfg.control):  # 定义control子类，继承自LeggedRobotCfg.control
        control_type = 'P'  # 控制类型
        stiffness = {  # 刚度参数（N*m/弧度）
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 200,
            'ankle': 40,
            'torso': 300,
            'shoulder': 40,
            "elbow": 40,
        }
        damping = {  # 阻尼参数（N*m*s/弧度）
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 10,
            'knee': 10,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            "elbow": 2,
        }
        action_scale = 0.25  # 动作尺度
        decimation = 4  # 减速比

    class normalization(LeggedRobotCfg.normalization):  # 定义normalization子类，继承自LeggedRobotCfg.normalization
        clip_actions = 10  # 动作裁剪

    class asset(LeggedRobotCfg.asset):  # 定义asset子类，继承自LeggedRobotCfg.asset
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'  # 机器人URDF文件路径
        torso_name = "torso_link"  # 躯干名称
        foot_name = "ankle"  # 脚部名称
        penalize_contacts_on = ["shoulder", "elbow", "hip"]  # 惩罚接触的部位
        terminate_after_contacts_on = ["torso_link"]  # 终止接触的部位
        self_collisions = 0  # 自碰撞（1表示禁用，0表示启用）

    class rewards(LeggedRobotCfg.rewards):  # 定义rewards子类，继承自LeggedRobotCfg.rewards
        class scales:  # 定义scales子类
            alive = 1  # 存活奖励
            tracking_lin_vel = 6  # 线速度跟踪奖励
            tracking_demo_yaw = 1  # 演示偏航跟踪奖励
            tracking_demo_roll_pitch = 1  # 演示滚转俯仰跟踪奖励
            orientation = -2  # 方向奖励
            tracking_demo_dof_pos = 3  # 演示关节位置跟踪奖励
            tracking_demo_key_body = 2  # 演示关键身体跟踪奖励
            lin_vel_z = -1.0  # 线速度z方向奖励
            ang_vel_xy = -0.4  # 角速度xy方向奖励
            dof_acc = -3e-7  # 关节加速度奖励
            collision = -10.  # 碰撞奖励
            action_rate = -0.1  # 动作速率奖励
            energy = -1e-3  # 能量奖励
            dof_error = -0.1  # 关节误差奖励
            feet_stumble = -2  # 脚部绊倒奖励
            feet_drag = -0.1  # 脚部拖拽奖励
            dof_pos_limits = -10.0  # 关节位置限制奖励
            feet_air_time = 10  # 脚部空中时间奖励
            feet_height = 2  # 脚部高度奖励
            feet_force = -3e-3  # 脚部力奖励

        only_positive_rewards = False  # 是否只使用正奖励
        clip_rewards = True  # 是否裁剪奖励
        soft_dof_pos_limit = 0.95  # 关节位置软限制
        base_height_target = 0.25  # 基础高度目标

    class domain_rand(LeggedRobotCfg.domain_rand):  # 定义domain_rand子类，继承自LeggedRobotCfg.domain_rand
        randomize_gravity = True  # 是否随机化重力
        gravity_rand_interval_s = 10  # 重力随机化间隔（秒）
        gravity_range = [-0.1, 0.1]  # 重力范围

    class noise:  # 定义noise子类
        add_noise = True  # 是否添加噪声
        noise_scale = 0.5  # 噪声尺度
        class noise_scales:  # 定义noise_scales子类
            dof_pos = 0.01  # 关节位置噪声尺度
            dof_vel = 0.15  # 关节速度噪声尺度
            ang_vel = 0.3  # 角速度噪声尺度
            imu = 0.2  # IMU噪声尺度

class H1MimicCfgPPO(LeggedRobotCfgPPO):  # 定义H1MimicCfgPPO类，继承自LeggedRobotCfgPPO
    class runner(LeggedRobotCfgPPO.runner):  # 定义runner子类，继承自LeggedRobotCfgPPO.runner
        runner_class_name = "OnPolicyRunnerMimic"  # 运行器类名
        policy_class_name = 'ActorCriticMimic'  # 策略类名
        algorithm_class_name = 'PPOMimic'  # 算法类名

    class policy(LeggedRobotCfgPPO.policy):  # 定义policy子类，继承自LeggedRobotCfgPPO.policy
        continue_from_last_std = False  # 是否从上一个标准继续
        text_feat_input_dim = H1MimicCfg.env.n_feature  # 文本特征输入维度
        text_feat_output_dim = 16  # 文本特征输出维度
        feat_hist_len = H1MimicCfg.env.prop_hist_len  # 特征历史长度

    class algorithm(LeggedRobotCfgPPO.algorithm):  # 定义algorithm子类，继承自LeggedRobotCfgPPO.algorithm
        entropy_coef = 0.005  # 熵系数

    class estimator:  # 定义estimator子类
        train_with_estimated_states = False  # 是否使用估计状态进行训练
        learning_rate = 1.e-4  # 学习率
        hidden_dims = [128, 64]  # 隐藏层维度
        priv_states_dim = H1MimicCfg.env.n_priv  # 私有状态维度
        priv_start = H1MimicCfg.env.n_feature + H1MimicCfg.env.n_proprio + H1MimicCfg.env.n_demo + H1MimicCfg.env.n_scan  # 私有状态起始位置
        prop_start = H1MimicCfg.env.n_feature  # 本体感受起始位置
        prop_dim = H1MimicCfg.env.n_proprio  # 本体感受维度

class H1MimicDistillCfgPPO(H1MimicCfgPPO):  # 定义H1MimicDistillCfgPPO类，继承自H1MimicCfgPPO
    class distill:  # 定义distill子类
        num_demo = 3  # 演示次数
        num_steps_per_env = 24  # 每个环境的步数
        num_pretrain_iter = 0  # 预训练迭代次数
        activation = "elu"  # 激活函数
        learning_rate = 1.e-4  # 学习率
        student_actor_hidden_dims = [1024, 1024, 512]  # 学生actor的隐藏层维度
        num_mini_batches = 4  # 小批量数量