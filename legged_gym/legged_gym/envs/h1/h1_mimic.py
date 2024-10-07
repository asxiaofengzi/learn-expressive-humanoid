from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class H1Mimic(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):  # 定义初始化方法，接受配置、仿真参数、物理引擎、仿真设备和是否无头模式
        self.cfg = cfg  # 保存配置
        self.sim_params = sim_params  # 保存仿真参数
        self.height_samples = None  # 初始化高度样本为None
        self.debug_viz = True  # 启用调试可视化
        self.init_done = False  # 初始化完成标志为False
        self._parse_cfg(self.cfg)  # 解析配置

        # 预初始化运动加载
        self.sim_device = sim_device  # 保存仿真设备
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)  # 解析仿真设备字符串
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:  # 如果设备类型是CUDA且使用GPU管道
            self.device = self.sim_device  # 使用CUDA设备
        else:  # 否则
            self.device = 'cpu'  # 使用CPU设备

        self.init_motions(cfg)  # 初始化运动
        if cfg.motion.num_envs_as_motions:  # 如果配置中指定将环境数量作为运动数量
            self.cfg.env.num_envs = self._motion_lib.num_motions()  # 设置环境数量为运动数量

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)  # 调用BaseTask的初始化方法

        if not self.headless:  # 如果不是无头模式
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)  # 设置摄像机位置和视角
        self._init_buffers()  # 初始化缓冲区
        self._prepare_reward_function()  # 准备奖励函数
        self.init_done = True  # 设置初始化完成标志为True
        self.global_counter = 0  # 初始化全局计数器为0
        self.total_env_steps_counter = 0  # 初始化总环境步数计数器为0

        self.init_motion_buffers(cfg)  # 初始化运动缓冲区
        # self.rand_vx_cmd = 4*torch.rand((self.num_envs, ), device=self.device) - 2  # 随机生成速度命令（注释掉）

        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)  # 重置索引
        self.post_physics_step()  # 执行物理步后的操作

    def _get_noise_scale_vec(self, cfg):  # 定义_get_noise_scale_vec方法，接受配置参数
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)  # 初始化噪声尺度向量，大小为1行n_proprio列，设备为self.device
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel  # 设置角速度噪声尺度
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu  # 设置IMU噪声尺度
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos  # 设置关节位置噪声尺度
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel  # 设置关节速度噪声尺度
        return noise_scale_vec  # 返回噪声尺度向量

    def init_motions(self, cfg):  # 定义init_motions方法，接受配置参数
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  # 初始化关键身体ID
        # 关键身体ID对应的身体部位注释
        # ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
        # 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
        # 'torso_link', 
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_keypoint_link', 
        # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_keypoint_link']
        self._key_body_ids_sim = torch.tensor([1, 4, 5,  # 左髋关节偏航、膝盖、脚踝
                                               6, 9, 10,  # 右髋关节偏航、膝盖、脚踝
                                               12, 15, 16,  # 左肩关节俯仰、肘部、手
                                               17, 20, 21], device=self.device)  # 右肩关节俯仰、肘部、手
        self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # 关键身体ID子集，不包括膝盖和脚踝

        self._num_key_bodies = len(self._key_body_ids_sim_subset)  # 关键身体数量
        self._dof_body_ids = [1, 2, 3,  # 髋关节、膝盖、脚踝
                              4, 5, 6,  # 髋关节、膝盖、脚踝
                              7,  # 躯干
                              8, 9, 10,  # 肩关节、肘部、手
                              11, 12, 13]  # 13
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10,  # 关节偏移
                             11, 
                             14, 15, 16, 19, 20, 21]  # 14
        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4, device=self.device, dtype=torch.bool)  # 初始化有效关节ID
        self._valid_dof_body_ids[-1] = 0  # 设置最后一个关节ID无效
        self._valid_dof_body_ids[-6] = 0  # 设置倒数第六个关节ID无效
        self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=self.device, dtype=torch.long)  # 仿真关节索引
        self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=self.device, dtype=torch.long)  # 运动关节索引

        # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # 不包括膝盖和脚踝
        self._dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # 不包括膝盖和脚踝
        self._n_demo_dof = len(self._dof_ids_subset)  # 演示关节数量

        # 关节名称注释
        #['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
        #'left_knee_joint', 'left_ankle_joint', 
        #'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 
        #'right_knee_joint', 'right_ankle_joint', 
        #'torso_joint', 
        #'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        # self.dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device, dtype=torch.long)
        # motion_name = "17_04_stealth"
        if cfg.motion.motion_type == "single":  # 如果运动类型是单一
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")  # 设置运动文件路径
        else:  # 否则
            assert cfg.motion.motion_type == "yaml"  # 确保运动类型是yaml
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")  # 设置运动文件路径

        self._load_motion(motion_file, cfg.motion.no_keybody)  # 加载运动文件

    def init_motion_buffers(self, cfg):  # 定义init_motion_buffers方法，接受配置参数
        num_motions = self._motion_lib.num_motions()  # 获取运动库中的运动数量
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)  # 初始化运动ID，范围为0到环境数量
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)  # 取运动ID对运动数量的余数，确保运动ID在有效范围内
        if cfg.motion.motion_curriculum:  # 如果使用运动课程
            self._max_motion_difficulty = 9  # 设置最大运动难度为9
            # self._motion_ids = self._motion_lib.sample_motions(self.num_envs, self._max_motion_difficulty)  # 采样运动ID（注释掉）
        else:  # 如果不使用运动课程
            self._max_motion_difficulty = 9  # 设置最大运动难度为9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)  # 采样运动时间
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)  # 获取运动长度
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)  # 获取运动难度
        # self._motion_features = self._motion_lib.get_motion_features(self._motion_ids)  # 获取运动特征（注释掉）

        self._motion_dt = self.dt  # 设置运动时间步长
        self._motion_num_future_steps = self.cfg.env.n_demo_steps  # 设置未来步数为演示步数
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)  # 初始化演示偏移
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)  # 初始化演示观测缓冲区
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]  # 当前演示观测缓冲区
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]  # 下一步演示观测缓冲区
        # self._curr_mimic_obs_buf = torch.zeros_like(self._curr_demo_obs_buf, device=self.device)  # 当前模仿观测缓冲区（注释掉）

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)  # 当前演示根位置
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)  # 当前演示四元数
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)  # 当前演示根速度
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)  # 当前演示关键身体
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 初始化原地标志

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)  # 关节终止阈值
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)  # 关键身体终止阈值
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)  # 偏航终止阈值
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)  # 高度终止阈值

        # self.step_inplace_ids = self.resample_step_inplace_ids()  # 重新采样原地步行ID（注释掉）

    def _load_motion(self, motion_file, no_keybody=False):  # 定义_load_motion方法，接受运动文件路径和是否无关键身体的标志
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used  # 确认关节偏移（注释掉）
        self._motion_lib = MotionLib(motion_file=motion_file,  # 初始化运动库
                                     dof_body_ids=self._dof_body_ids,  # 关节身体ID
                                     dof_offsets=self._dof_offsets,  # 关节偏移
                                     key_body_ids=self._key_body_ids.cpu().numpy(),  # 关键身体ID
                                     device=self.device,  # 设备
                                     no_keybody=no_keybody,  # 是否无关键身体
                                     regen_pkl=self.cfg.motion.regen_pkl)  # 是否重新生成pkl文件
        return  # 返回

    def step(self, actions):  # 定义step方法，接受动作参数
        actions = self.reindex(actions)  # 重新索引动作

        actions.to(self.device)  # 将动作移动到设备
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)  # 更新动作历史缓冲区
        if self.cfg.domain_rand.action_delay:  # 如果使用动作延迟
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:  # 如果全局计数器达到延迟更新步数
                if len(self.cfg.domain_rand.action_curr_step) != 0:  # 如果当前步骤不为空
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)  # 更新延迟
            if self.viewer:  # 如果有查看器
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)  # 更新延迟
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)  # 随机延迟（注释掉）
            indices = -self.delay - 1  # 计算延迟索引
            actions = self.action_history_buf[:, indices.long()]  # 应用延迟

        self.global_counter += 1  # 增加全局计数器
        self.total_env_steps_counter += 1  # 增加总环境步数计数器
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale  # 计算动作裁剪值
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # 裁剪动作并移动到设备
        self.render()  # 渲染

        self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -0.5, 0.5)  # 限制特定动作的范围
        for _ in range(self.cfg.control.decimation):  # 根据减速比循环
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)  # 计算扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))  # 设置关节驱动扭矩
            self.gym.simulate(self.sim)  # 进行仿真
            self.gym.fetch_results(self.sim, True)  # 获取仿真结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态张量
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():  # 打印前三个扭矩（注释掉）
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])

        self.post_physics_step()  # 执行物理步后的操作

        clip_obs = self.cfg.normalization.clip_observations  # 获取观测裁剪值
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)  # 裁剪观测缓冲区
        if self.privileged_obs_buf is not None:  # 如果有特权观测缓冲区
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)  # 裁剪特权观测缓冲区
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:  # 如果使用摄像头且达到更新间隔
            self.extras["depth"] = self.depth_buffer[:, -2]  # 更新深度缓冲区
        else:  # 否则
            self.extras["depth"] = None  # 设置深度为None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras  # 返回观测、奖励、重置缓冲区和额外信息
    
    def resample_motion_times(self, env_ids):  # 定义resample_motion_times方法，接受环境ID
        return self._motion_lib.sample_time(self._motion_ids[env_ids])  # 重新采样指定环境ID的运动时间

    def update_motion_ids(self, env_ids):  # 定义update_motion_ids方法，接受环境ID
        self._motion_times[env_ids] = self.resample_motion_times(env_ids)  # 更新指定环境ID的运动时间
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])  # 更新指定环境ID的运动长度
        self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])  # 更新指定环境ID的运动难度

    def reset_idx(self, env_ids, init=False):  # 定义reset_idx方法，接受环境ID和初始化标志
        if len(env_ids) == 0:  # 如果环境ID为空
            return  # 返回
        # RSI
        if self.cfg.motion.motion_curriculum:  # 如果使用运动课程
            # ep_length = self.episode_length_buf[env_ids] * self.dt
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]  # 计算完成率
            completion_rate_mean = completion_rate.mean()  # 计算平均完成率
            # if completion_rate_mean > 0.8:
            #     self._max_motion_difficulty = min(self._max_motion_difficulty + 1, 9)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            # elif completion_rate_mean < 0.4:
            #     self._max_motion_difficulty = max(self._max_motion_difficulty - 1, 0)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            relax_ids = completion_rate < 0.3  # 低完成率的环境ID
            strict_ids = completion_rate > 0.9  # 高完成率的环境ID
            # self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05  # 减小高完成率环境的关节终止阈值
            self.dof_term_threshold.clamp_(1.5, 3)  # 限制关节终止阈值的范围

            self.height_term_threshold[env_ids[relax_ids]] += 0.01  # 增加低完成率环境的高度终止阈值
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01  # 减小高完成率环境的高度终止阈值
            self.height_term_threshold.clamp_(0.03, 0.1)  # 限制高度终止阈值的范围

            relax_ids = completion_rate < 0.6  # 低完成率的环境ID
            strict_ids = completion_rate > 0.9  # 高完成率的环境ID
            self.keybody_term_threshold[env_ids[relax_ids]] -= 0.05  # 减小低完成率环境的关键身体终止阈值
            self.keybody_term_threshold[env_ids[strict_ids]] += 0.05  # 增加高完成率环境的关键身体终止阈值
            self.keybody_term_threshold.clamp_(0.1, 0.4)  # 限制关键身体终止阈值的范围

            relax_ids = completion_rate < 0.4  # 低完成率的环境ID
            strict_ids = completion_rate > 0.8  # 高完成率的环境ID
            self.yaw_term_threshold[env_ids[relax_ids]] -= 0.05  # 减小低完成率环境的偏航终止阈值
            self.yaw_term_threshold[env_ids[strict_ids]] += 0.05  # 增加高完成率环境的偏航终止阈值
            self.yaw_term_threshold.clamp_(0.1, 0.6)  # 限制偏航终止阈值的范围

        self.update_motion_ids(env_ids)  # 更新运动ID

        motion_ids = self._motion_ids[env_ids]  # 获取指定环境ID的运动ID
        motion_times = self._motion_times[env_ids]  # 获取指定环境ID的运动时间
        root_pos, root_rot, dof_pos_motion, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, motion_times)  # 获取运动状态

        # Intialize dof state from default position and reference position
        dof_pos_motion, dof_vel = self.reindex_dof_pos_vel(dof_pos_motion, dof_vel)  # 重新索引关节位置和速度

        # update curriculum
        if self.cfg.terrain.curriculum:  # 如果使用地形课程
            self._update_terrain_curriculum(env_ids)  # 更新地形课程

        # reset robot states
        self._reset_dofs(env_ids, dof_pos_motion, dof_vel)  # 重置关节状态
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])  # 重置根状态

        if init:  # 如果是初始化
            self.init_root_pos_global = self.root_states[:, :3].clone()  # 初始化全局根位置
            self.init_root_pos_global_demo = root_pos[:].clone()  # 初始化全局根位置演示
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]  # 初始化目标位置
        else:  # 否则
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()  # 更新全局根位置
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()  # 更新全局根位置演示
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]  # 更新目标位置

        self._resample_commands(env_ids)  # 重新采样命令
        self.gym.simulate(self.sim)  # 进行仿真
        self.gym.fetch_results(self.sim, True)  # 获取仿真结果
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态张量

        # reset buffers
        self.last_actions[env_ids] = 0.  # 重置最后的动作
        self.last_dof_vel[env_ids] = 0.  # 重置最后的关节速度
        self.last_torques[env_ids] = 0.  # 重置最后的扭矩
        self.last_root_vel[:] = 0.  # 重置最后的根速度
        self.feet_air_time[env_ids] = 0.  # 重置脚部空中时间
        self.reset_buf[env_ids] = 1  # 重置缓冲区
        self.obs_history_buf[env_ids, :, :] = 0.  # 重置观测历史缓冲区
        self.contact_buf[env_ids, :, :] = 0.  # 重置接触缓冲区
        self.action_history_buf[env_ids, :, :] = 0.  # 重置动作历史缓冲区
        self.cur_goal_idx[env_ids] = 0  # 重置当前目标索引
        self.reach_goal_timer[env_ids] = 0  # 重置到达目标计时器

        # fill extras
        self.extras["episode"] = {}  # 初始化额外信息
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean  # 记录课程完成率
        for key in self.episode_sums.keys():  # 遍历每个奖励项
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # 记录平均奖励
            self.episode_sums[key][env_ids] = 0.  # 重置奖励和
        self.episode_length_buf[env_ids] = 0  # 重置回合长度缓冲区

        self.extras["episode"]["curriculum_motion_difficulty_level"] = self._max_motion_difficulty  # 记录课程运动难度级别
        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()  # 记录关节终止阈值
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()  # 记录关键身体终止阈值
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()  # 记录偏航终止阈值
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()  # 记录高度终止阈值

        # log additional curriculum info
        if self.cfg.terrain.curriculum:  # 如果使用地形课程
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())  # 记录地形级别
        if self.cfg.commands.curriculum:  # 如果使用命令课程
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]  # 记录最大命令x
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:  # 如果需要发送超时信息
            self.extras["time_outs"] = self.time_out_buf  # 记录超时缓冲区
        return  # 返回
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):  # 定义_reset_dofs方法，接受环境ID、关节位置和关节速度
        # dof_pos_default = self.default_dof_pos + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device) * self.default_dof_pos
        self.dof_pos[env_ids] = dof_pos  # 设置指定环境ID的关节位置
        self.dof_vel[env_ids] = dof_vel  # 设置指定环境ID的关节速度

        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.5, (len(env_ids), self.num_dof), device=self.device)
        # self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 将环境ID转换为int32类型
        self.gym.set_dof_state_tensor_indexed(self.sim,  # 设置关节状态张量
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):  # 定义post_physics_step方法
        # self._motion_sync()
        super().post_physics_step()  # 调用父类的post_physics_step方法

        # step motion lib
        self._motion_times += self._motion_dt  # 更新运动时间
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.  # 重置超过运动长度的运动时间
        self.update_demo_obs()  # 更新演示观测
        # self.update_mimic_obs()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:  # 如果有查看器且启用同步和调试可视化
            self.gym.clear_lines(self.viewer)  # 清除查看器中的线条
            self.draw_rigid_bodies_demo()  # 绘制演示刚体
            self.draw_rigid_bodies_actual()  # 绘制实际刚体

        return  # 返回

    def _post_physics_step_callback(self):  # 定义_post_physics_step_callback方法
        super()._post_physics_step_callback()  # 调用父类的_post_physics_step_callback方法
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:  # 如果达到重力随机化间隔
            self._randomize_gravity()  # 随机化重力
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:  # 如果达到重新采样原地步行间隔
            self.resample_step_inplace_ids()  # 重新采样原地步行ID

    def resample_step_inplace_ids(self):  # 定义resample_step_inplace_ids方法
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob  # 重新采样原地步行ID

    def _randomize_gravity(self, external_force=None):  # 定义_randomize_gravity方法，接受外部力参数
        if self.cfg.domain_rand.randomize_gravity and external_force is None:  # 如果启用重力随机化且外部力为空
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range  # 获取重力范围
            external_force = torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False) * (max_gravity - min_gravity) + min_gravity  # 生成随机重力

        sim_params = self.gym.get_sim_params(self.sim)  # 获取仿真参数
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)  # 计算重力
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)  # 归一化重力向量
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])  # 设置仿真参数中的重力
        self.gym.set_sim_params(self.sim, sim_params)  # 应用仿真参数

    def _parse_cfg(self, cfg):  # 定义_parse_cfg方法，接受配置参数
        super()._parse_cfg(cfg)  # 调用父类的_parse_cfg方法
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)  # 计算重力随机化间隔
        self.cfg.motion.resample_step_inplace_interval = np.ceil(self.cfg.motion.resample_step_inplace_interval_s / self.dt)  # 计算重新采样原地步行间隔

    def _update_goals(self):  # 定义_update_goals方法
        # self.target_pos_abs = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        # self.target_pos_rel = self.target_pos_abs - self.root_states[:, :2]
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0  # 计算需要重置目标位置的环境ID
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]  # 重置目标位置
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]  # 更新目标位置
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])  # 计算相对目标位置
        # print(self.target_pos_rel[self.lookat_id])
        r, p, y = euler_from_quaternion(self._curr_demo_quat)  # 从四元数计算欧拉角
        self.target_yaw = y.clone()  # 更新目标偏航角
        # self.desired_vel_scalar = torch.norm(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+2], dim=-1)

    def update_demo_obs(self):  # 定义update_demo_obs方法
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # 计算演示运动时间
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)  # 获取运动状态
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)  # 重新索引关节位置和速度

        self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]  # 更新当前演示根位置
        self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]  # 更新当前演示四元数
        self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]  # 更新当前演示根速度
        self._curr_demo_keybody[:] = local_key_body_pos[:, self._key_body_ids_sim_subset].view(self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]  # 更新当前演示关键身体
        self._in_place_flag = 0 * (torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2)  # 更新原地标志
        # for i in range(13):
        #     feet_pos_global = key_pos[:, i]# - root_pos + self.root_states[:, :3]
        #     pose = gymapi.Transform(gymapi.Vec3(feet_pos_global[self.lookat_id, 0], feet_pos_global[self.lookat_id, 1], feet_pos_global[self.lookat_id, 2]), r=None)
        #     gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos[:, self._dof_ids_subset], dof_vel, key_pos, local_key_body_pos[:, self._key_body_ids_sim_subset, :], self._dof_offsets)  # 构建演示观测
        self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]  # 更新演示观测缓冲区

    def compute_obs_buf(self):  # 定义compute_obs_buf方法
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)  # 获取IMU观测
        return torch.cat((  # 拼接观测缓冲区
            # motion_id_one_hot,
            self.base_ang_vel * self.obs_scales.ang_vel,  # 基础角速度
            imu_obs,  # IMU观测
            torch.sin(self.yaw - self.target_yaw)[:, None],  # 目标偏航角的正弦值
            torch.cos(self.yaw - self.target_yaw)[:, None],  # 目标偏航角的余弦值
            # self.target_pos_rel,
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 关节位置
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),  # 关节速度
            self.reindex(self.action_history_buf[:, -1]),  # 动作历史
            self.reindex_feet(self.contact_filt.float() * 0 - 0.5),  # 脚部接触
        ), dim=-1)

    def compute_obs_demo(self):  # 定义compute_obs_demo方法
        obs_demo = self._next_demo_obs_buf.clone()  # 克隆下一步演示观测缓冲区
        obs_demo[self._in_place_flag, self._n_demo_dof:self._n_demo_dof + 3] = 0  # 更新原地标志的观测
        return obs_demo  # 返回演示观测

    def compute_observations(self):  # 定义compute_observations方法
        # motion_id_one_hot = torch.zeros((self.num_envs, self._motion_lib.num_motions()), device=self.device)
        # motion_id_one_hot[torch.arange(self.num_envs, device=self.device), self._motion_ids] = 1.

        obs_buf = self.compute_obs_buf()  # 计算观测缓冲区

        if self.cfg.noise.add_noise:  # 如果添加噪声
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale  # 添加噪声

        obs_demo = self.compute_obs_demo()  # 计算演示观测

        # obs_demo[:, :] = 0
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset)-1] = 1
        # obs_demo[:, self._n_demo_dof:self._n_demo_dof+8] = 1 #self.rand_vx_cmd
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset):] = torch.tensor([ 0.0049,  0.1554,  0.4300,  
        #                                                                      0.0258,  0.2329,  0.1076,  
        #                                                                      0.3195,  0.2040,  0.0537,  
        #                                                                      0.0061, -0.1553,  0.4300,  
        #                                                                      0.0292, -0.2305,  0.1076,  
        #                                                                      0.3225, -0.1892,  0.0598], device=self.device)
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)  # 获取运动特征
        priv_explicit = torch.cat((0 * self.base_lin_vel * self.obs_scales.lin_vel,  # 拼接显式特权观测
            #    global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1),
        ), dim=-1)
        priv_latent = torch.cat((  # 拼接隐式特权观测
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1,
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:  # 如果测量高度
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)  # 计算高度
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)  # 拼接观测缓冲区
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)  # 拼接观测缓冲区

        self.obs_history_buf = torch.where(  # 更新观测历史缓冲区
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(  # 更新接触缓冲区
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    def _motion_sync(self):  # 定义_motion_sync方法
        num_motions = self._motion_lib.num_motions()  # 获取运动库中的运动数量
        motion_ids = self._motion_ids  # 获取当前的运动ID
        # print(self._motion_times[self.lookat_id])
        # motion_times = self.episode_length_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, self._motion_times)  # 获取运动状态
        
        root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]  # 更新根位置的前两维
        root_vel = torch.zeros_like(root_vel)  # 将根速度置零
        root_ang_vel = torch.zeros_like(root_ang_vel)  # 将根角速度置零
        dof_vel = torch.zeros_like(dof_vel)  # 将关节速度置零

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)  # 获取环境ID

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)  # 重新索引关节位置和速度

        self._set_env_state(env_ids=env_ids,  # 设置环境状态
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 将环境ID转换为int32类型
        self.gym.set_actor_root_state_tensor_indexed(self.sim,  # 设置演员根状态张量
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,  # 设置关节状态张量
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return  # 返回

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):  # 定义_set_env_state方法
        self.root_states[env_ids, 0:3] = root_pos  # 设置根位置
        self.root_states[env_ids, 3:7] = root_rot  # 设置根旋转
        self.root_states[env_ids, 7:10] = root_vel  # 设置根速度
        self.root_states[env_ids, 10:13] = root_ang_vel  # 设置根角速度

        self.dof_pos[env_ids] = dof_pos  # 设置关节位置
        self.dof_vel[env_ids] = dof_vel  # 设置关节速度
        return  # 返回

    def check_termination(self):  # 定义check_termination方法
        """ 检查环境是否需要重置 """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)  # 检查接触力是否超过阈值
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        dof_dev = self._reward_tracking_demo_dof_pos() < 0.1  # 检查关节位置偏差
        self.reset_buf |= dof_dev  # 更新重置缓冲区

        # demo_dofs = self._curr_demo_obs_buf[:, :self.num_dof]
        # ref_deviation = torch.norm(self.dof_pos - demo_dofs, dim=1) >= self.dof_term_threshold
        # self.reset_buf |= ref_deviation
        
        # height_dev = torch.abs(self.root_states[:, 2] - self._curr_demo_root_pos[:, 2]) >= self.height_term_threshold
        # self.reset_buf |= height_dev

        # yaw_dev = self._reward_tracking_demo_yaw() < self.yaw_term_threshold
        # self.reset_buf |= yaw_dev

        # ref_keybody_dev = self._reward_tracking_demo_key_body() < 0.2
        # self.reset_buf |= ref_keybody_dev

        # ref_deviation = (torch.norm(self.dof_pos - demo_dofs, dim=1) >= 1.5) & \
        #                 (self._motion_difficulty < 3)
        # self.reset_buf |= ref_deviation
        
        # ref_keybody_dev = (self._reward_tracking_demo_key_body() < 0.3) & \
        #                   (self._motion_difficulty < 3)
        # self.reset_buf |= ref_keybody_dev

        motion_end = self.episode_length_buf * self.dt >= self._motion_lengths  # 检查运动是否结束
        self.reset_buf |= motion_end  # 更新重置缓冲区

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 检查是否超时
        self.time_out_buf |= motion_end  # 更新超时缓冲区

        self.reset_buf |= self.time_out_buf  # 更新重置缓冲区
        # self.reset_buf |= roll_cutoff
        # self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    ######### demonstrations #########
    # def get_demo_obs(self, ):
    #     demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
    #     # get the motion state at the demo times
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids.repeat(self._motion_num_future_steps), demo_motion_times.flatten())
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    # def get_curr_demo(self):
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids, self._motion_times)
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    
    ######### utils #########

    def reindex_dof_pos_vel(self, dof_pos, dof_vel):  # 定义reindex_dof_pos_vel方法，接受关节位置和关节速度
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)  # 重新索引关节位置
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)  # 重新索引关节速度
        return dof_pos, dof_vel  # 返回重新索引后的关节位置和速度

    def draw_rigid_bodies_demo(self):  # 定义draw_rigid_bodies_demo方法
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))  # 创建绿色的线框球体几何体
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)  # 获取当前演示的关键身体位置
        if self.cfg.motion.global_keybody:  # 如果使用全局关键身体
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)  # 获取当前演示的全局位置
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)  # 获取当前演示的局部位置
        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)  # 将局部位置转换为全局位置
        for i in range(global_body_pos.shape[1]):  # 遍历每个关键身体位置
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)  # 创建变换
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 绘制线框球体

    def draw_rigid_bodies_actual(self):  # 定义draw_rigid_bodies_actual方法
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))  # 创建红色的线框球体几何体
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids_sim, :3].clone()  # 获取刚体位置
        for i in range(rigid_body_pos.shape[1]):  # 遍历每个刚体位置
            pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[self.lookat_id, i, 0], rigid_body_pos[self.lookat_id, i, 1], rigid_body_pos[self.lookat_id, i, 2]), r=None)  # 创建变换
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 绘制线框球体

    def _draw_goals(self):  # 定义_draw_goals方法
        demo_geom = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(1, 0, 0))  # 创建红色的线框球体几何体
        
        pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()  # 获取机器人位置
        # print(self._curr_demo_obs_buf[self.lookat_id, 2*self.num_dof:2*self.num_dof+3])
        # demo_pos = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[self.lookat_id]
        # pose = gymapi.Transform(gymapi.Vec3(demo_pos[0], demo_pos[1], demo_pos[2]), r=None)
        # gymutil.draw_lines(demo_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if not self.cfg.depth.use_camera:  # 如果不使用相机
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))  # 创建箭头的线框球体几何体
            # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            norm = torch.norm(self._curr_demo_root_vel[:, :2], dim=-1, keepdim=True)  # 计算当前演示根速度的范数
            target_vec_norm = self._curr_demo_root_vel[:, :2] / (norm + 1e-5)  # 归一化目标向量
            for i in range(5):  # 绘制箭头
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()  # 计算箭头位置
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)  # 创建变换
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 绘制箭头

    ######### Rewards #########

    def compute_reward(self):  # 定义compute_reward方法
        self.rew_buf[:] = 0.  # 初始化奖励缓冲区
        for i in range(len(self.reward_functions)):  # 遍历每个奖励函数
            name = self.reward_names[i]  # 获取奖励名称
            rew = self.reward_functions[i]() * self.reward_scales[name]  # 计算奖励
            self.rew_buf += rew  # 累加奖励
            self.episode_sums[name] += rew  # 累加回合奖励
        if self.cfg.rewards.only_positive_rewards:  # 如果只使用正奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)  # 将奖励裁剪为非负值
        if self.cfg.rewards.clip_rewards:  # 如果裁剪奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)  # 将奖励裁剪为[-0.5, +∞)

        # add termination reward after clipping
        if "termination" in self.reward_scales:  # 如果有终止奖励
            rew = self._reward_termination() * self.reward_scales["termination"]  # 计算终止奖励
            self.rew_buf += rew  # 累加终止奖励
            self.episode_sums["termination"] += rew  # 累加回合终止奖励

    def _reward_tracking_demo_goal_vel(self):  # 定义_reward_tracking_demo_goal_vel方法
        norm = torch.norm(self._curr_demo_root_vel[:, :3], dim=-1, keepdim=True)  # 计算当前演示根速度的范数
        target_vec_norm = self._curr_demo_root_vel[:, :3] / (norm + 1e-5)  # 归一化目标向量
        cur_vel = self.root_states[:, 7:10]  # 获取当前速度
        norm_squeeze = norm.squeeze(-1)  # 压缩范数
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), norm_squeeze) / (norm_squeeze + 1e-5)  # 计算奖励

        rew_zeros = torch.exp(-4*torch.norm(cur_vel, dim=-1))  # 计算零速度奖励
        small_cmd_ids = (norm<0.1).squeeze(-1)  # 获取小命令ID
        rew[small_cmd_ids] = rew_zeros[small_cmd_ids]  # 更新小命令奖励
        # return torch.exp(-2 * torch.norm(cur_vel - self._curr_demo_root_vel[:, :2], dim=-1))
        return rew.squeeze(-1)  # 返回奖励

    def _reward_tracking_vx(self):  # 定义_reward_tracking_vx方法
        rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)  # 计算x方向速度奖励
        # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
        return rew  # 返回奖励

    def _reward_tracking_ang_vel(self):  # 定义_reward_tracking_ang_vel方法
        rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)  # 计算角速度奖励
        return rew  # 返回奖励

    def _reward_tracking_demo_yaw(self):  # 定义_reward_tracking_demo_yaw方法
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))  # 计算偏航角奖励
        # print("yaw rew", rew, self.target_yaw, self.yaw)
        return rew  # 返回奖励

    def _reward_dof_pos_limits(self):  # 定义_reward_dof_pos_limits方法
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit  # 计算关节位置下限的惩罚
        # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)  # 计算关节位置上限的惩罚
        # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
        return torch.sum(out_of_limits, dim=1)  # 返回总惩罚

    def _reward_tracking_demo_dof_pos(self):  # 定义_reward_tracking_demo_dof_pos方法
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]  # 获取演示关节位置
        dof_pos = self.dof_pos[:, self._dof_ids_subset]  # 获取当前关节位置
        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))  # 计算关节位置奖励
        # print(rew[self.lookat_id].cpu().numpy())
        # print("dof_pos", dof_pos)
        # print("demo_dofs", demo_dofs)
        return rew  # 返回奖励

    # def _reward_tracking_demo_dof_vel(self):
    #     demo_dof_vel = self._curr_demo_obs_buf[:, self.num_dof:self.num_dof*2]
    #     rew = torch.exp(- 0.01 * torch.norm(self.dof_vel - demo_dof_vel, dim=1))
    #     return rew

    def _reward_stand_still(self):  # 定义_reward_stand_still方法
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)  # 计算关节位置误差
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)  # 计算关节速度误差
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error)  # 计算静止奖励
        rew[~self._in_place_flag] = 0  # 更新非原地标志的奖励
        return rew  # 返回奖励

    def _reward_tracking_lin_vel(self):  # 定义_reward_tracking_lin_vel方法
        demo_vel = self._curr_demo_obs_buf[:, self._n_demo_dof:self._n_demo_dof+3]  # 获取演示线速度
        demo_vel[self._in_place_flag] = 0  # 更新原地标志的演示线速度
        rew = torch.exp(- 4 * torch.norm(self.base_lin_vel - demo_vel, dim=1))  # 计算线速度奖励
        return rew  # 返回奖励

    def _reward_tracking_demo_ang_vel(self):  # 定义_reward_tracking_demo_ang_vel方法
        demo_ang_vel = self._curr_demo_obs_buf[:, self._n_demo_dof+3:self._n_demo_dof+6]  # 获取演示角速度
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))  # 计算角速度奖励
        return rew  # 返回奖励

    def _reward_tracking_demo_roll_pitch(self):  # 定义_reward_tracking_demo_roll_pitch方法
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]  # 获取演示横滚和俯仰角
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)  # 获取当前横滚和俯仰角
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))  # 计算横滚和俯仰角奖励
        return rew  # 返回奖励

    def _reward_tracking_demo_height(self):  # 定义_reward_tracking_demo_height方法
        demo_height = self._curr_demo_obs_buf[:, self._n_demo_dof+8]  # 获取演示高度
        cur_height = self.root_states[:, 2]  # 获取当前高度
        rew = torch.exp(- 4 * torch.abs(cur_height - demo_height))  # 计算高度奖励
        return rew  # 返回奖励

    def _reward_tracking_demo_key_body(self):  # 定义_reward_tracking_demo_key_body方法
        # demo_key_body_pos_local = self._curr_demo_obs_buf[:, self.num_dof*2+8:].view(self.num_envs, self._num_key_bodies, 3)[:,self._key_body_ids_sim_subset,:].view(self.num_envs, -1)
        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)  # 获取演示关键身体局部位置
        if self.cfg.motion.global_keybody:  # 如果使用全局关键身体
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)  # 获取当前演示的全局位置
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)  # 获取当前演示的局部位置
        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)  # 将局部位置转换为全局位置
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)  # 获取当前刚体全局位置

        # cur_local_body_pos = global_to_local(self.base_quat, cur_global_body_pos.view(self.num_envs, -1, 3), self.root_states[:, :3]).view(self.num_envs, -1)
        # print(cur_local_body_pos)
        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))  # 计算关键身体奖励
        # print("key body rew", rew[self.lookat_id].cpu().numpy())
        return rew  # 返回奖励

    def _reward_tracking_mul(self):  # 定义_reward_tracking_mul方法
        rew_key_body = self._reward_tracking_demo_key_body()  # 获取关键身体的奖励
        rew_roll_pitch = self._reward_tracking_demo_roll_pitch()  # 获取横滚和俯仰角的奖励
        rew_ang_vel = self._reward_tracking_demo_yaw()  # 获取偏航角的奖励
        # rew_dof_vel = self._reward_tracking_demo_dof_vel()
        rew_dof_pos = self._reward_tracking_demo_dof_pos()  # 获取关节位置的奖励
        # rew_goal_vel = self._reward_tracking_lin_vel()#self._reward_tracking_demo_goal_vel()
        rew = rew_key_body * rew_roll_pitch * rew_ang_vel * rew_dof_pos  # 计算总奖励
        # print(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+3][self.lookat_id], self.base_lin_vel[self.lookat_id])
        return rew  # 返回总奖励

    # def _reward_tracking_demo_vel(self):
    #     demo_vel = self.get_curr_demo()[:, self.num_dof:]

    def _reward_feet_drag(self):  # 定义_reward_feet_drag方法
        # print(contact_bool)
        # contact_forces = self.contact_forces[:, self.feet_indices, 2]
        # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
        # print(self.contact_filt[self.lookat_id])
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)  # 计算脚部速度
        dragging_vel = self.contact_filt * feet_xyz_vel  # 计算拖拽速度
        rew = dragging_vel.sum(dim=-1)  # 计算拖拽奖励
        # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
        return rew  # 返回拖拽奖励

    def _reward_energy(self):  # 定义_reward_energy方法
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)  # 计算能量奖励

    def _reward_feet_air_time(self):  # 定义_reward_feet_air_time方法
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检查脚部接触力
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 过滤接触
        self.last_contacts = contact  # 更新上次接触
        first_contact = (self.feet_air_time > 0.) * contact_filt  # 检查首次接触
        self.feet_air_time += self.dt  # 更新脚部空中时间
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # 计算空中时间奖励
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt  # 重置空中时间
        rew_airTime[self._in_place_flag] = 0  # 更新原地标志的奖励
        return rew_airTime  # 返回空中时间奖励

    def _reward_feet_height(self):  # 定义_reward_feet_height方法
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]  # 获取脚部高度
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)  # 计算脚部高度奖励
        rew[self._in_place_flag] = 0  # 更新原地标志的奖励
        # print("height: ", rew[self.lookat_id])
        return rew  # 返回脚部高度奖励

    def _reward_feet_force(self):  # 定义_reward_feet_force方法
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)  # 计算脚部接触力奖励
        rew[rew < 500] = 0  # 过滤小于500的奖励
        rew[rew > 500] -= 500  # 减去500
        rew[self._in_place_flag] = 0  # 更新原地标志的奖励
        # print(rew[self.lookat_id])
        # print(self.dof_names)
        return rew  # 返回脚部接触力奖励

    def _reward_dof_error(self):  # 定义_reward_dof_error方法
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)  # 计算关节位置误差
        return dof_error  # 返回关节位置误差

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):  # 定义build_demo_observations函数
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)  # 计算局部根角速度
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)  # 计算局部根速度
    # print(local_root_vel[0])

    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    # local_root_vel = quat_rotate(heading_rot, root_vel)
    # print(local_root_vel[0], "\n")

    # root_pos_expand = root_pos.unsqueeze(-2)  # [num_envs, 1, 3]
    # local_key_body_pos = key_body_pos - root_pos_expand
    
    # heading_rot_expand = heading_rot.unsqueeze(-2)
    # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    roll, pitch, yaw = euler_from_quaternion(root_rot)  # 计算欧拉角
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)  # 返回拼接后的观测

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):  # 定义reindex_motion_dof函数
    dof = dof.clone()  # 克隆关节位置或速度
    dof[:, indices_sim] = dof[:, indices_motion]  # 重新索引
    return dof[:, valid_dof_body_ids]  # 返回重新索引后的关节位置或速度

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):  # 定义local_to_global函数
    num_key_bodies = rigid_body_pos.shape[1]  # 获取关键身体数量
    num_envs = rigid_body_pos.shape[0]  # 获取环境数量
    total_bodies = num_key_bodies * num_envs  # 计算总身体数量
    heading_rot_expand = quat.unsqueeze(-2)  # 扩展四元数维度
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))  # 重复四元数
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])  # 展平四元数

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)  # 展平刚体位置
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]  # 计算全局身体位置
    return global_body_pos  # 返回全局身体位置

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):  # 定义global_to_local函数
    num_key_bodies = rigid_body_pos.shape[1]  # 获取关键身体数量
    num_envs = rigid_body_pos.shape[0]  # 获取环境数量
    total_bodies = num_key_bodies * num_envs  # 计算总身体数量
    heading_rot_expand = quat.unsqueeze(-2)  # 扩展四元数维度
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))  # 重复四元数
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])  # 展平四元数

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)  # 计算局部刚体位置
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)  # 计算局部身体位置
    return local_end_pos  # 返回局部身体位置

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):  # 定义global_to_local_xy函数
    cos_yaw = torch.cos(yaw)  # 计算偏航角的余弦值
    sin_yaw = torch.sin(yaw)  # 计算偏航角的正弦值

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)  # 构建旋转矩阵
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))  # 计算局部位置变化
    return local_pos_delta.squeeze(-1)  # 返回局部位置变化