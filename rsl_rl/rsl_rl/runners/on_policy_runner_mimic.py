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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from rsl_rl.algorithms import PPOMimic
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
from rsl_rl.utils.utils import Normalizer

class OnPolicyRunnerMimic:

    def __init__(self,
                 env: VecEnv,  # 环境实例
                 train_cfg,  # 训练配置
                 log_dir=None,  # 日志目录，默认为None
                 init_wandb=True,  # 是否初始化wandb，默认为True
                 device='cpu',  # 设备，默认为'cpu'
                 **kwargs):  # 其他关键字参数

        self.cfg = train_cfg["runner"]  # 获取运行器配置
        self.alg_cfg = train_cfg["algorithm"]  # 获取算法配置
        self.policy_cfg = train_cfg["policy"]  # 获取策略配置
        self.estimator_cfg = train_cfg["estimator"]  # 获取估计器配置
        self.depth_encoder_cfg = train_cfg["depth_encoder"]  # 获取深度编码器配置

        if "distill" in train_cfg:  # 如果训练配置中包含蒸馏配置
            self.if_distill = True  # 设置蒸馏标志为True
            self.distill_cfg = train_cfg["distill"]  # 获取蒸馏配置
        else:
            self.if_distill = False  # 设置蒸馏标志为False
            self.distill_cfg = None  # 设置蒸馏配置为None

        self.device = device  # 设置设备
        self.env = env  # 设置环境实例

        policy_class = eval(self.cfg["policy_class_name"])  # 获取策略类
        actor_critic = policy_class(self.env.cfg.env.n_proprio,  # 创建策略实例
                                    self.env.cfg.env.n_demo,
                                    self.env.cfg.env.n_scan,
                                    self.env.num_obs,
                                    self.env.cfg.env.n_priv_latent,
                                    self.env.cfg.env.n_priv,
                                    self.env.cfg.env.history_len,
                                    self.env.cfg.env.num_policy_actions,
                                    **self.policy_cfg).to(self.device)  # 将策略实例移动到指定设备

        estimator = Estimator(input_dim=env.cfg.env.n_proprio,  # 创建估计器实例
                              output_dim=env.cfg.env.n_priv,
                              hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)  # 将估计器实例移动到指定设备

        self.if_depth = self.depth_encoder_cfg["if_depth"]  # 获取是否使用深度编码器的标志
        if self.if_depth:  # 如果使用深度编码器
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio,  # 创建深度编码器骨干网络
                                                      self.policy_cfg["scan_encoder_dims"][-1],
                                                      self.depth_encoder_cfg["hidden_dims"])
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)  # 创建递归深度编码器并移动到指定设备
            depth_actor = deepcopy(actor_critic.actor)  # 深度演员为策略演员的深度复制
        else:
            depth_encoder = None  # 不使用深度编码器
            depth_actor = None  # 不使用深度演员

        if self.if_distill:  # 如果使用蒸馏
            student_actor = ActorDistill(actor_critic.actor, self.distill_cfg).to(self.device)  # 创建学生演员并移动到指定设备
        else:
            student_actor = None  # 不使用学生演员

        alg_class = eval(self.cfg["algorithm_class_name"])  # 获取算法类
        self.alg: PPOMimic = alg_class(self.env,  # 创建算法实例
                                       actor_critic,
                                       estimator, self.estimator_cfg,
                                       depth_encoder, self.depth_encoder_cfg, depth_actor, student_actor, self.distill_cfg,
                                       device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 获取每个环境的步数
        self.save_interval = self.cfg["save_interval"]  # 获取保存间隔
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]  # 获取DAGGER更新频率

        self.alg.init_storage(  # 初始化算法存储
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.cfg.env.num_policy_actions],
        )

        self.learn = self.learn_RL if not self.if_distill else self.learn_distill  # 根据是否使用蒸馏选择学习方法

        self.log_dir = log_dir  # 设置日志目录
        self.writer = None  # 初始化日志写入器为None
        self.tot_timesteps = 0  # 初始化总时间步数为0
        self.tot_time = 0  # 初始化总时间为0
        self.current_learning_iteration = 0  # 初始化当前学习迭代次数为0
        

    # def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):  # 定义learn_RL方法，接受学习迭代次数和是否随机初始化参数
    #     mean_value_loss = 0.  # 初始化平均值损失
    #     mean_surrogate_loss = 0.  # 初始化平均代理损失
    #     mean_estimator_loss = 0.  # 初始化平均估计器损失
    #     mean_disc_loss = 0.  # 初始化平均判别器损失
    #     mean_disc_acc = 0.  # 初始化平均判别器准确率
    #     mean_hist_latent_loss = 0.  # 初始化平均历史潜在损失
    #     mean_priv_reg_loss = 0.  # 初始化平均隐私正则化损失
    #     priv_reg_coef = 0.  # 初始化隐私正则化系数
    #     entropy_coef = 0.  # 初始化熵系数

    #     # 初始化日志写入器
    #     # if self.log_dir is not None and self.writer is None:
    #     #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    #     if init_at_random_ep_len:  # 如果需要随机初始化回合长度
    #         self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))  # 随机初始化回合长度缓冲区

    #     obs = self.env.get_observations()  # 获取环境的观测值
    #     privileged_obs = self.env.get_privileged_observations()  # 获取环境的特权观测值
    #     critic_obs = privileged_obs if privileged_obs is not None else obs  # 如果有特权观测值，则使用特权观测值，否则使用普通观测值
    #     obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)  # 将观测值移动到指定设备
    #     infos = {}  # 初始化信息字典
    #     infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None  # 如果使用深度信息，则克隆深度缓冲区并移动到指定设备
    #     self.alg.actor_critic.train()  # 将策略网络切换到训练模式（例如，启用dropout）

    #     ep_infos = []  # 初始化回合信息列表
    #     rewbuffer = deque(maxlen=100)  # 初始化奖励缓冲区，最大长度为100
    #     rew_explr_buffer = deque(maxlen=100)  # 初始化探索奖励缓冲区，最大长度为100
    #     rew_entropy_buffer = deque(maxlen=100)  # 初始化熵奖励缓冲区，最大长度为100
    #     lenbuffer = deque(maxlen=100)  # 初始化长度缓冲区，最大长度为100
    #     cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前奖励总和
    #     cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前探索奖励总和
    #     cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前熵奖励总和
    #     cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前回合长度

    #     task_rew_buf = deque(maxlen=100)  # 初始化任务奖励缓冲区，最大长度为100
    #     cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前任务奖励总和

    #     tot_iter = self.current_learning_iteration + num_learning_iterations  # 计算总迭代次数
    #     self.start_learning_iteration = copy(self.current_learning_iteration)  # 复制当前学习迭代次数

    #     for it in range(self.current_learning_iteration, tot_iter):  # 遍历每次迭代
    #         start = time.time()  # 记录开始时间
    #         hist_encoding = it % self.dagger_update_freq == 0  # 判断是否需要进行DAGGER更新
    #         # Rollout
    #         with torch.inference_mode():  # 使用推理模式
    #             for i in range(self.num_steps_per_env):  # 遍历每个环境的步数
    #                 actions = self.alg.act(obs, critic_obs, infos, hist_encoding)  # 获取动作
    #                 obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # 执行动作，获取新的观测值、奖励、完成标志和信息
    #                 critic_obs = privileged_obs if privileged_obs is not None else obs  # 如果有特权观测值，则使用特权观测值，否则使用普通观测值
    #                 obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)  # 将观测值、奖励和完成标志移动到指定设备
    #                 total_rew = self.alg.process_env_step(rewards, dones, infos)  # 处理环境步，获取总奖励

    #                 if self.log_dir is not None:  # 如果日志目录不为空
    #                     # 记录信息
    #                     if 'episode' in infos:  # 如果信息中包含回合信息
    #                         ep_infos.append(infos['episode'])  # 将回合信息添加到列表中
    #                     cur_reward_sum += total_rew  # 更新当前奖励总和
    #                     cur_reward_explr_sum += 0  # 更新当前探索奖励总和
    #                     cur_reward_entropy_sum += 0  # 更新当前熵奖励总和
    #                     cur_episode_length += 1  # 更新当前回合长度

    #                     new_ids = (dones > 0).nonzero(as_tuple=False)  # 获取完成标志为True的索引

    #                     rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前奖励总和添加到奖励缓冲区
    #                     rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前探索奖励总和添加到探索奖励缓冲区
    #                     rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前熵奖励总和添加到熵奖励缓冲区
    #                     lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())  # 将当前回合长度添加到长度缓冲区

    #                     cur_reward_sum[new_ids] = 0  # 重置当前奖励总和
    #                     cur_reward_explr_sum[new_ids] = 0  # 重置当前探索奖励总和
    #                     cur_reward_entropy_sum[new_ids] = 0  # 重置当前熵奖励总和
    #                     cur_episode_length[new_ids] = 0  # 重置当前回合长度
    #                     # AMP
    #                     task_rew_buf.extend(cur_task_rew_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前任务奖励总和添加到任务奖励缓冲区
    #                     cur_task_rew_sum[new_ids] = 0  # 重置当前任务奖励总和
    #         stop = time.time()  # 记录停止时间
    #         collection_time = stop - start  # 计算收集时间

    #         # 学习步骤
    #         start = stop  # 更新开始时间
    #         self.alg.compute_returns(critic_obs)  # 计算回报

    #     mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()  # 更新算法，获取损失和准确率
    #     if hist_encoding:  # 如果需要进行DAGGER更新
    #         print("Updating dagger...")  # 打印更新信息
    #         mean_hist_latent_loss = self.alg.update_dagger()  # 更新DAGGER，获取历史潜在损失

    #     stop = time.time()  # 记录停止时间
    #     learn_time = stop - start  # 计算学习时间
    #     if self.log_dir is not None:  # 如果日志目录不为空
    #         self.log(locals())  # 记录日志
    #     if it < 2500:  # 如果迭代次数小于2500
    #         if it % self.save_interval == 0:  # 如果迭代次数是保存间隔的倍数
    #             self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
    #     elif it < 5000:  # 如果迭代次数小于5000
    #         if it % (2 * self.save_interval) == 0:  # 如果迭代次数是保存间隔的2倍
    #             self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
    #     else:  # 如果迭代次数大于等于5000
    #         if it % (5 * self.save_interval) == 0:  # 如果迭代次数是保存间隔的5倍
    #             self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
    #     ep_infos.clear()  # 清空回合信息列表

    #     # self.current_learning_iteration += num_learning_iterations  # 更新当前学习迭代次数
    #     self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))  # 保存模型

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):  # 定义learn_RL方法，接受学习迭代次数和是否在随机回合长度初始化的参数
        mean_value_loss = 0.  # 初始化平均值损失
        mean_surrogate_loss = 0.  # 初始化平均代理损失
        mean_estimator_loss = 0.  # 初始化平均估计器损失
        mean_disc_loss = 0.  # 初始化平均判别器损失
        mean_disc_acc = 0.  # 初始化平均判别器准确率
        mean_hist_latent_loss = 0.  # 初始化平均历史潜在损失
        mean_priv_reg_loss = 0.  # 初始化平均隐私正则化损失
        priv_reg_coef = 0.  # 初始化隐私正则化系数
        entropy_coef = 0.  # 初始化熵系数
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:  # 如果在随机回合长度初始化
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))  # 随机初始化回合长度缓冲区
        obs = self.env.get_observations()  # 获取环境的观测值
        privileged_obs = self.env.get_privileged_observations()  # 获取环境的特权观测值
        critic_obs = privileged_obs if privileged_obs is not None else obs  # 如果有特权观测值，则使用特权观测值作为评论家观测值，否则使用普通观测值
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)  # 将观测值和评论家观测值移动到指定设备
        infos = {}  # 初始化信息字典
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None  # 如果使用深度信息，则将深度缓冲区克隆并移动到指定设备
        self.alg.actor_critic.train()  # 切换到训练模式（例如，为了使用dropout）

        ep_infos = []  # 初始化回合信息列表
        rewbuffer = deque(maxlen=100)  # 初始化奖励缓冲区，最大长度为100
        rew_explr_buffer = deque(maxlen=100)  # 初始化探索奖励缓冲区，最大长度为100
        rew_entropy_buffer = deque(maxlen=100)  # 初始化熵奖励缓冲区，最大长度为100
        lenbuffer = deque(maxlen=100)  # 初始化回合长度缓冲区，最大长度为100
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前奖励总和
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前探索奖励总和
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前熵奖励总和
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前回合长度

        task_rew_buf = deque(maxlen=100)  # 初始化任务奖励缓冲区，最大长度为100
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前任务奖励总和

        tot_iter = self.current_learning_iteration + num_learning_iterations  # 计算总迭代次数
        self.start_learning_iteration = copy(self.current_learning_iteration)  # 复制当前学习迭代次数

        for it in range(self.current_learning_iteration, tot_iter):  # 遍历每次迭代
            start = time.time()  # 记录开始时间
            hist_encoding = it % self.dagger_update_freq == 0  # 判断是否进行历史编码更新
            # Rollout
            with torch.inference_mode():  # 使用推理模式
                for i in range(self.num_steps_per_env):  # 遍历每个环境的步数
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)  # 获取动作
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # 执行动作，获取新的观测值、特权观测值、奖励、完成标志和信息
                    critic_obs = privileged_obs if privileged_obs is not None else obs  # 更新评论家观测值
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)  # 将观测值、评论家观测值、奖励和完成标志移动到指定设备
                    total_rew = self.alg.process_env_step(rewards, dones, infos)  # 处理环境步进，获取总奖励
                    
                    if self.log_dir is not None:  # 如果指定了日志目录
                        # Book keeping
                        if 'episode' in infos:  # 如果信息中包含回合信息
                            ep_infos.append(infos['episode'])  # 添加回合信息到列表
                        cur_reward_sum += total_rew  # 累加当前奖励总和
                        cur_reward_explr_sum += 0  # 累加当前探索奖励总和
                        cur_reward_entropy_sum += 0  # 累加当前熵奖励总和
                        cur_episode_length += 1  # 增加当前回合长度

                        new_ids = (dones > 0).nonzero(as_tuple=False)  # 获取完成标志为True的索引
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前奖励总和添加到奖励缓冲区
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前探索奖励总和添加到探索奖励缓冲区
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前熵奖励总和添加到熵奖励缓冲区
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())  # 将当前回合长度添加到回合长度缓冲区
                        
                        cur_reward_sum[new_ids] = 0  # 重置当前奖励总和
                        cur_reward_explr_sum[new_ids] = 0  # 重置当前探索奖励总和
                        cur_reward_entropy_sum[new_ids] = 0  # 重置当前熵奖励总和
                        cur_episode_length[new_ids] = 0  # 重置当前回合长度
                        # AMP
                        task_rew_buf.extend(cur_task_rew_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前任务奖励总和添加到任务奖励缓冲区
                        cur_task_rew_sum[new_ids] = 0  # 重置当前任务奖励总和
                stop = time.time()  # 记录停止时间
                collection_time = stop - start  # 计算收集时间

                # Learning step
                start = stop  # 更新开始时间
                self.alg.compute_returns(critic_obs)  # 计算回报
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef  = self.alg.update()  # 更新算法，获取损失和准确率
            if hist_encoding:  # 如果进行历史编码更新
                print("Updating dagger...")  # 打印更新信息
                mean_hist_latent_loss = self.alg.update_dagger()  # 更新历史编码
            
            stop = time.time()  # 记录停止时间
            learn_time = stop - start  # 计算学习时间
            if self.log_dir is not None:  # 如果指定了日志目录
                self.log(locals())  # 记录日志
            if it < 2500:  # 如果迭代次数小于2500
                if it % self.save_interval == 0:  # 如果迭代次数是保存间隔的倍数
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
            elif it < 5000:  # 如果迭代次数小于5000
                if it % (2*self.save_interval) == 0:  # 如果迭代次数是2倍保存间隔的倍数
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
            else:  # 如果迭代次数大于等于5000
                if it % (5*self.save_interval) == 0:  # 如果迭代次数是5倍保存间隔的倍数
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
            ep_infos.clear()  # 清空回合信息列表
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))  # 保存当前学习迭代次数的模型


    def learn_distill(self, num_learning_iterations, init_at_random_ep_len=False):  # 定义learn_distill方法，接受学习迭代次数和是否随机初始化参数
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 计算总迭代次数
        self.start_learning_iteration = copy(self.current_learning_iteration)  # 复制当前学习迭代次数

        ep_infos = []  # 初始化回合信息列表
        rewbuffer = deque(maxlen=100)  # 初始化奖励缓冲区，最大长度为100
        lenbuffer = deque(maxlen=100)  # 初始化长度缓冲区，最大长度为100
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前奖励总和
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前回合长度

        obs = self.env.get_observations()  # 获取环境的观测值
        infos = {}  # 初始化信息字典
        infos["decoder_demo_obs"] = self.env.get_decoder_demo_obs()  # 获取解码器演示观测值
        self.alg.student_actor.student_actor_backbone.train()  # 将学生演员的骨干网络切换到训练模式

        buffer_size = self.distill_cfg["num_steps_per_env"] * self.env.num_envs  # 计算缓冲区大小
        obs_teacher_buffer = ReplayBuffer(obs.shape[1], buffer_size, self.device)  # 初始化教师观测缓冲区
        decoder_demo_obs_buffer = ReplayBuffer(infos["decoder_demo_obs"].shape[1], buffer_size, self.device)  # 初始化解码器演示观测缓冲区

        num_pretrain_iter = self.distill_cfg["num_pretrain_iter"]  # 获取预训练迭代次数
        for it in range(self.current_learning_iteration, tot_iter):  # 遍历每次迭代
            start = time.time()  # 记录开始时间
            # actions_teacher_buffer = []
            # actions_student_buffer = []
            for i in range(self.distill_cfg["num_steps_per_env"]):  # 遍历每个环境的步数

                with torch.no_grad():  # 使用推理模式
                    actions_teacher = self.alg.student_actor.act_teacher(obs, hist_encoding=True)  # 获取教师动作
                    # actions_teacher_buffer.append(actions_teacher)

                obs_teacher_buffer.insert(obs)  # 插入观测值到教师观测缓冲区
                decoder_demo_obs_buffer.insert(infos["decoder_demo_obs"])  # 插入解码器演示观测值到缓冲区

                obs_student = obs.clone()  # 克隆观测值
                actions_student = self.alg.student_actor(obs_student, infos["decoder_demo_obs"], hist_encoding=True)  # 获取学生动作
                # actions_student_buffer.append(actions_student)

                if it < num_pretrain_iter:  # 如果当前迭代次数小于预训练迭代次数
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_teacher.detach())  # 使用教师动作执行环境步
                else:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())  # 使用学生动作执行环境步
                critic_obs = privileged_obs if privileged_obs is not None else obs  # 如果有特权观测值，则使用特权观测值，否则使用普通观测值
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)  # 将观测值、奖励和完成标志移动到指定设备

                if self.log_dir is not None:  # 如果日志目录不为空
                    # 记录信息
                    if 'episode' in infos:  # 如果信息中包含回合信息
                        ep_infos.append(infos['episode'])  # 将回合信息添加到列表中
                    cur_reward_sum += rewards  # 更新当前奖励总和
                    cur_episode_length += 1  # 更新当前回合长度
                    new_ids = (dones > 0).nonzero(as_tuple=False)  # 获取完成标志为True的索引
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())  # 将当前奖励总和添加到奖励缓冲区
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())  # 将当前回合长度添加到长度缓冲区
                    cur_reward_sum[new_ids] = 0  # 重置当前奖励总和
                    cur_episode_length[new_ids] = 0  # 重置当前回合长度

            stop = time.time()  # 记录停止时间
            collection_time = stop - start  # 计算收集时间
            start = stop  # 更新开始时间

            # actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            # actions_student_buffer = torch.cat(actions_student_buffer, dim=0)

            teacher_obs_generater = obs_teacher_buffer.feed_forward_generator(self.distill_cfg["num_mini_batches"], buffer_size // self.distill_cfg["num_mini_batches"])  # 创建教师观测生成器
            decoder_demo_obs_generater = decoder_demo_obs_buffer.feed_forward_generator(self.distill_cfg["num_mini_batches"], buffer_size // self.distill_cfg["num_mini_batches"])  # 创建解码器演示观测生成器
            losses = []  # 初始化损失列表
            for (teacher_obs, decoder_demo_obs) in zip(teacher_obs_generater, decoder_demo_obs_generater):  # 遍历生成器
                with torch.no_grad():  # 使用推理模式
                    actions_teacher_buffer = self.alg.student_actor.act_teacher(teacher_obs, hist_encoding=True)  # 获取教师动作缓冲区
                actions_student_buffer = self.alg.student_actor(teacher_obs, decoder_demo_obs, hist_encoding=True)  # 获取学生动作缓冲区
                distill_loss = self.alg.update_distill(actions_student_buffer, actions_teacher_buffer)  # 更新蒸馏损失
                losses.append(distill_loss)  # 将蒸馏损失添加到损失列表中
            distill_loss = np.mean(losses)  # 计算平均蒸馏损失

            stop = time.time()  # 记录停止时间
            learn_time = stop - start  # 计算学习时间

            if self.log_dir is not None:  # 如果日志目录不为空
                self.log_distill(locals())  # 记录蒸馏日志
            if (it - self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it - self.start_learning_iteration < 5000 and it % (2 * self.save_interval) == 0) or \
               (it - self.start_learning_iteration >= 5000 and it % (5 * self.save_interval) == 0):  # 根据迭代次数和保存间隔条件保存模型
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 保存模型
            ep_infos.clear()  # 清空回合信息列表
    
    def log_distill(self, locs, width=80, pad=35):  # 定义log_distill方法，接受日志信息、宽度和填充参数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs  # 更新总时间步数
        self.tot_time += locs['collection_time'] + locs['learn_time']  # 更新总时间
        iteration_time = locs['collection_time'] + locs['learn_time']  # 计算迭代时间

        ep_string = f''  # 初始化回合信息字符串
        wandb_dict = {}  # 初始化wandb字典
        if locs['ep_infos']:  # 如果回合信息不为空
            for key in locs['ep_infos'][0]:  # 遍历回合信息的键
                infotensor = torch.tensor([], device=self.device)  # 初始化信息张量
                for ep_info in locs['ep_infos']:  # 遍历每个回合信息
                    # 处理标量和零维张量信息
                    if not isinstance(ep_info[key], torch.Tensor):  # 如果信息不是张量
                        ep_info[key] = torch.Tensor([ep_info[key]])  # 将信息转换为张量
                    if len(ep_info[key].shape) == 0:  # 如果信息是零维张量
                        ep_info[key] = ep_info[key].unsqueeze(0)  # 将信息扩展为一维张量
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))  # 将信息张量连接到一起
                value = torch.mean(infotensor)  # 计算信息张量的均值
                if "tracking" in key:  # 如果键中包含"tracking"
                    wandb_dict['Episode_rew_tracking/' + key] = value  # 将信息添加到wandb字典中
                else:
                    wandb_dict['Episode_rew_regularization/' + key] = value  # 将信息添加到wandb字典中
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""  # 将信息添加到回合信息字符串中
        mean_std = self.alg.actor_critic.std.mean()  # 计算策略网络的标准差均值
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))  # 计算每秒帧数

        wandb_dict['Loss_distill/student_actor'] = locs['distill_loss']  # 将蒸馏损失添加到wandb字典中
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()  # 将噪声标准差均值添加到wandb字典中
        wandb_dict['Perf/total_fps'] = fps  # 将每秒帧数添加到wandb字典中
        wandb_dict['Perf/collection time'] = locs['collection_time']  # 将收集时间添加到wandb字典中
        wandb_dict['Perf/learning_time'] = locs['learn_time']  # 将学习时间添加到wandb字典中
        if len(locs['rewbuffer']) > 0:  # 如果奖励缓冲区不为空
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])  # 将平均奖励添加到wandb字典中
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])  # 将平均回合长度添加到wandb字典中

        wandb.log(wandb_dict, step=locs['it'])  # 记录wandb日志

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "  # 构建学习迭代信息字符串

        if len(locs['rewbuffer']) > 0:  # 如果奖励缓冲区不为空
            log_string = (f"""{'#' * width}\n"""  # 构建日志字符串
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        f"""{'Student actor loss:':>{pad}} {locs['distill_loss']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")  # 如果奖励缓冲区为空，构建简单的日志字符串

        log_string += f"""{'-' * width}\n"""  # 添加分隔线到日志字符串
        log_string += ep_string  # 添加回合信息字符串到日志字符串
        curr_it = locs['it'] - self.start_learning_iteration  # 计算当前迭代次数
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)  # 计算预计完成时间
        mins = eta // 60  # 计算预计完成时间的分钟部分
        secs = eta % 60  # 计算预计完成时间的秒部分
        log_string += (f"""{'-' * width}\n"""  # 添加预计完成时间信息到日志字符串
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)  # 打印日志字符串

    def log(self, locs, width=80, pad=35):  # 定义log方法，接受日志信息、宽度和填充参数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs  # 更新总时间步数
        self.tot_time += locs['collection_time'] + locs['learn_time']  # 更新总时间
        iteration_time = locs['collection_time'] + locs['learn_time']  # 计算迭代时间

        ep_string = f''  # 初始化回合信息字符串
        wandb_dict = {}  # 初始化wandb字典
        if locs['ep_infos']:  # 如果回合信息不为空
            for key in locs['ep_infos'][0]:  # 遍历回合信息的键
                infotensor = torch.tensor([], device=self.device)  # 初始化信息张量
                for ep_info in locs['ep_infos']:  # 遍历每个回合信息
                    # 处理标量和零维张量信息
                    if not isinstance(ep_info[key], torch.Tensor):  # 如果信息不是张量
                        ep_info[key] = torch.Tensor([ep_info[key]])  # 将信息转换为张量
                    if len(ep_info[key].shape) == 0:  # 如果信息是零维张量
                        ep_info[key] = ep_info[key].unsqueeze(0)  # 将信息扩展为一维张量
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))  # 将信息张量连接到一起
                value = torch.mean(infotensor)  # 计算信息张量的均值
                if "tracking" in key:  # 如果键中包含"tracking"
                    wandb_dict['Episode_rew_tracking/' + key] = value  # 将信息添加到wandb字典中
                elif "curriculum" in key:  # 如果键中包含"curriculum"
                    wandb_dict['Episode_curriculum/' + key] = value  # 将信息添加到wandb字典中
                else:
                    wandb_dict['Episode_rew_regularization/' + key] = value  # 将信息添加到wandb字典中
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""  # 将信息添加到回合信息字符串中

        mean_std = self.alg.actor_critic.std.mean()  # 计算策略网络的标准差均值
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))  # 计算每秒帧数

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']  # 将值函数损失添加到wandb字典中
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']  # 将代理损失添加到wandb字典中
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']  # 将熵系数添加到wandb字典中
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate  # 将学习率添加到wandb字典中
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Adaptation/estimator'] = locs['mean_estimator_loss']  # 将估计器损失添加到wandb字典中
        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']  # 将历史潜在损失添加到wandb字典中
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']  # 将隐私正则化损失添加到wandb字典中
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']  # 将隐私正则化系数添加到wandb字典中

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()  # 将噪声标准差均值添加到wandb字典中
        wandb_dict['Perf/total_fps'] = fps  # 将每秒帧数添加到wandb字典中
        wandb_dict['Perf/collection time'] = locs['collection_time']  # 将收集时间添加到wandb字典中
        wandb_dict['Perf/learning_time'] = locs['learn_time']  # 将学习时间添加到wandb字典中
        if len(locs['rewbuffer']) > 0:  # 如果奖励缓冲区不为空
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])  # 将平均奖励添加到wandb字典中
            # wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = statistics.mean(locs['task_rew_buf'])  # 将任务奖励添加到wandb字典中
            # wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])  # 将平均回合长度添加到wandb字典中
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])  # 记录wandb日志

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "  # 构建学习迭代信息字符串

        if len(locs['rewbuffer']) > 0:  # 如果奖励缓冲区不为空
            log_string = (f"""{'#' * width}\n"""  # 构建日志字符串
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                        f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['task_rew_buf']):.2f}\n"""
                        #   f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                        f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""  # 如果奖励缓冲区为空，构建简单的日志字符串
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""  # 添加分隔线到日志字符串
        log_string += ep_string  # 添加回合信息字符串到日志字符串
        curr_it = locs['it'] - self.start_learning_iteration  # 计算当前迭代次数
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)  # 计算预计完成时间
        mins = eta // 60  # 计算预计完成时间的分钟部分
        secs = eta % 60  # 计算预计完成时间的秒部分
        log_string += (f"""{'-' * width}\n"""  # 添加预计完成时间信息到日志字符串
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)  # 打印日志字符串

    def save(self, path, infos=None):  # 定义save方法，接受保存路径和可选的附加信息
        state_dict = {  # 创建状态字典
            'model_state_dict': self.alg.actor_critic.state_dict(),  # 保存actor_critic模型的状态字典
            'estimator_state_dict': self.alg.estimator.state_dict(),  # 保存估计器模型的状态字典
            'optimizer_state_dict': self.alg.optimizer.state_dict(),  # 保存优化器的状态字典
            'iter': self.current_learning_iteration,  # 保存当前学习迭代次数
            'infos': infos,  # 保存附加信息
        }
        if self.if_depth:  # 如果使用深度信息
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()  # 保存深度编码器的状态字典
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()  # 保存深度actor的状态字典
        if self.if_distill:  # 如果使用蒸馏
            state_dict['student_actor_state_dict'] = self.alg.student_actor.state_dict()  # 保存学生actor的状态字典
        torch.save(state_dict, path)  # 将状态字典保存到指定路径

    def load(self, path, load_optimizer=True):  # 定义load方法，接受模型路径和是否加载优化器的标志
        print("*" * 80)  # 打印分隔线
        print("Loading model from {}...".format(path))  # 打印加载模型的路径
        loaded_dict = torch.load(path, map_location=self.device)  # 加载模型字典到指定设备
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])  # 加载actor_critic模型的状态字典
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])  # 加载估计器模型的状态字典
        if self.if_depth:  # 如果使用深度信息
            if 'depth_encoder_state_dict' not in loaded_dict:  # 如果没有深度编码器的状态字典
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")  # 发出警告
            else:  # 如果有深度编码器的状态字典
                print("Saved depth encoder detected, loading...")  # 打印加载深度编码器的消息
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])  # 加载深度编码器的状态字典
            if 'depth_actor_state_dict' in loaded_dict:  # 如果有深度actor的状态字典
                print("Saved depth actor detected, loading...")  # 打印加载深度actor的消息
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])  # 加载深度actor的状态字典
            else:  # 如果没有深度actor的状态字典
                print("No saved depth actor, Copying actor critic actor to depth actor...")  # 打印复制actor_critic的actor到深度actor的消息
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())  # 复制actor_critic的actor到深度actor
        if self.if_distill:  # 如果使用蒸馏
            try:  # 尝试加载学生actor的状态字典
                self.alg.student_actor.load_state_dict(loaded_dict['student_actor_state_dict'])  # 加载学生actor的状态字典
                print("Saved student actor detected, loading...")  # 打印加载学生actor的消息
            except:  # 如果没有学生actor的状态字典
                print("No saved student actor")  # 打印没有学生actor的消息
        if load_optimizer:  # 如果需要加载优化器
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])  # 加载优化器的状态字典
        # self.current_learning_iteration = loaded_dict['iter']  # 更新当前学习迭代次数（注释掉）
        print("*" * 80)  # 打印分隔线
        return loaded_dict['infos']  # 返回附加信息

    def get_inference_policy(self, device=None):  # 定义get_inference_policy方法，接受设备参数
        self.alg.actor_critic.eval()  # 将actor_critic模型切换到评估模式（例如，禁用dropout）
        if device is not None:  # 如果指定了设备
            self.alg.actor_critic.to(device)  # 将actor_critic模型移动到指定设备
        return self.alg.actor_critic.act_inference  # 返回actor_critic模型的推理方法

    def get_depth_actor_inference_policy(self, device=None):  # 定义get_depth_actor_inference_policy方法，接受设备参数
        self.alg.depth_actor.eval()  # 将深度actor模型切换到评估模式（例如，禁用dropout）
        if device is not None:  # 如果指定了设备
            self.alg.depth_actor.to(device)  # 将深度actor模型移动到指定设备
        return self.alg.depth_actor  # 返回深度actor模型

    def get_actor_critic(self, device=None):  # 定义get_actor_critic方法，接受设备参数
        self.alg.actor_critic.eval()  # 将actor_critic模型切换到评估模式（例如，禁用dropout）
        if device is not None:  # 如果指定了设备
            self.alg.actor_critic.to(device)  # 将actor_critic模型移动到指定设备
        return self.alg.actor_critic  # 返回actor_critic模型

    def get_estimator_inference_policy(self, device=None):  # 定义get_estimator_inference_policy方法，接受设备参数
        self.alg.estimator.eval()  # 将估计器模型切换到评估模式（例如，禁用dropout）
        if device is not None:  # 如果指定了设备
            self.alg.estimator.to(device)  # 将估计器模型移动到指定设备
        return self.alg.estimator.inference  # 返回估计器模型的推理方法

    def get_depth_encoder_inference_policy(self, device=None):  # 定义get_depth_encoder_inference_policy方法，接受设备参数
        self.alg.depth_encoder.eval()  # 将深度编码器模型切换到评估模式
        if device is not None:  # 如果指定了设备
            self.alg.depth_encoder.to(device)  # 将深度编码器模型移动到指定设备
        return self.alg.depth_encoder  # 返回深度编码器模型

    def get_disc_inference_policy(self, device=None):  # 定义get_disc_inference_policy方法，接受设备参数
        self.alg.discriminator.eval()  # 将判别器模型切换到评估模式（例如，禁用dropout）
        if device is not None:  # 如果指定了设备
            self.alg.discriminator.to(device)  # 将判别器模型移动到指定设备
        return self.alg.discriminator.inference  # 返回判别器模型的推理方法
