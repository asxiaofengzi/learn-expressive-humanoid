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

from copy import deepcopy  # 导入deepcopy函数，用于深度复制对象
import os  # 导入os库，用于操作系统相关功能
from datetime import datetime  # 导入datetime库，用于处理日期和时间
from typing import Tuple  # 导入Tuple类型，用于类型注解
import torch  # 导入PyTorch库，用于深度学习
import numpy as np  # 导入NumPy库，用于数值计算

from rsl_rl.env import VecEnv  # 从rsl_rl.env模块导入VecEnv类，用于向量化环境
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerMimic, OnPolicyRunnerMimicAMP  # 从rsl_rl.runners模块导入多个运行器类

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 从legged_gym模块导入根目录和环境目录常量
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params  # 从helpers模块导入多个辅助函数
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO  # 从legged_robot_config模块导入配置类

class TaskRegistry():  # 定义TaskRegistry类
    def __init__(self):  # 初始化方法
        self.task_classes = {}  # 初始化任务类字典
        self.env_cfgs = {}  # 初始化环境配置字典
        self.train_cfgs = {}  # 初始化训练配置字典
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):  # 注册任务方法
        self.task_classes[name] = task_class  # 将任务类存入字典
        self.env_cfgs[name] = env_cfg  # 将环境配置存入字典
        self.train_cfgs[name] = train_cfg  # 将训练配置存入字典
    
    def get_task_class(self, name: str) -> VecEnv:  # 获取任务类方法
        return self.task_classes[name]  # 返回任务类
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:  # 获取配置方法
        train_cfg = self.train_cfgs[name]  # 获取训练配置
        env_cfg = self.env_cfgs[name]  # 获取环境配置
        env_cfg.seed = train_cfg.seed  # 将训练配置的种子复制到环境配置
        return env_cfg, train_cfg  # 返回环境配置和训练配置
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:  # 创建环境方法
        if args is None:  # 如果没有传入args
            args = get_args()  # 获取命令行参数
        if name in self.task_classes:  # 如果任务类字典中存在该任务
            task_class = self.get_task_class(name)  # 获取任务类
        else:
            raise ValueError(f"Task with name: {name} was not registered")  # 抛出任务未注册的错误
        if env_cfg is None:  # 如果没有传入环境配置
            env_cfg, _ = self.get_cfgs(name)  # 获取环境配置
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)  # 从args更新配置
        set_seed(env_cfg.seed)  # 设置随机种子
        sim_params = {"sim": class_to_dict(env_cfg.sim)}  # 将环境配置转换为字典
        sim_params = parse_sim_params(args, sim_params)  # 解析仿真参数
        env = task_class(  # 创建环境实例
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless
        )
        return env, env_cfg  # 返回环境实例和环境配置

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, init_wandb=True, log_root="default", **kwargs):  # 创建算法运行器方法
        if args is None:  # 如果没有传入args
            args = get_args()  # 获取命令行参数
        if train_cfg is None:  # 如果没有传入训练配置
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")  # 抛出错误
            _, train_cfg = self.get_cfgs(name)  # 获取训练配置
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")  # 忽略name参数
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)  # 从args更新配置
        
        if log_root == "default":  # 如果日志根目录为默认值
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)  # 设置日志根目录
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)  # 设置日志目录
        elif log_root is None:
            log_dir = None  # 不设置日志目录
        else:
            log_dir = log_root  # 使用传入的日志根目录
        
        train_cfg_dict = class_to_dict(train_cfg)  # 将训练配置转换为字典
        runner_class = eval(train_cfg.runner.runner_class_name)  # 获取运行器类
        runner = runner_class(  # 创建运行器实例
            env, 
            train_cfg_dict, 
            log_dir, 
            init_wandb=init_wandb,
            device=args.rl_device, **kwargs
        )
        resume = train_cfg.runner.resume  # 获取是否恢复训练
        if args.resumeid:  # 如果传入了恢复ID
            log_root = LEGGED_GYM_ROOT_DIR + f"/logs/{args.proj_name}/" + args.resumeid  # 设置日志根目录
            resume = True  # 设置恢复训练为True
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)  # 获取恢复路径
            runner.load(resume_path)  # 加载模型
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 19, device=runner.device)  # 重置标准差

        if "return_log_dir" in kwargs:  # 如果需要返回日志目录
            return runner, train_cfg, os.path.dirname(resume_path)  # 返回运行器、训练配置和日志目录
        else:    
            return runner, train_cfg  # 返回运行器和训练配置

task_registry = TaskRegistry()  # 创建全局任务注册表实例
