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

import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入os库，用于操作系统相关功能
from datetime import datetime  # 导入datetime库，用于处理日期和时间

import isaacgym  # 导入isaacgym库，NVIDIA的物理仿真库
from legged_gym.envs import *  # 从legged_gym.envs模块导入所有内容
from legged_gym.utils import get_args, task_registry  # 从legged_gym.utils模块导入get_args和task_registry
from shutil import copyfile  # 导入shutil库中的copyfile函数，用于文件复制
import torch  # 导入PyTorch库，用于深度学习
import wandb  # 导入Weights & Biases库，用于实验跟踪和可视化

def train(args):  # 定义train函数，接受一个参数args
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid  # 构建日志路径
    try:
        os.makedirs(log_pth)  # 尝试创建日志目录
    except:
        pass  # 如果目录已存在，则忽略异常
    if args.debug:  # 如果处于调试模式
        mode = "disabled"  # 设置wandb模式为禁用
        args.rows = 10  # 设置环境行数为10
        args.cols = 5  # 设置环境列数为5
        args.num_envs = 64  # 设置环境数量为64
    else:
        mode = "online"  # 否则设置wandb模式为在线
    
    if args.no_wandb:  # 如果不使用wandb
        mode = "disabled"  # 设置wandb模式为禁用
    wandb.init(project=args.proj_name, name=args.exptid, entity=args.entity, mode=mode, dir="../../logs")  # 初始化wandb
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")  # 保存配置文件到wandb
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")  # 保存机器人文件到wandb
    wandb.save(LEGGED_GYM_ENVS_DIR + "/h1/h1_mimic_config.py", policy="now")  # 保存h1配置文件到wandb
    wandb.save(LEGGED_GYM_ENVS_DIR + "/h1/h1_mimic.py", policy="now")  # 保存h1文件到wandb

    env, env_cfg = task_registry.make_env(name=args.task, args=args)  # 创建环境和环境配置
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)  # 创建PPO算法运行器和训练配置
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)  # 开始训练，指定最大迭代次数和随机初始化

if __name__ == '__main__':  # 如果脚本作为主程序运行
    args = get_args()  # 获取命令行参数
    args.task = "h1_view"  # 设置任务为h1_view
    args.max_iterations = 10 # 设置最大迭代次数为10
    args.exptid = '000-05'  # 设置实验ID为当前日期和时间
    args.motion_name = "motions_jump.yaml"  # 设置运动名称
    args.motion_type = "yaml"  # 设置运动类型
    args.headless = True  # 设置args.headless为True，表示无头模式
    train(args)  # 调用train函数进行训练
