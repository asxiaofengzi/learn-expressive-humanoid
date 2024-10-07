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

import os  # 导入操作系统模块
import copy  # 导入拷贝模块
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
import random  # 导入随机数生成模块
from isaacgym import gymapi  # 从Isaac Gym导入gymapi模块
from isaacgym import gymutil  # 从Isaac Gym导入gymutil模块
import argparse  # 导入命令行参数解析模块

from pygments.lexer import default

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 从legged_gym导入根目录和环境目录常量


def class_to_dict(obj) -> dict:
    # 将类对象转换为字典
    if not hasattr(obj, "__dict__"):
        # 如果对象没有__dict__属性，直接返回对象
        return obj
    result = {}  # 初始化结果字典
    for key in dir(obj):
        # 遍历对象的所有属性
        if key.startswith("_"):
            # 跳过以"_"开头的私有属性
            continue
        element = []  # 初始化元素列表
        val = getattr(obj, key)  # 获取属性值
        if isinstance(val, list):
            # 如果属性值是列表
            for item in val:
                # 遍历列表中的每个元素
                element.append(class_to_dict(item))  # 递归调用class_to_dict
        else:
            element = class_to_dict(val)  # 递归调用class_to_dict
        result[key] = element  # 将结果添加到字典中
    return result  # 返回结果字典


def update_class_from_dict(obj, dict):
    # 从字典更新类对象的属性
    for key, val in dict.items():
        # 遍历字典中的每个键值对
        attr = getattr(obj, key, None)  # 获取对象的属性值
        if isinstance(attr, type):
            # 如果属性值是类型
            update_class_from_dict(attr, val)  # 递归调用update_class_from_dict
        else:
            setattr(obj, key, val)  # 设置对象的属性值
    return


def set_seed(seed):
    # 设置随机数种子
    if seed == -1:
        # 如果种子为-1，则生成一个随机种子
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))  # 打印设置的种子

    random.seed(seed)  # 设置Python的随机数种子
    np.random.seed(seed)  # 设置NumPy的随机数种子
    torch.manual_seed(seed)  # 设置PyTorch的随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    torch.cuda.manual_seed(seed)  # 设置CUDA的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的随机数种子


def parse_sim_params(args, cfg):
    # 解析模拟参数
    # 从Isaac Gym Preview 2代码初始化模拟参数
    sim_params = gymapi.SimParams()

    # 从命令行参数设置一些值
    if args.physics_engine == gymapi.SIM_FLEX:
        # 如果物理引擎是FLEX
        if args.device != "cpu":
            # 如果设备不是CPU，打印警告信息
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 如果物理引擎是PHYSX
        sim_params.physx.use_gpu = args.use_gpu  # 设置是否使用GPU
        sim_params.physx.num_subscenes = args.subscenes  # 设置子场景数量
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline  # 设置是否使用GPU管道

    # 如果在配置中提供了模拟选项，解析它们并更新/覆盖上述设置
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果在命令行中传递了num_threads，覆盖默认值
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads  # 设置线程数量

    return sim_params  # 返回模拟参数


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    # 如果root不是一个目录，则尝试匹配前4个字符以找到运行名称
    if not os.path.isdir(root):
        model_name_cand = os.path.basename(root)  # 获取root的基本名称
        model_parent = os.path.dirname(root)  # 获取root的父目录
        model_names = os.listdir(model_parent)  # 列出父目录中的所有文件和目录
        # 过滤出父目录中的所有目录
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                # 如果目录名称的前6个字符与model_name_cand匹配，则更新root
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    # 如果checkpoint为-1，则加载最新的模型
    if checkpoint == -1:
        # 列出root目录中包含model_name_include的所有文件
        models = [file for file in os.listdir(root) if model_name_include in file]
        # 按文件名排序
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]  # 获取最新的模型文件
    else:
        # 否则，加载指定的checkpoint模型
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)  # 构建模型的完整路径
    return load_path  # 返回模型路径


def update_cfg_from_args(env_cfg, cfg_train, args):
    # 更新环境配置和训练配置
    if env_cfg is not None:
        if args.use_camera:
            env_cfg.depth.use_camera = args.use_camera  # 设置是否使用摄像头
        if env_cfg.depth.use_camera and args.headless:  # 如果使用摄像头且为无头模式，设置摄像头相关参数
            env_cfg.env.num_envs = env_cfg.depth.camera_num_envs
            env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows
            env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols
            env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera
            env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera
            env_cfg.terrain.simplify_grid = True
            env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_flat"] = 0.05
            env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_step"] = 0.2
            env_cfg.terrain.terrain_dict["demo"] = 0.15
            env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        if env_cfg.depth.use_camera:
            env_cfg.terrain.y_range = [-0.1, 0.1]  # 设置地形的y轴范围

        if args.task == "h1_view":
            env_cfg.env.num_envs = 1  # 如果任务是h1_view，则设置环境数量为1
        if args.motion_name is not None:
            env_cfg.motion.motion_name = args.motion_name  # 设置运动名称
        if args.motion_type is not None:
            env_cfg.motion.motion_type = args.motion_type  # 设置运动类型
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs  # 设置环境数量
        if args.seed is not None:
            env_cfg.seed = args.seed  # 设置随机种子
        if args.task_both:
            env_cfg.env.task_both = args.task_both  # 设置是否同时执行两个任务
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows  # 设置地形行数
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols  # 设置地形列数
        if args.delay:
            env_cfg.domain_rand.action_delay = args.delay  # 设置动作延迟
        if not args.delay and not args.resume and not args.use_camera and args.headless:  # 如果从头开始训练
            env_cfg.domain_rand.action_delay = True
            env_cfg.domain_rand.action_curr_step = env_cfg.domain_rand.action_curr_step_scratch
        if args.record_video:
            env_cfg.env.record_video = args.record_video  # 设置是否录制视频
        if args.record_frame:
            env_cfg.env.record_frame = args.record_frame  # 设置是否录制帧
        if args.fix_base:
            env_cfg.asset.fix_base_link = args.fix_base  # 设置是否固定基础链接
        if args.regen_pkl:
            env_cfg.motion.regen_pkl = args.regen_pkl  # 设置是否重新生成pkl文件
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed  # 设置随机种子
        if args.use_camera:
            cfg_train.depth_encoder.if_depth = args.use_camera  # 设置是否使用深度编码器
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations  # 设置最大迭代次数
        if args.resume:
            cfg_train.runner.resume = args.resume  # 设置是否恢复训练
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name  # 设置实验名称
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name  # 设置运行名称
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run  # 设置加载的运行名称
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint  # 设置检查点

    return env_cfg, cfg_train  # 返回更新后的配置


def get_args():
    custom_parameters = [
        # 定义一个名为 "--task" 的字符串参数，默认值为 "h1_view"，用于从检查点恢复训练或开始测试。可以覆盖配置文件中的设置。
        {"name": "--task", "type": str, "default": "h1_mimic",
         "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},

        # 定义一个名为 "--resume" 的布尔参数，默认值为 False，用于指示是否从检查点恢复训练。
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},

        # 定义一个名为 "--experiment_name" 的字符串参数，用于指定实验的名称。可以覆盖配置文件中的设置。
        {"name": "--experiment_name", "type": str,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},

        # 定义一个名为 "--run_name" 的字符串参数，用于指定运行的名称。可以覆盖配置文件中的设置。
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},

        # 定义一个名为 "--load_run" 的字符串参数，用于在 resume=True 时指定要加载的运行名称。如果值为 -1，则加载最后一次运行。可以覆盖配置文件中的设置。
        {"name": "--load_run", "type": str,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},

        # 定义一个名为 "--checkpoint" 的整数参数，默认值为 -1，用于指定要加载的模型检查点编号。如果值为 -1，则加载最后一个检查点。可以覆盖配置文件中的设置。
        {"name": "--checkpoint", "type": int, "default": -1,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        # 定义一个名为 "--headless" 的布尔参数，默认值为 False，用于强制关闭显示。
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},

        # 定义一个名为 "--horovod" 的布尔参数，默认值为 False，用于指示是否使用 Horovod 进行多 GPU 训练。
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},

        # 定义一个名为 "--rl_device" 的字符串参数，默认值为 "cuda:0"，用于指定强化学习算法使用的设备（如 CPU、GPU 等）。
        {"name": "--rl_device", "type": str, "default": "cuda:0",
         "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},

        # 定义一个名为 "--num_envs" 的整数参数，用于指定要创建的环境数量。可以覆盖配置文件中的设置。
        {"name": "--num_envs", "type": int,
         "help": "Number of environments to create. Overrides config file if provided."},

        # 定义一个名为 "--seed" 的整数参数，用于指定随机种子。可以覆盖配置文件中的设置。
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},

        # 定义一个名为 "--max_iterations" 的整数参数，用于指定最大训练迭代次数。可以覆盖配置文件中的设置。
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},

        # 定义一个名为 "--device" 的字符串参数，默认值为 "cuda:0"，用于指定用于模拟、强化学习和图形的设备。
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},

        # 定义一个名为 "--rows" 的整数参数，用于指定行数。
        {"name": "--rows", "type": int, "help": "num_rows."},

        # 定义一个名为 "--cols" 的整数参数，用于指定列数。
        {"name": "--cols", "type": int, "help": "num_cols"},

        # 定义一个名为 "--debug" 的布尔参数，默认值为 True，用于禁用 wandb 日志记录。
        {"name": "--debug", "action": "store_true", "default": True, "help": "Disable wandb logging"},

        # 定义一个名为 "--proj_name" 的字符串参数，默认值为 "h1"，用于指定运行文件夹的名称。
        {"name": "--proj_name", "type": str, "default": "h1", "help": "run folder name."},

        # 定义一个名为 "--teacher" 的字符串参数，用于指定在蒸馏过程中使用的教师策略的名称。
        {"name": "--teacher", "type": str, "help": "Name of the teacher policy to use when distilling"},

        # 定义一个名为 "exptid" 的字符串参数，用于指定实验 ID。
        {"name": "--exptid", "type": str, "default": "060-40", "help": "exptid"},

        # 定义一个名为 "--entity" 的字符串参数，默认值为空字符串，用于指定 wandb 实体。
        {"name": "--entity", "type": str, "default": "", "help": "wandb entity"},

        # 定义一个名为 "--resumeid" 的字符串参数，用于指定实验 ID。
        {"name": "--resumeid", "type": str, "help": "exptid"},

        # 定义一个名为 "--daggerid" 的字符串参数，用于指定 Dagger 运行的名称。
        {"name": "--daggerid", "type": str, "help": "name of dagger run"},

        # 定义一个名为 "--use_camera" 的布尔参数，默认值为 False，用于指示是否在蒸馏过程中渲染相机。
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "render camera for distillation"},

        # 定义一个名为 "--mask_obs" 的布尔参数，默认值为 False，用于在游戏过程中屏蔽观察。
        {"name": "--mask_obs", "action": "store_true", "default": False, "help": "Mask observation when playing"},

        # 定义一个名为 "--use_jit" 的布尔参数，默认值为 False，用于在游戏过程中加载 JIT 脚本。
        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},

        # 定义一个名为 "--use_latent" 的布尔参数，默认值为 False，用于在游戏过程中加载深度潜在变量。
        {"name": "--use_latent", "action": "store_true", "default": False, "help": "Load depth latent when playing"},

        # 定义一个名为 "--draw" 的布尔参数，默认值为 False，用于在游戏过程中绘制调试图。
        {"name": "--draw", "action": "store_true", "default": False, "help": "draw debug plot when playing"},

        # 定义一个名为 "--save" 的布尔参数，默认值为 False，用于保存评估数据。
        {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},

        # 定义一个名为 "--task_both" 的布尔参数，默认值为 False，用于指示是否同时使用攀爬和击打策略。
        {"name": "--task_both", "action": "store_true", "default": False, "help": "Both climbing and hitting policies"},

        # 定义一个名为 "--nodelay" 的布尔参数，默认值为 False，用于添加动作延迟。
        {"name": "--nodelay", "action": "store_true", "default": False, "help": "Add action delay"},

        # 定义一个名为 "--delay" 的布尔参数，默认值为 False，用于添加动作延迟。
        {"name": "--delay", "action": "store_true", "default": False, "help": "Add action delay"},

        # 定义一个名为 "--hitid" 的字符串参数，默认值为 None，用于指定击打策略的实验 ID。
        {"name": "--hitid", "type": str, "default": None, "help": "exptid fot hitting policy"},

        # 定义一个名为 "--web" 的布尔参数，默认值为 False，用于指示是否使用 Web 查看器。
        {"name": "--web", "action": "store_true", "default": True, "help": "if use web viewer"},

        # 定义一个名为 "--no_wandb" 的布尔参数，默认值为 False，用于禁用 wandb。
        {"name": "--no_wandb", "action": "store_true", "default": True, "help": "no wandb"},

        # 定义一个名为 "--motion_name" 的字符串参数，默认值为 "motions_autogen_all.yaml"，用于指定用于生成关键身体位置的运动名称。
        {"name": "--motion_name", "type": str, "default": "motions_jump.yaml",
         "help": "motion name used for generating key body positions"},

        # 定义一个名为 "--motion_type" 的字符串参数，用于指定用于生成关键身体位置的运动类型。
        {"name": "--motion_type", "type": str, "help": "motion type used for generating key body positions"},

        # 定义一个名为 "--record_video" 的布尔参数，默认值为 False，用于记录视频。
        {"name": "--record_video", "action": "store_true", "default": False, "help": "record video"},

        # 定义一个名为 "--record_frame" 的布尔参数，默认值为 False，用于记录视频帧。
        {"name": "--record_frame", "action": "store_true", "default": False, "help": "record video"},

        # 定义一个名为 "--record_data" 的布尔参数，默认值为 False，用于记录数据。
        {"name": "--record_data", "action": "store_true", "default": False, "help": "record video"},

        # 定义一个名为 "--fix_base" 的布尔参数，默认值为 False，用于固定基础链接。
        {"name": "--fix_base", "action": "store_true", "default": False, "help": "fix base link"},

        # 定义一个名为 "--regen_pkl" 的布尔参数，默认值为 False，用于重新生成 pkl 文件。
        {"name": "--regen_pkl", "action": "store_true", "default": False, "help": "re-generate pkl file"},
    ]

    # 解析命令行参数
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    args.no_wandb = True

    # 打印命令行参数
    # print(args)

    # #设置默认参数值
    # args.task = "h1_view"
    # args.resume = False
    # args.experiment_name = "default_experiment"
    # args.run_name = "default_run"
    # args.load_run = -1
    # args.checkpoint = -1
    # args.headless = False
    # args.horovod = False
    # args.rl_device = "cuda:0"
    # args.num_envs = 1
    # args.seed = 42
    # args.max_iterations = 10000
    # args.device = "cuda:0"
    # args.rows = 10
    # args.cols = 10
    # args.debug = True
    # args.proj_name = "h1"
    # args.teacher = None
    # args.exptid = "debug"
    # args.entity = ""
    # args.resumeid = None
    # args.daggerid = None
    # args.use_camera = False
    # args.mask_obs = False
    # args.use_jit = False
    # args.use_latent = False
    # args.draw = False
    # args.save = False
    # args.task_both = False
    # args.nodelay = False
    # args.delay = False
    # args.hitid = None
    # args.web = False
    # args.no_wandb = False
    # args.motion_name = "motions_autogen_all.yaml"
    # args.motion_type = None
    # args.record_video = False
    # args.record_frame = False
    # args.record_data = False
    # args.fix_base = False
    # args.regen_pkl = False

    # 名称对齐
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args  # 返回解析后的参数


def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, 'memory_a'):
        # 假设是LSTM：TODO 添加GRU支持
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)  # 创建路径
        path = os.path.join(path, name + ".pt")  # 构建模型保存路径
        model = copy.deepcopy(actor_critic.actor).to('cpu')  # 深拷贝模型并移动到CPU
        traced_script_module = torch.jit.script(model)  # 使用TorchScript跟踪模型
        traced_script_module.save(path)  # 保存跟踪的模型


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()  # 调用父类的初始化方法
        self.actor = copy.deepcopy(actor_critic.actor)  # 深拷贝actor_critic的actor部分
        self.is_recurrent = actor_critic.is_recurrent  # 获取actor_critic的is_recurrent属性
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)  # 深拷贝actor_critic的RNN部分
        self.memory.cpu()  # 将RNN部分移动到CPU上
        # 注册hidden_state缓冲区，初始化为全零张量
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        # 注册cell_state缓冲区，初始化为全零张量
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        # 将输入x扩展维度并传入RNN，获取输出和新的隐藏状态、细胞状态
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # 更新hidden_state
        self.cell_state[:] = c  # 更新cell_state
        return self.actor(out.squeeze(0))  # 将RNN的输出传入actor并返回结果

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.  # 将hidden_state重置为全零
        self.cell_state[:] = 0.  # 将cell_state重置为全零

    def export(self, path):
        os.makedirs(path, exist_ok=True)  # 创建目录，如果目录不存在则创建
        path = os.path.join(path, 'policy_lstm_1.pt')  # 构建模型保存路径
        self.to('cpu')  # 将模型移动到CPU上
        traced_script_module = torch.jit.script(self)  # 使用TorchScript跟踪模型
        traced_script_module.save(path)  # 保存跟踪的模型


# 覆盖gymutil中的parse_device_str函数
def parse_device_str(device_str):
    # 默认值
    device = 'cpu'  # 默认设备为CPU
    device_id = 0  # 默认设备ID为0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str  # 如果设备字符串为'cpu'或'cuda'，则直接使用
        device_id = 0  # 设备ID为0
    else:
        device_args = device_str.split(':')  # 分割设备字符串
        # 确保设备字符串格式正确
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args  # 获取设备类型和设备ID字符串
        try:
            device_id = int(device_id_s)  # 尝试将设备ID字符串转换为整数
        except ValueError:
            # 如果转换失败，抛出异常
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id  # 返回设备类型和设备ID


def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)  # 创建ArgumentParser对象
    if headless:
        # 如果headless为True，添加'--headless'参数
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        # 如果no_graphics为True，添加'--nographics'参数
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    # 添加'sim_device'参数，默认值为'cuda:0'
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    # 添加'pipeline'参数，默认值为'gpu'
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    # 添加'graphics_device_id'参数，默认值为0
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()  # 创建互斥参数组
    # 添加'--flex'参数，用于选择FleX物理引擎
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    # 添加'--physx'参数，用于选择PhysX物理引擎
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    # 添加'num_threads'参数，默认值为0
    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    # 添加'subscenes'参数，默认值为0
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    # 添加'slices'参数
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        # 遍历自定义参数列表
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]  # 获取帮助字符串

            if "type" in argument:
                if "default" in argument:
                    # 添加带有默认值的参数
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"],
                                        help=help_str)
                else:
                    # 添加不带默认值的参数
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                # 添加带有动作的参数
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            # 如果参数不包含"name"和"type/action"键，则打印错误信息
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()  # 解析命令行参数

    # 设置默认参数值
    # args.task = "h1_view"
    # args.resume = False
    # args.experiment_name = "default_experiment"
    # args.run_name = "default_run"
    # args.load_run = -1
    # args.checkpoint = -1
    # args.headless = False
    # args.horovod = False
    # args.rl_device = "cuda:0"
    # args.num_envs = 1
    # args.seed = 42
    # args.max_iterations = 10000
    # args.device = "cuda:0"
    # args.rows = 10
    # args.cols = 10
    # args.debug = True
    # args.proj_name = "h1"
    # args.teacher = None
    # args.exptid = "debug"
    # args.entity = ""
    # args.resumeid = None
    # args.daggerid = None
    # args.use_camera = False
    # args.mask_obs = False
    # args.use_jit = False
    # args.use_latent = False
    # args.draw = False
    # args.save = False
    # args.task_both = False
    # args.nodelay = False
    # args.delay = False
    # args.hitid = None
    # args.web = False
    # args.no_wandb = False
    # args.motion_name = "motions_autogen_all.yaml"
    # args.motion_type = None
    # args.record_video = False
    # args.record_frame = False
    # args.record_data = False
    # args.fix_base = False
    # args.regen_pkl = False

    if args.device is not None:
        args.sim_device = args.device  # 设置sim_device
        args.rl_device = args.device  # 设置rl_device
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)  # 解析设备字符串
    pipeline = args.pipeline.lower()  # 将pipeline转换为小写

    # 确保pipeline值有效
    assert (pipeline == 'cpu' or pipeline in (
    'gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))  # 设置是否使用GPU管道

    if args.sim_device_type != 'cuda' and args.flex:
        # 如果设备类型不是'cuda'且选择了FleX，打印警告信息并更改设备为'cuda:0'
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        # 如果设备类型不是'cuda'且pipeline为'gpu'，打印警告信息并更改pipeline为'CPU'
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # 默认使用PhysX物理引擎
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')  # 设置是否使用GPU

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX  # 如果选择了FleX，设置物理引擎为FleX

    # 使用--nographics隐含--headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes  # 如果slices未设置，使用subscenes的值

    return args  # 返回解析后的参数