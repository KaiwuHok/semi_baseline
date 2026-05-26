#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 1.5,
        "tower_hp_point": 8.0,
        "money": 0.006,
        "exp": 0.006,
        "ep_rate": 0.6,
        "death": -1.0,
        "kill": 0.8,
        "last_hit": 0.6,
        "forward": 0.01,
        "crab_kill": 0.7,
        "crab_vision": 0.3,
        "skill_hit_hero": 0.15,
        "skill2atk_combo": 0.25,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 8000
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    # 叠帧后总特征维度: ally_hero(64×8)+enemy_hero(64×8)+enemy_tower(15×16)
    # +ally_tower(5×8)+ally_soldier(57×8)+enemy_soldier(57×8)+crab(11×6)=2282
    DIM_OF_FEATURE = [2282]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    # 每步数据拆分形状: feature(2282)+legal_action(85)=2367, reward_sum(1), advantage(1),
    # action(6), is_train(1), old_probs(12+16+16+16+16+9=85), weights(7), lstm(512+512)
    DATA_SPLIT_SHAPE = [
        2282 + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(2282,), (85,)]
    INIT_LEARNING_RATE_START = 1.2e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.03
    BETA_END = 0.005
    BETA_DECAY_STEPS = 20000
    LOG_EPSILON = 1e-6

    # Entropy adaptive beta parameters (方案C — 加速收敛)
    # 熵自适应beta参数
    TARGET_ENTROPY = 7.0
    P0_ADAPTIVE_UP_RATE = 0.02
    P0_ADAPTIVE_DOWN_RATE = 0.1
    P0_BETA_MAX = 0.08
    P0_BETA_MIN = 0.0001
    P0_MAX_UP_RATE = 0.3
    P0_MAX_DOWN_RATE = 1.5
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    # data_shapes: LSTM_TIME_STEPS=16 步展平后的各字段总维数
    # [0]: (2282+85)*16=37872 (feature+legal_action展平)
    # [1-8]: 8×16 (reward(1), advantage(1), action(6)) 各乘16步
    # [9-13]: old_probs (12*16, 16*16, 16*16, 16*16, 16*16)
    # [14]: old_probs最后一维 9*16
    # [15-21]: 7×16 weights
    # [22-23]: lstm_cell(512), lstm_hidden(512)
    data_shapes = [
        [(2282 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
