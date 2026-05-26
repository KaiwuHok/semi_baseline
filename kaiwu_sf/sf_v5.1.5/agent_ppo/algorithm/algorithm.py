#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        # Extract sample field from SampleData objects and stack into tensor
        # 从 SampleData 对象中提取 sample 字段并 stack 成 tensor
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        # Expand LSTM hidden/cell states to match the 16x timestep-unrolled batch
        # 将LSTM隐藏/细胞状态扩展以匹配16倍时间步展开后的batch大小
        lstm_hidden_state = lstm_hidden_state.repeat_interleave(Config.LSTM_TIME_STEPS, dim=0)
        lstm_cell_state = lstm_cell_state.repeat_interleave(Config.LSTM_TIME_STEPS, dim=0)

        # 准备自回归 Teacher Forcing 所需的真实标签
        labels = []
        for i in range(len(Config.LABEL_SIZE_LIST)):
            label = data_list[3 + i].reshape(-1).long()
            labels.append(label)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state] + labels

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 梯度剪裁
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        # beta decay + entropy adaptive adjustment
        # beta衰减 + 熵自适应调节
        self._update_beta(_info_list)

    def _update_beta(self, _info_list):
        """
        Beta decay + entropy adaptive adjustment.
        beta = decayed_beta * adaptive_rate, clamped to [P0_BETA_MIN, P0_BETA_MAX].

        beta衰减 + 熵自适应调节。
        beta = 衰减后的beta × 自适应比率, 截断到 [P0_BETA_MIN, P0_BETA_MAX]。
        """
        # linear decay: beta(t) = BETA_START - (BETA_START - BETA_END) * min(t / BETA_DECAY_STEPS, 1.0)
        # 线性衰减
        decay_ratio = min(self.train_step / Config.BETA_DECAY_STEPS, 1.0)
        beta_decayed = Config.BETA_START - (Config.BETA_START - Config.BETA_END) * decay_ratio

        # entropy adaptive: rate = 1.0 + clamp(entropy_gap * rate, -max_down, max_up)
        # 熵自适应
        _, (_, _, entropy_cost) = _info_list
        current_entropy = abs(entropy_cost)  # entropy_cost is -H, so abs gives actual H
        entropy_gap = Config.TARGET_ENTROPY - current_entropy

        if entropy_gap > 0:
            adaptive_rate = 1.0 + min(entropy_gap * Config.P0_ADAPTIVE_UP_RATE, Config.P0_MAX_UP_RATE)
        else:
            adaptive_rate = 1.0 + max(entropy_gap * Config.P0_ADAPTIVE_DOWN_RATE, -Config.P0_MAX_DOWN_RATE)

        beta_new = beta_decayed * adaptive_rate
        beta_new = max(Config.P0_BETA_MIN, min(Config.P0_BETA_MAX, beta_new))
        self.model.var_beta = beta_new
