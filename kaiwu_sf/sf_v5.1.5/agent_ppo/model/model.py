#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


class StackConvEncoder(nn.Module):
    """叠帧分组编码器：沿时间轴做 Conv1D 提取时序特征。

    设计原则（方案C）：特征为通道，时间为卷积极轴。
    输入展平叠帧 [batch, feat_dim * stack]，内部 reshape 为 [batch, feat_dim, stack]，
    Conv1D 仅沿时间轴（dim=-1）滑动。多层 Conv1D + AdaptiveAvgPool1d(1) 将时间轴压缩到 1。

    参数：
        in_channels: 单帧特征维数（如 64）
        stack:      叠帧窗口大小（如 8）
        channels:   通道转换列表（如 [48, 32]），表示 in_c→48→32 两层 Conv1D
        kernel_size: 卷积核大小（默认 3）
    """
    def __init__(self, in_channels, stack, channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.stack = stack

        conv_layers = []
        prev_c = in_channels
        for next_c in channels:
            conv = nn.Conv1d(prev_c, next_c, kernel_size=kernel_size, stride=1, padding=0)
            nn.init.orthogonal(conv.weight)
            nn.init.zeros_(conv.bias)
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU())
            prev_c = next_c

        self.convs = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch, in_channels * stack] 展平叠帧
        batch_size = x.shape[0]
        # 恢复时序结构：特征为通道，时间为长度
        x = x.reshape(batch_size, self.in_channels, self.stack)  # [B, C, T]  # 修改：view → reshape
        # Conv1D 沿时间轴提取时序特征
        x = self.convs(x)        # [B, C_out, T']
        # 全局池化：将剩余时间步压缩到 1
        x = self.pool(x)         # [B, C_out, 1]
        x = x.squeeze(-1)        # [B, C_out]
        return x


class StackSoldierConvEncoder(nn.Module):
    """小兵叠帧编码器：按个体独立编码，保留个体+时序结构。

    每个小兵的特征独立为一组 [per_unit_dim, stack] 做 Conv1D 时序编码，
    三个小兵共享同一套 Conv1D 权重，最后 flatten 拼接。

    参数：
        per_unit_dim:  单兵单帧特征维数（如 19）
        num_units:     小兵个数（3）
        stack:         叠帧窗口大小（如 8）
        channels:      通道转换列表（如 [12, 8]）
        kernel_size:   卷积核大小（默认 3）
    """
    def __init__(self, per_unit_dim, num_units, stack, channels, kernel_size=3):
        super().__init__()
        self.num_units = num_units
        self.per_unit_dim = per_unit_dim
        self.stack = stack

        # 为每个小兵构建共享权重的 Conv1D 编码器
        conv_layers = []
        prev_c = per_unit_dim
        for next_c in channels:
            conv = nn.Conv1d(prev_c, next_c, kernel_size=kernel_size, stride=1, padding=0)
            nn.init.orthogonal(conv.weight)
            nn.init.zeros_(conv.bias)
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU())
            prev_c = next_c

        self.unit_convs = nn.Sequential(*conv_layers)
        self.unit_pool = nn.AdaptiveAvgPool1d(1)
        self.output_channels = channels[-1]

    def forward(self, x):
        # x: [batch, num_units * per_unit_dim * stack]
        batch_size = x.shape[0]
        unit_dim = self.per_unit_dim * self.stack  # 每个兵的展平维度

        # 拆分为个体，共享权重处理
        x = x.reshape(batch_size * self.num_units, unit_dim)           # [B×3, per_unit_dim*stack]  # 修改：view → reshape
        x = x.reshape(batch_size * self.num_units, self.per_unit_dim, self.stack)  # [B×3, C, T]  # 修改：view → reshape

        # Conv1D 沿时间轴
        x = self.unit_convs(x)     # [B×3, C_out, T']
        x = self.unit_pool(x)      # [B×3, C_out, 1]
        x = x.squeeze(-1)          # [B×3, C_out]

        # 拼接回 3 兵
        x = x.reshape(batch_size, self.num_units * self.output_channels)  # [B, 3 × C_out]
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])

        # ========== 分组独立编码器（叠帧 Conv1D） ==========
        # 各分组叠帧输入通过 reshape 恢复 [channel=feat_dim, length=stack] 结构，
        # 用 Conv1D 沿时间轴做卷积提取时序特征，输出总维度 138。
        #
        # 英雄 (64维→叠8帧→32维):
        #   Conv1D(64→48,k=3)→Conv1D(48→32,k=3)→Pool(1)
        # 敌方塔 (15维→叠16帧→12维):
        #   Conv1D(15→24,k=3)→Conv1D(24→16,k=3)→Conv1D(16→12,k=3)→Pool(1)
        # 己方塔 (5维→叠8帧→6维):
        #   Conv1D(5→8,k=3)→Conv1D(8→6,k=3)→Pool(1)
        # 小兵 (19维→叠8帧→8维/兵×3兵=24维，共享权重):
        #   Conv1D(19→12,k=3)→Conv1D(12→8,k=3)→Pool(1)
        # 河蟹 (11维→叠6帧→8维):
        #   Conv1D(11→16,k=3)→Conv1D(16→8,k=3)→Pool(1)
        # diff 分组已移除
        self.group_encoders = nn.ModuleDict({
            "ally_hero":     StackConvEncoder(64, 8, [48, 32], kernel_size=3),
            "enemy_hero":    StackConvEncoder(64, 8, [48, 32], kernel_size=3),
            "enemy_tower":   StackConvEncoder(15, 16, [24, 16, 12], kernel_size=3),
            "ally_tower":    StackConvEncoder(5, 8, [8, 6], kernel_size=3),
            "ally_soldier":  StackSoldierConvEncoder(19, 3, 8, [12, 8], kernel_size=3),
            "enemy_soldier": StackSoldierConvEncoder(19, 3, 8, [12, 8], kernel_size=3),
            "crab":          StackConvEncoder(11, 6, [16, 8], kernel_size=3),
        })
        # 编码后总维度: 32+32+12+6+24+24+8 = 138
        # ==========================================

        # concat_mlp: 融合各分组编码后的特征，138→256→256
        fc_concat_dim_list = [138, 256, 256]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)

        # LSTM: input_size=256(concat_mlp输出), hidden_size=512
        self.lstm = torch.nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        # 自回归动作 Embedding 层：将离散动作编号映射为16维连续向量
        self.action_embed_main = nn.Embedding(12, 16)
        self.action_embed_skill1 = nn.Embedding(16, 16)
        self.action_embed_skill2 = nn.Embedding(16, 16)
        self.action_embed_skill3 = nn.Embedding(16, 16)
        self.action_embed_summoner = nn.Embedding(16, 16)

        # 自回归 label_mlp：输入维度随前置动作 Embedding 逐步增大
        # 基向量为 lstm_out(512维)，每一步拼接前置动作的 16 维 Embedding
        # 隐层从原来的 256 降至 128（分组编码提高了上游特征质量，动作头不需过宽）
        self.label_mlp = ModuleDict(
            {
                "hero_label0_mlp": MLP([512, 128, self.label_size_list[0]], "hero_label0_mlp"),
                "hero_label1_mlp": MLP([528, 128, self.label_size_list[1]], "hero_label1_mlp"),
                "hero_label2_mlp": MLP([544, 128, self.label_size_list[2]], "hero_label2_mlp"),
                "hero_label3_mlp": MLP([560, 128, self.label_size_list[3]], "hero_label3_mlp"),
                "hero_label4_mlp": MLP([576, 128, self.label_size_list[4]], "hero_label4_mlp"),
                "hero_label5_mlp": MLP([592, 128, self.label_size_list[5]], "hero_label5_mlp"),
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        self.value_mlp = MLP([512, 128, 1], "hero_value_mlp")

        self.target_embed_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        if inference:
            feature_vec, lstm_hidden_init, lstm_cell_init = data_list
            labels = None
        else:
            feature_vec, lstm_hidden_init, lstm_cell_init = data_list[:3]
            labels = data_list[3:]

        result_list = []

        # ========== 分组独立编码（叠帧 Conv1D） ==========
        # 将 2282 维平铺叠帧特征按分组切片，各分组内 reshape 为 [C, T] 后沿时间轴做 Conv1D
        # 切片边界: [0, 512, 1024, 1264, 1304, 1760, 2216, 2282]
        # ally_hero(512) + enemy_hero(512) + enemy_tower(240) + ally_tower(40)
        # + ally_soldier(456) + enemy_soldier(456) + crab(66) = 2282
        # diff 分组已移除：叠帧+Conv1D 已完全取代显式差分
        g = feature_vec
        ally_hero_enc     = self.group_encoders["ally_hero"](    g[:,    0:512])
        enemy_hero_enc    = self.group_encoders["enemy_hero"](   g[:,  512:1024])
        enemy_tower_enc   = self.group_encoders["enemy_tower"](  g[:, 1024:1264])
        ally_tower_enc    = self.group_encoders["ally_tower"](   g[:, 1264:1304])
        ally_soldier_enc  = self.group_encoders["ally_soldier"]( g[:, 1304:1760])
        enemy_soldier_enc = self.group_encoders["enemy_soldier"](g[:, 1760:2216])
        crab_enc          = self.group_encoders["crab"](         g[:, 2216:2282])

        # 拼接编码后特征 → [batch, 138]
        encoded_feature = torch.cat([
            ally_hero_enc, enemy_hero_enc,
            enemy_tower_enc, ally_tower_enc,
            ally_soldier_enc, enemy_soldier_enc,
            crab_enc,
        ], dim=1)
        # =======================================

        # public concat
        # 公共连接层
        fc_public_result = self.concat_mlp(encoded_feature)

        # LSTM
        lstm_input = fc_public_result.unsqueeze(1)  # [batch, seq_len=1, 256]
        lstm_hidden = (lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0))
        lstm_out, (lstm_hidden_out, lstm_cell_out) = self.lstm(lstm_input, lstm_hidden)
        lstm_out = lstm_out.squeeze(1)  # [batch, 512]

        self.lstm_cell_output = lstm_cell_out
        self.lstm_hidden_output = lstm_hidden_out

        if inference:
            # 推理时：自回归 argmax 采样
            # main
            main_logits = self.label_mlp["hero_label0_mlp"](lstm_out)
            main_action = torch.argmax(main_logits, dim=1)
            main_embed = self.action_embed_main(main_action)

            # skill1
            skill1_input = torch.cat([lstm_out, main_embed], dim=1)
            skill1_logits = self.label_mlp["hero_label1_mlp"](skill1_input)
            skill1_action = torch.argmax(skill1_logits, dim=1)
            skill1_embed = self.action_embed_skill1(skill1_action)

            # skill2
            skill2_input = torch.cat([lstm_out, main_embed, skill1_embed], dim=1)
            skill2_logits = self.label_mlp["hero_label2_mlp"](skill2_input)
            skill2_action = torch.argmax(skill2_logits, dim=1)
            skill2_embed = self.action_embed_skill2(skill2_action)

            # skill3
            skill3_input = torch.cat([lstm_out, main_embed, skill1_embed, skill2_embed], dim=1)
            skill3_logits = self.label_mlp["hero_label3_mlp"](skill3_input)
            skill3_action = torch.argmax(skill3_logits, dim=1)
            skill3_embed = self.action_embed_skill3(skill3_action)

            # summoner
            summoner_input = torch.cat([lstm_out, main_embed, skill1_embed, skill2_embed, skill3_embed], dim=1)
            summoner_logits = self.label_mlp["hero_label4_mlp"](summoner_input)
            summoner_action = torch.argmax(summoner_logits, dim=1)
            summoner_embed = self.action_embed_summoner(summoner_action)

            # target
            target_input = torch.cat(
                [lstm_out, main_embed, skill1_embed, skill2_embed, skill3_embed, summoner_embed], dim=1
            )
            target_logits = self.label_mlp["hero_label5_mlp"](target_input)

            # concat all logits
            logits = torch.cat(
                [main_logits, skill1_logits, skill2_logits, skill3_logits, summoner_logits, target_logits], dim=1
            )

            # value
            value_result = self.value_mlp(lstm_out)

            return [logits, value_result, self.lstm_cell_output, self.lstm_hidden_output]

        else:
            # 训练时：Teacher Forcing，使用真实标签
            main_label = labels[0]
            skill1_label = labels[1]
            skill2_label = labels[2]
            skill3_label = labels[3]
            summoner_label = labels[4]
            target_label = labels[5]

            # main
            main_logits = self.label_mlp["hero_label0_mlp"](lstm_out)
            result_list.append(main_logits)
            main_embed = self.action_embed_main(main_label)

            # skill1
            skill1_input = torch.cat([lstm_out, main_embed], dim=1)
            skill1_logits = self.label_mlp["hero_label1_mlp"](skill1_input)
            result_list.append(skill1_logits)
            skill1_embed = self.action_embed_skill1(skill1_label)

            # skill2
            skill2_input = torch.cat([lstm_out, main_embed, skill1_embed], dim=1)
            skill2_logits = self.label_mlp["hero_label2_mlp"](skill2_input)
            result_list.append(skill2_logits)
            skill2_embed = self.action_embed_skill2(skill2_label)

            # skill3
            skill3_input = torch.cat([lstm_out, main_embed, skill1_embed, skill2_embed], dim=1)
            skill3_logits = self.label_mlp["hero_label3_mlp"](skill3_input)
            result_list.append(skill3_logits)
            skill3_embed = self.action_embed_skill3(skill3_label)

            # summoner
            summoner_input = torch.cat([lstm_out, main_embed, skill1_embed, skill2_embed, skill3_embed], dim=1)
            summoner_logits = self.label_mlp["hero_label4_mlp"](summoner_input)
            result_list.append(summoner_logits)
            summoner_embed = self.action_embed_summoner(summoner_label)

            # target
            target_input = torch.cat(
                [lstm_out, main_embed, skill1_embed, skill2_embed, skill3_embed, summoner_embed], dim=1
            )
            target_logits = self.label_mlp["hero_label5_mlp"](target_input)
            result_list.append(target_logits)

            # value
            value_result = self.value_mlp(lstm_out)
            result_list.append(value_result)

            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        # 用于熵损失计算
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * frame_is_train
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * frame_is_train), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * frame_is_train)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)