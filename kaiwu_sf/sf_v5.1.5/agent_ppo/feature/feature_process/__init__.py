#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import collections

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess
from agent_ppo.feature.feature_process.crab_process import CrabProcess


# ========== 叠帧配置 ==========
# 各分组的叠帧窗口大小（以决策步为单位，1步=6帧=198ms）
# 原则：变化快（英雄）叠8帧(1.6s)，变化慢（敌方塔）叠16帧(3.2s)
STACK_CONFIG = {
    "ally_hero":     8,    # 1.6s，1套完整技能连招
    "enemy_hero":    8,    # 同上，参数独立
    "enemy_tower":   16,   # 3.2s，与LSTM窗口等长，可见血量衰减曲线
    "ally_tower":    8,    # 1.6s，己方塔被推时变化较快
    "ally_soldier":  8,    # 1.6s，推进/撤退方向稳定可辨
    "enemy_soldier": 8,    # 同上
    "crab":          6,    # 1.2s，移动慢，6步够看方向
}


# ========== 特征分组维度注册表 ==========
# 定义各分组的名称、叠后维度和在 2282 维平铺向量中的切片位置
# 叠后维度 = 单帧维度 × 叠帧数
# 用于：显式管理分组边界，供模型层在 forward 入口做维度切片还原分组
FEATURE_GROUP_DEFS = [
    # (分组名称,          叠后维度, 切片起始, 切片结束, 说明)
    ("ally_hero",           512,     0,   512,  "己方英雄 64×8"),
    ("enemy_hero",          512,   512,  1024,  "敌方英雄 64×8"),
    ("enemy_tower",         240,  1024,  1264,  "敌方防御塔 15×16"),
    ("ally_tower",           40,  1264,  1304,  "己方防御塔 5×8"),
    ("ally_soldier",        456,  1304,  1760,  "己方小兵 57×8"),
    ("enemy_soldier",       456,  1760,  2216,  "敌方小兵 57×8"),
    ("crab",                 66,  2216,  2282,  "河蟹 11×6"),
    # diff 分组已移除：叠帧+Conv1D 沿时间轴卷积已完全取代显式差分
]

# 自动校验总维度
_TOTAL_DIM = sum(dim for _, dim, _, _, _ in FEATURE_GROUP_DEFS)
assert _TOTAL_DIM == 2282, f"Feature group dimension mismatch: {_TOTAL_DIM} != 2282"

# 各分组维度常量，便于模型层导入使用
GROUP_DIMS = {name: dim for name, dim, _, _, _ in FEATURE_GROUP_DEFS}
# 特征向量中各组分的切片边界列表: [0, 512, 1024, 1264, 1304, 1760, 2216, 2282]
GROUP_SPLIT_BOUNDARIES = [start for _, _, start, _, _ in FEATURE_GROUP_DEFS] + [2282]

# 单帧维度（每种实体的原始特征维数，不含叠帧倍数）
PER_FRAME_DIM = {
    "ally_hero":     64,
    "enemy_hero":    64,
    "enemy_tower":   15,
    "ally_tower":     5,
    "ally_soldier":  57,
    "enemy_soldier": 57,
    "crab":          11,
}


class FeatureProcess:
    """特征处理主入口。

    通过叠帧缓冲区（deque）为各分组维护多步历史帧，展平拼接为 2282 维 1D 向量。
    模型入口通过 reshape 恢复 [channel=feat_dim, length=stack] 的时序结构后，
    用 Conv1D 沿时间轴做卷积。

    diff 分组已移除：叠帧 + Conv1D 时间轴卷积提供了更强的时序提取能力，
    显式差分成为冗余。
    """

    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.crab_process = CrabProcess(camp)

        # 叠帧缓冲区：每个分组一个定长 deque，按时间顺序（旧→新）存储各帧特征
        # 首次调用 process_feature 时填满当前帧副本
        self.buffers = {}
        self._buffers_initialized = False

    def _init_buffers(self):
        """延迟初始化叠帧缓冲区（避免在 __init__ 中重复创建）"""
        self.buffers = {
            name: collections.deque(maxlen=stack)
            for name, stack in STACK_CONFIG.items()
        }
        self._buffers_initialized = True

    def reset(self, camp):
        """重置 FeatureProcess 状态（新 Episodd 开始时调用）"""
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.crab_process = CrabProcess(camp)

        # 清空叠帧缓冲区，并在下次 process_feature 时重新初始化
        self._buffers_initialized = False
        self.buffers.clear()

    def process_feature(self, observation):
        """提取 2282 维叠帧平铺特征向量。

        流程：
        1. 各 Process 提取当前帧的分组特征（与旧版一致）
        2. 各分组特征入队到对应的定长 deque 缓冲区
        3. 首次调用时用当前帧副本填满所有缓冲区（保证维度一致）
        4. 按 FEATURE_GROUP_DEFS 顺序将各分组缓冲区展平拼接为 2282 维
        """
        # 延迟初始化缓冲区
        if not self._buffers_initialized:
            self._init_buffers()

        frame_state = observation["frame_state"]

        # --- 第1步：提取各分组当前帧特征 ---
        # 英雄特征：己方64维 + 敌方64维
        main_camp_hero_vec, enemy_camp_hero_vec = self.hero_process.process_vec_hero(frame_state)

        # 防御塔特征：敌方塔15维 + 己方塔5维
        enemy_organ_vec, ally_organ_vec = self.organ_process.process_vec_organ(frame_state)

        # 小兵特征：己方57维(3×19) + 敌方57维(3×19)
        ally_soldier_vec, enemy_soldier_vec = self.soldier_process.process_vec_soldier(frame_state)

        # 河蟹特征：11维
        crab_vec = self.crab_process.process_vec_crab(frame_state)

        # --- 第2步：各分组特征入队 ---
        current_features = {
            "ally_hero":     main_camp_hero_vec,
            "enemy_hero":    enemy_camp_hero_vec,
            "enemy_tower":   enemy_organ_vec,
            "ally_tower":    ally_organ_vec,
            "ally_soldier":  ally_soldier_vec,
            "enemy_soldier": enemy_soldier_vec,
            "crab":          crab_vec,
        }

        for name, feat in current_features.items():
            self.buffers[name].append(feat)

        # 首次调用：所有缓冲区用当前帧副本填满
        # 这确保了从第1步起 deque 始终满窗，返回维度固定为 2282
        if any(len(self.buffers[name]) < STACK_CONFIG[name] for name in STACK_CONFIG):
            self._fill_buffers(current_features)

        # --- 第3步：按分组注册表顺序展平拼接为 2282 维 ---
        # 对每个分组：遍历其 deque 中的所有帧（时间从旧到新），逐帧展平
        feature = []
        for name, _, _, _, _ in FEATURE_GROUP_DEFS:
            for frame_feat in self.buffers[name]:
                feature.extend(frame_feat)

        return feature

    def _fill_buffers(self, current_features):
        """首次调用时将当前帧特征复制填满各缓冲区。

        例如 ally_hero 需要叠 8 帧：
        第1步: deque = [f1, f1, f1, f1, f1, f1, f1, f1]
        第2步: deque = [f1, f1, ..., f2]
        ...
        第8步后: deque = [f1, f2, ..., f8] 全部为真实帧

        这是强化学习叠帧的标准做法（如 Atari DQN 的 frame stack）。
        """
        for name, stack in STACK_CONFIG.items():
            buf = self.buffers[name]
            feat = current_features[name]
            # 用当前帧特征重复填充直到 deque 满
            while len(buf) < stack:
                buf.append(feat)
