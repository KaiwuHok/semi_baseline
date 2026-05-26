#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        # 英雄总特征维度：is_hero_alive(1)+location_x(1)+location_z(1)+hp_rate(1)+hp_discrete(5)+ep_rate(1)+ep_discrete(3)
        # +level(15)+money_rate(1)+skill_0~5_cd_rate(6)+skill_0~5_cd_state(18)+in_tower_range(1)+is_tower_target(1)+behav_mode(9)=64
        self.one_unit_feature_num = 64
        self.unit_buff_num = 1
        self.frame_state = None

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):
        self.frame_state = frame_state
        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        # Generate hero features for enemy camp
        # 生成敌方阵营的英雄特征
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        # 特征分组提取：分别返回己方英雄(64维)和敌方英雄(64维)
        return (main_camp_hero_vector_feature, enemy_camp_hero_vector_feature)

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    # ============ 血量相关 ============

    def get_hp_rate(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0 or hero["max_hp"] <= 0:
            vector_feature.append(0)
            return
        value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_hp_discrete(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0 or hero["max_hp"] <= 0:
            vector_feature.append(0)  # raw value 0 → bucket 0 after one_hot
            return
        hp_rate = hero["hp"] / hero["max_hp"]
        if hp_rate < 0.3:
            bucket = 0
        elif hp_rate < 0.5:
            bucket = 1
        elif hp_rate < 0.7:
            bucket = 2
        elif hp_rate < 0.9:
            bucket = 3
        else:
            bucket = 4
        vector_feature.append(bucket)

    # ============ 法力相关 ============

    def get_ep_rate(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0 or hero["max_ep"] <= 0:
            vector_feature.append(0)
            return
        value = hero["ep"] / hero["max_ep"]
        vector_feature.append(value)

    def get_ep_discrete(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0 or hero["max_ep"] <= 0:
            vector_feature.append(0)
            return
        ep_rate = hero["ep"] / hero["max_ep"]
        if ep_rate < 0.3:
            bucket = 0
        elif ep_rate < 0.7:
            bucket = 1
        else:
            bucket = 2
        vector_feature.append(bucket)

    # ============ 等级 ============

    def get_level(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)
            return
        level = hero.get("level", 1)
        # level ∈ [1, 15] → index ∈ [0, 14]
        bucket = max(0, min(14, level - 1))
        vector_feature.append(bucket)

    # ============ 经济 ============

    def get_money_rate(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)
            return
        money_cnt = hero.get("money_cnt", 0)
        value = money_cnt / 10000.0
        vector_feature.append(value)

    # ============ 技能冷却 ============

    def _get_skill_slot(self, hero, slot_idx):
        """获取指定技能槽的 SkillSlotState，不存在返回 None"""
        skill_state = hero.get("skill_state", {})
        slot_states = skill_state.get("slot_states", [])
        if not slot_states:
            return None
        for slot in slot_states:
            if slot.get("slot_type") == slot_idx:
                return slot
        return None

    def _is_skill_learned(self, slot):
        """技能是否已学习：configId > 0（协议中为camelCase，兼容config_id）"""
        if slot is None:
            return False
        config_id = slot.get("configId", 0) or slot.get("config_id", 0)
        return config_id > 0

    def get_skill_cd_rate(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)
            return
        # feature_name like "skill_0_cd_rate"
        slot_idx = int(feature_name.split("_")[1])
        slot = self._get_skill_slot(hero, slot_idx)
        if slot is None or not self._is_skill_learned(slot):
            vector_feature.append(0)
            return
        cooldown = slot.get("cooldown", 0)
        cooldown_max = slot.get("cooldown_max", 1)
        if cooldown_max <= 0:
            vector_feature.append(0)
            return
        value = cooldown / cooldown_max
        vector_feature.append(value)

    def get_skill_cd_state(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(2)  # 不可用
            return
        # feature_name like "skill_0_cd_state"
        slot_idx = int(feature_name.split("_")[1])
        slot = self._get_skill_slot(hero, slot_idx)
        if slot is None or not self._is_skill_learned(slot):
            vector_feature.append(2)  # 不可用
            return
        if slot.get("usable", False):
            vector_feature.append(0)  # 可用
        else:
            vector_feature.append(1)  # 冷却中

    # ============ 塔下关系 ============

    def _get_enemy_towers(self):
        """获取敌方防御塔列表（sub_type==21, camp!=己方阵营）"""
        towers = []
        if self.frame_state is None:
            return towers
        for npc in self.frame_state.get("npc_states", []):
            if npc.get("sub_type") == 21 and npc.get("camp") != self.main_camp:
                towers.append(npc)
        return towers

    def _cal_dist_2d(self, pos1, pos2):
        """计算两个位置之间的欧几里得距离（使用原始坐标）"""
        dx = pos1["x"] - pos2["x"]
        dz = pos1["z"] - pos2["z"]
        return math.sqrt(dx * dx + dz * dz)

    def get_in_tower_range(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)
            return
        hero_loc = hero["location"]
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(0)
            return
        enemy_towers = self._get_enemy_towers()
        for tower in enemy_towers:
            tower_loc = tower.get("location", {})
            tower_range = tower.get("attack_range", 0)
            if tower_range <= 0:
                continue
            dist = self._cal_dist_2d(hero_loc, tower_loc)
            if dist <= tower_range:
                vector_feature.append(1)
                return
        vector_feature.append(0)

    def get_is_tower_target(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)
            return
        hero_runtime_id = hero.get("runtime_id", -1)
        if hero_runtime_id < 0:
            vector_feature.append(0)
            return
        enemy_towers = self._get_enemy_towers()
        for tower in enemy_towers:
            if tower.get("attack_target") == hero_runtime_id:
                vector_feature.append(1)
                return
        vector_feature.append(0)

    # ============ 行为状态 ============

    def get_behav_mode(self, hero, vector_feature, feature_name):
        if hero["hp"] <= 0:
            vector_feature.append(0)  # 死亡
            return
        behav_mode = hero.get("behav_mode", 1)
        # 0=死亡,1=空闲,2=移动,3=普攻,4=复活,5=技能1,6=技能2,7=技能3,8=其他
        if behav_mode < 0 or behav_mode > 8:
            behav_mode = 8  # 归入"其他"
        vector_feature.append(behav_mode)
