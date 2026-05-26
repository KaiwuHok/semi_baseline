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


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp

        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}

        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        # 敌方塔特征维度：is_alive(1)+belong_to_main_camp(1)+location_x(1)+location_z(1)
        # +relative_location_x(1)+relative_location_z(1)+hp_rate(1)=7 (现有)
        # +enemy_tower_hp_rate(1)+enemy_tower_hp_discrete(3)+enemy_tower_attack_target(3)+dist_to_enemy_tower(1)=8 (新增)
        self.one_unit_feature_num = 15
        self.unit_buff_num = 1
        self.frame_state = None

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

        # Get ally-specific feature config and function map
        # 获取己方塔特征的配置和函数映射
        self.ally_feature_config = []
        for feature, config in self.config["ally_feature_config"].items():
            self.ally_feature_config.append(f"{feature}:{config}")
        self.map_ally_feature_to_norm = self.normalizer.parse_config(self.ally_feature_config)

        self.ally_feature_func_map = {}
        for feature, func_name in self.config["ally_feature_functions"].items():
            if hasattr(self, func_name):
                self.ally_feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, frame_state):
        self.frame_state = frame_state
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        # Generate features for enemy team's towers
        # 生成敌方阵营的防御塔特征（15维）
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")

        # Generate features for ally team's towers
        # 生成己方阵营的防御塔特征（5维）
        ally_camp_organ_vector_feature = self.generate_ally_organ_feature(self.main_camp_organ_dict)

        # 特征分组提取：分别返回敌方塔(15维)和己方塔(5维)
        return (enemy_camp_organ_vector_feature, ally_camp_organ_vector_feature)

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == self.main_camp:
                if organ_subtype == 21:
                    self.main_camp_organ_dict["tower"] = organ
            else:
                if organ_subtype == 21:
                    self.enemy_camp_organ_dict["tower"] = organ

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
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
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    # ============ 敌方塔新增特征 ============

    def get_enemy_tower_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0 and organ["hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def get_enemy_tower_hp_discrete(self, organ, vector_feature):
        if organ["hp"] <= 0 or organ["max_hp"] <= 0:
            vector_feature.append(0)
            return
        hp_rate = organ["hp"] / organ["max_hp"]
        if hp_rate < 0.3:
            bucket = 0
        elif hp_rate < 0.7:
            bucket = 1
        else:
            bucket = 2
        vector_feature.append(bucket)

    def get_enemy_tower_attack_target(self, organ, vector_feature):
        if organ["hp"] <= 0:
            vector_feature.append(0)
            return
        target_id = organ.get("attack_target", 0)
        if target_id <= 0:
            vector_feature.append(0)
            return
        if self.frame_state is None:
            vector_feature.append(0)
            return
        # 在英雄列表中查找攻击目标
        for hero in self.frame_state.get("hero_states", []):
            if hero.get("runtime_id") == target_id:
                vector_feature.append(1)
                return
        # 在NPC列表中查找攻击目标（排除防御塔自身 sub_type==21）
        for npc in self.frame_state.get("npc_states", []):
            if npc.get("sub_type") != 21 and npc.get("runtime_id") == target_id:
                vector_feature.append(2)
                return
        vector_feature.append(0)

    def get_dist_to_enemy_tower(self, organ, vector_feature):
        if organ["hp"] <= 0:
            vector_feature.append(1)
            return
        hero_loc = self.main_hero_info.get("location", {})
        tower_loc = organ.get("location", {})
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(1)
            return
        dist = math.sqrt(
            (hero_loc["x"] - tower_loc["x"]) ** 2 + (hero_loc["z"] - tower_loc["z"]) ** 2
        )
        normalized = dist / 12000.0
        vector_feature.append(min(normalized, 1.0))

    # ============ 己方塔特征 ============

    def get_ally_tower_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0 and organ["hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def get_ally_tower_hp_discrete(self, organ, vector_feature):
        if organ["hp"] <= 0 or organ["max_hp"] <= 0:
            vector_feature.append(0)
            return
        hp_rate = organ["hp"] / organ["max_hp"]
        if hp_rate < 0.3:
            bucket = 0
        elif hp_rate < 0.7:
            bucket = 1
        else:
            bucket = 2
        vector_feature.append(bucket)

    def get_dist_to_ally_tower(self, organ, vector_feature):
        if organ["hp"] <= 0:
            vector_feature.append(1)
            return
        hero_loc = self.main_hero_info.get("location", {})
        tower_loc = organ.get("location", {})
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(1)
            return
        dist = math.sqrt(
            (hero_loc["x"] - tower_loc["x"]) ** 2 + (hero_loc["z"] - tower_loc["z"]) ** 2
        )
        normalized = dist / 12000.0
        vector_feature.append(min(normalized, 1.0))

    def generate_ally_organ_feature(self, one_type_organ_info):
        """仅生成己方塔特征（ally_tower_hp_rate, ally_tower_hp_discrete, dist_to_ally_tower）"""
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            for feature_name, feature_func in self.ally_feature_func_map.items():
                value = []
                feature_func(organ, value)
                if feature_name not in self.map_ally_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_ally_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        # 己方塔不存在时填充5维0值
        if num_organs_considered < self.unit_buff_num:
            for _ in range((self.unit_buff_num - num_organs_considered) * 5):
                vector_feature.append(0)
        return vector_feature

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, organ, vector_feature):
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 0.0
        if organ["camp"] == self.main_hero_info["camp"]:
            value = 1.0
        vector_feature.append(value)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def relative_location_x(self, organ, vector_feature):
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def relative_location_z(self, organ, vector_feature):
        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)
