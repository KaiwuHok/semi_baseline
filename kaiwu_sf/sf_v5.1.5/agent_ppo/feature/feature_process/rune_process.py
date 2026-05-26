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


class RuneProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_rune_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.rune_feature_config)
        self.frame_state = None
        self.main_hero_info = None

    def get_rune_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "rune_feature_config.ini")
        self.config.read(config_path)

        self.rune_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.rune_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_rune(self, frame_state):
        self.frame_state = frame_state
        self._set_main_hero_info(frame_state)

        # 获取双方防御塔位置用于归属判断
        ally_tower_loc = self._get_ally_tower_loc(frame_state)
        enemy_tower_loc = self._get_enemy_tower_loc(frame_state)

        # 遍历神符，判断归属
        ally_runes = []
        enemy_runes = []
        for cake in frame_state.get("cakes", []):
            collider = cake.get("collider")
            if collider is None:
                continue
            rune_loc = collider.get("location")
            if rune_loc is None:
                continue
            # 判断神符归属：离哪方防御塔更近就属于哪方
            if ally_tower_loc and enemy_tower_loc:
                dist_to_ally = self._cal_dist(rune_loc, ally_tower_loc)
                dist_to_enemy = self._cal_dist(rune_loc, enemy_tower_loc)
                if dist_to_ally <= dist_to_enemy:
                    ally_runes.append(rune_loc)
                else:
                    enemy_runes.append(rune_loc)
            elif ally_tower_loc:
                # 敌方塔已破，默认归属己方
                ally_runes.append(rune_loc)
            elif enemy_tower_loc:
                # 己方塔已破，默认归属敌方
                enemy_runes.append(rune_loc)
            # 双方塔都不存在时不处理

        # 计算4维特征
        ally_rune_exists = 1 if ally_runes else 0
        enemy_rune_exists = 1 if enemy_runes else 0

        if self.main_hero_info and ally_runes:
            ally_dist = min(self._cal_dist(self.main_hero_info["location"], r) for r in ally_runes)
            ally_dist = min(ally_dist / 12000.0, 1.0)
        else:
            ally_dist = 1.0

        if self.main_hero_info and enemy_runes:
            enemy_dist = min(self._cal_dist(self.main_hero_info["location"], r) for r in enemy_runes)
            enemy_dist = min(enemy_dist / 12000.0, 1.0)
        else:
            enemy_dist = 1.0

        # 将计算结果存储为实例属性，供 feature_func_map 中的函数使用
        self._rune_feature_values = {
            "ally_rune_exists": ally_rune_exists,
            "enemy_rune_exists": enemy_rune_exists,
            "dist_to_ally_rune": ally_dist,
            "dist_to_enemy_rune": enemy_dist,
        }

        vector_feature = []
        for feature_name, feature_func in self.feature_func_map.items():
            value = []
            feature_func(value)
            if feature_name not in self.map_feature_to_norm:
                assert False
            for k in value:
                norm_func, *params = self.map_feature_to_norm[feature_name]
                normalized_value = norm_func(k, *params)
                if isinstance(normalized_value, list):
                    vector_feature.extend(normalized_value)
                else:
                    vector_feature.append(normalized_value)
        return vector_feature

    def _set_main_hero_info(self, frame_state):
        self.main_hero_info = None
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_hero_info = hero
                break

    def _get_ally_tower_loc(self, frame_state):
        for organ in frame_state.get("npc_states", []):
            if organ["camp"] == self.main_camp and organ.get("sub_type") == 21:
                if organ.get("hp", 0) > 0:
                    return organ.get("location")
        return None

    def _get_enemy_tower_loc(self, frame_state):
        for organ in frame_state.get("npc_states", []):
            if organ["camp"] != self.main_camp and organ.get("sub_type") == 21:
                if organ.get("hp", 0) > 0:
                    return organ.get("location")
        return None

    def _cal_dist(self, pos1, pos2):
        return math.sqrt(
            (pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2
        )

    # ============ 神符特征函数 ============

    def get_ally_rune_exists(self, vector_feature):
        vector_feature.append(self._rune_feature_values["ally_rune_exists"])

    def get_enemy_rune_exists(self, vector_feature):
        vector_feature.append(self._rune_feature_values["enemy_rune_exists"])

    def get_dist_to_ally_rune(self, vector_feature):
        vector_feature.append(self._rune_feature_values["dist_to_ally_rune"])

    def get_dist_to_enemy_rune(self, vector_feature):
        vector_feature.append(self._rune_feature_values["dist_to_enemy_rune"])
