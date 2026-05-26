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


class CrabProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_crab_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.crab_feature_config)
        self.frame_state = None
        self.main_hero_info = None

    def get_crab_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "crab_feature_config.ini")
        self.config.read(config_path)

        self.crab_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.crab_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_crab(self, frame_state):
        self.frame_state = frame_state
        self._set_main_hero_info(frame_state)

        # 识别河蟹：sub_type ∈ [30, 40)
        crab = self._find_crab(frame_state)

        vector_feature = []
        for feature_name, feature_func in self.feature_func_map.items():
            value = []
            feature_func(crab, value, feature_name)
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

    def _find_crab(self, frame_state):
        for npc in frame_state.get("npc_states", []):
            sub_type = npc.get("sub_type", 0)
            if 30 <= sub_type < 40:
                return npc
        return None

    # ============ 河蟹特征函数 ============

    def get_crab_hp_rate(self, crab, vector_feature, feature_name):
        if crab is None or crab.get("hp", 0) <= 0 or crab.get("max_hp", 0) <= 0:
            vector_feature.append(0)
            return
        vector_feature.append(crab["hp"] / crab["max_hp"])

    def get_crab_dist(self, crab, vector_feature, feature_name):
        if crab is None or self.main_hero_info is None:
            vector_feature.append(1)
            return
        hero_loc = self.main_hero_info["location"]
        crab_loc = crab["location"]
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(1)
            return
        dist = math.sqrt(
            (hero_loc["x"] - crab_loc["x"]) ** 2 + (hero_loc["z"] - crab_loc["z"]) ** 2
        )
        vector_feature.append(min(dist / 12000.0, 1.0))

    def get_crab_relative_angle(self, crab, vector_feature, feature_name):
        if crab is None or self.main_hero_info is None:
            vector_feature.append(0)
            return
        hero_loc = self.main_hero_info["location"]
        crab_loc = crab["location"]
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(0)
            return
        dx = crab_loc["x"] - hero_loc["x"]
        dz = crab_loc["z"] - hero_loc["z"]
        if self.transform_camp2_to_camp1:
            dx = -dx
            dz = -dz
        angle = math.atan2(dz, dx)
        # 8方位桶：[-π, π] → [0, 8)
        bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
        vector_feature.append(bin_idx)

    def get_crab_is_alive(self, crab, vector_feature, feature_name):
        if crab is None or crab.get("hp", 0) <= 0:
            vector_feature.append(0)
        else:
            vector_feature.append(1)
