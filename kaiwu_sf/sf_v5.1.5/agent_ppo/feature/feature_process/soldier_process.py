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


class SoldierProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_soldier_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.soldier_feature_config)
        # 每个小兵19维: hp_rate(1)+hp_discrete(3)+dist(1)+relative_angle(8)+type(3)+in_tower_range(1)+is_tower_target(1)+mask(1)=19
        self.one_unit_feature_num = 19
        self.unit_buff_num = 3
        self.frame_state = None
        self.main_hero_info = None

    def get_soldier_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path)

        self.soldier_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.soldier_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_soldier(self, frame_state):
        self.frame_state = frame_state
        self._set_main_hero_info(frame_state)

        # 己方小兵：按阵营筛选，排序，取前3个
        ally_soldiers = self._filter_soldiers_by_camp(frame_state, self.main_camp)
        ally_soldiers = self._sort_by_distance(ally_soldiers)
        ally_features = self._generate_soldier_features(ally_soldiers, 3)

        # 敌方小兵：按阵营筛选，排序，取前3个
        enemy_camp = 1 if self.main_camp == 2 else 2
        enemy_soldiers = self._filter_soldiers_by_camp(frame_state, enemy_camp)
        enemy_soldiers = self._sort_by_distance(enemy_soldiers)
        enemy_features = self._generate_soldier_features(enemy_soldiers, 3)

        # 特征分组提取：分别返回己方小兵(57维)和敌方小兵(57维)
        return (ally_features, enemy_features)

    def _set_main_hero_info(self, frame_state):
        self.main_hero_info = None
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_hero_info = hero
                break

    def _filter_soldiers(self, frame_state):
        soldiers = []
        for npc in frame_state.get("npc_states", []):
            sub_type = npc.get("sub_type", 0)
            hp = npc.get("hp", 0)
            if sub_type in (11, 12, 13) and hp > 0:
                soldiers.append(npc)
        return soldiers

    def _filter_soldiers_by_camp(self, frame_state, camp):
        """按阵营筛选小兵：sub_type∈{11,12,13} 且 hp>0 且 camp 匹配"""
        soldiers = []
        for npc in frame_state.get("npc_states", []):
            sub_type = npc.get("sub_type", 0)
            hp = npc.get("hp", 0)
            npc_camp = npc.get("camp", 0)
            if sub_type in (11, 12, 13) and hp > 0 and npc_camp == camp:
                soldiers.append(npc)
        return soldiers

    def _sort_by_distance(self, soldiers):
        if self.main_hero_info is None or not soldiers:
            return soldiers
        hero_loc = self.main_hero_info["location"]
        if hero_loc.get("x", 0) == 100000:
            return soldiers

        def _dist(soldier):
            sl = soldier["location"]
            return math.sqrt((hero_loc["x"] - sl["x"]) ** 2 + (hero_loc["z"] - sl["z"]) ** 2)

        soldiers.sort(key=_dist)
        return soldiers

    def _generate_soldier_features(self, soldiers, count):
        vector_feature = []
        for i in range(count):
            if i < len(soldiers):
                soldier = soldiers[i]
                for feature_name, feature_func in self.feature_func_map.items():
                    value = []
                    feature_func(soldier, value, feature_name)
                    if feature_name not in self.map_feature_to_norm:
                        assert False
                    for k in value:
                        norm_func, *params = self.map_feature_to_norm[feature_name]
                        normalized_value = norm_func(k, *params)
                        if isinstance(normalized_value, list):
                            vector_feature.extend(normalized_value)
                        else:
                            vector_feature.append(normalized_value)
            else:
                # 不足3个小兵时，用全零向量填充，对应mask=0
                for _ in range(self.one_unit_feature_num):
                    vector_feature.append(0)
        return vector_feature

    # ============ 小兵特征函数 ============

    def get_soldier_hp_rate(self, soldier, vector_feature, feature_name):
        if soldier.get("hp", 0) <= 0 or soldier.get("max_hp", 0) <= 0:
            vector_feature.append(0)
            return
        vector_feature.append(soldier["hp"] / soldier["max_hp"])

    def get_soldier_hp_discrete(self, soldier, vector_feature, feature_name):
        if soldier.get("hp", 0) <= 0 or soldier.get("max_hp", 0) <= 0:
            vector_feature.append(0)
            return
        hp_rate = soldier["hp"] / soldier["max_hp"]
        if hp_rate < 0.3:
            vector_feature.append(0)
        elif hp_rate < 0.7:
            vector_feature.append(1)
        else:
            vector_feature.append(2)

    def get_soldier_dist(self, soldier, vector_feature, feature_name):
        if self.main_hero_info is None:
            vector_feature.append(1)
            return
        hero_loc = self.main_hero_info["location"]
        soldier_loc = soldier["location"]
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(1)
            return
        dist = math.sqrt(
            (hero_loc["x"] - soldier_loc["x"]) ** 2 + (hero_loc["z"] - soldier_loc["z"]) ** 2
        )
        vector_feature.append(min(dist / 12000.0, 1.0))

    def get_soldier_relative_angle(self, soldier, vector_feature, feature_name):
        if self.main_hero_info is None:
            vector_feature.append(0)
            return
        hero_loc = self.main_hero_info["location"]
        soldier_loc = soldier["location"]
        if hero_loc.get("x", 0) == 100000:
            vector_feature.append(0)
            return
        dx = soldier_loc["x"] - hero_loc["x"]
        dz = soldier_loc["z"] - hero_loc["z"]
        if self.transform_camp2_to_camp1:
            dx = -dx
            dz = -dz
        angle = math.atan2(dz, dx)
        # 8方位桶：[-π, π] → [0, 8)
        bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
        vector_feature.append(bin_idx)

    def get_soldier_type(self, soldier, vector_feature, feature_name):
        sub_type = soldier.get("sub_type", 11)
        if sub_type == 11:
            vector_feature.append(0)
        elif sub_type == 12:
            vector_feature.append(1)
        elif sub_type == 13:
            vector_feature.append(2)
        else:
            vector_feature.append(0)

    def get_soldier_in_tower_range(self, soldier, vector_feature, feature_name):
        if soldier.get("hp", 0) <= 0:
            vector_feature.append(0)
            return
        soldier_loc = soldier["location"]
        if soldier_loc.get("x", 0) == 100000:
            vector_feature.append(0)
            return
        enemy_towers = self._get_enemy_towers()
        for tower in enemy_towers:
            tower_loc = tower.get("location", {})
            tower_range = tower.get("attack_range", 0)
            if tower_range <= 0:
                continue
            dist = self._cal_dist_2d(soldier_loc, tower_loc)
            if dist <= tower_range:
                vector_feature.append(1)
                return
        vector_feature.append(0)

    def get_soldier_is_tower_target(self, soldier, vector_feature, feature_name):
        if soldier.get("hp", 0) <= 0:
            vector_feature.append(0)
            return
        soldier_id = soldier.get("runtime_id", -1)
        if soldier_id < 0:
            vector_feature.append(0)
            return
        enemy_towers = self._get_enemy_towers()
        for tower in enemy_towers:
            if tower.get("attack_target") == soldier_id:
                vector_feature.append(1)
                return
        vector_feature.append(0)

    def get_mask(self, soldier, vector_feature, feature_name):
        # 有效小兵实体置1
        vector_feature.append(1)

    # ============ 辅助方法 ============

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
        dx = pos1["x"] - pos2["x"]
        dz = pos1["z"] - pos2["z"]
        return math.sqrt(dx * dx + dz * dz)
