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

MAX_BULLET_SPEED = 5000
MAX_BULLET_DIST = 12000


class BulletProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        # camp is int: 1=蓝方, 2=红方, mirror coordinates for red camp
        self.transform_camp2_to_camp1 = camp == 2
        self.get_bullet_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.bullet_feature_config)
        self.one_unit_feature_num = 25
        self.unit_buff_num = 3
        self.frame_state = None
        self.main_hero_info = None
        # 缓存上一帧子弹位置：runtime_id → (x, z)
        self.prev_bullet_positions = {}

    def get_bullet_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "bullet_feature_config.ini")
        self.config.read(config_path)

        self.bullet_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.bullet_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_bullet(self, frame_state):
        self.frame_state = frame_state
        self._set_main_hero_info(frame_state)

        # 筛选敌方阵营子弹
        enemy_bullets = self._filter_enemy_bullets(frame_state)

        if self.main_hero_info is None or not enemy_bullets:
            # 缓存当前帧子弹位置后返回全零特征
            self._cache_bullet_positions(frame_state)
            return [0.0] * (self.unit_buff_num * self.one_unit_feature_num)

        hero_loc = self.main_hero_info["location"]

        # 计算每个敌方子弹的威胁距离
        bullet_threats = []
        for bullet in enemy_bullets:
            bullet_loc = bullet["location"]
            actual_dist = self._cal_dist(hero_loc, bullet_loc)
            bullet_dir = self._get_bullet_direction(bullet)

            if bullet_dir is not None:
                # 子弹→英雄方向向量
                to_hero_x = hero_loc["x"] - bullet_loc["x"]
                to_hero_z = hero_loc["z"] - bullet_loc["z"]
                to_hero_len = math.sqrt(to_hero_x ** 2 + to_hero_z ** 2)
                if to_hero_len > 0:
                    to_hero_dir_x = to_hero_x / to_hero_len
                    to_hero_dir_z = to_hero_z / to_hero_len
                    cos_angle = bullet_dir[0] * to_hero_dir_x + bullet_dir[1] * to_hero_dir_z
                    if cos_angle > 0:
                        threat_dist = actual_dist
                    else:
                        threat_dist = actual_dist + 10000
                else:
                    threat_dist = actual_dist
            else:
                # 第一帧无法计算方向时，使用实际距离作为威胁距离
                threat_dist = actual_dist

            bullet_threats.append((threat_dist, bullet))

        # 按威胁距离从小到大排序，取前3个
        bullet_threats.sort(key=lambda x: x[0])
        selected = [b for _, b in bullet_threats[:self.unit_buff_num]]

        # 缓存当前帧子弹位置供下一帧使用
        self._cache_bullet_positions(frame_state)

        return self._generate_bullet_features(selected, self.unit_buff_num)

    def _set_main_hero_info(self, frame_state):
        self.main_hero_info = None
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_hero_info = hero
                break

    def _filter_enemy_bullets(self, frame_state):
        enemy_bullets = []
        for bullet in frame_state.get("bullets", []):
            if bullet.get("camp") != self.main_camp:
                enemy_bullets.append(bullet)
        return enemy_bullets

    def _get_bullet_direction(self, bullet):
        """获取子弹飞行方向。通过帧间差分计算。"""
        runtime_id = bullet.get("runtime_id")
        cur_loc = bullet.get("location")
        if cur_loc is None:
            return None

        if runtime_id in self.prev_bullet_positions:
            prev_x, prev_z = self.prev_bullet_positions[runtime_id]
            dx = cur_loc["x"] - prev_x
            dz = cur_loc["z"] - prev_z
            length = math.sqrt(dx * dx + dz * dz)
            if length > 1e-6:
                return (dx / length, dz / length)

        return None

    def _get_bullet_speed(self, bullet):
        """通过帧间位移量估算子弹速度。"""
        runtime_id = bullet.get("runtime_id")
        cur_loc = bullet.get("location")
        if cur_loc is None:
            return 0.0

        if runtime_id in self.prev_bullet_positions:
            prev_x, prev_z = self.prev_bullet_positions[runtime_id]
            dx = cur_loc["x"] - prev_x
            dz = cur_loc["z"] - prev_z
            return math.sqrt(dx * dx + dz * dz)

        return 0.0

    def _get_source_type(self, bullet):
        """slot_type → 来源类型：0=普攻, 1=技能1, 2=技能2, 3=技能3"""
        slot_type = bullet.get("slot_type", 0)
        if slot_type == 0:
            return 0
        elif slot_type == 1:
            return 1
        elif slot_type == 2:
            return 2
        elif slot_type == 3:
            return 3
        return 0

    def _get_is_tracking(self, bullet):
        """判断是否追踪型子弹。通过slot_type判断：普攻(slot_type=0)一般为非追踪。"""
        slot_type = bullet.get("slot_type", 0)
        # 普攻通常不是追踪型，技能可能是
        return 0 if slot_type == 0 else 1

    def _compute_collision_risk(self, bullet):
        """计算碰撞风险 = (1 - bullet_dist) × direction_match"""
        if self.main_hero_info is None:
            return 0.0

        hero_loc = self.main_hero_info["location"]
        bullet_loc = bullet["location"]
        dist = self._cal_dist(hero_loc, bullet_loc)
        bullet_dist = min(dist / MAX_BULLET_DIST, 1.0)

        bullet_dir = self._get_bullet_direction(bullet)
        if bullet_dir is None:
            return 0.0

        to_hero_x = hero_loc["x"] - bullet_loc["x"]
        to_hero_z = hero_loc["z"] - bullet_loc["z"]
        to_hero_len = math.sqrt(to_hero_x ** 2 + to_hero_z ** 2)
        if to_hero_len <= 0:
            return 0.0

        cos_angle = bullet_dir[0] * (to_hero_x / to_hero_len) + bullet_dir[1] * (to_hero_z / to_hero_len)
        direction_match = max(0.0, cos_angle)
        return (1.0 - bullet_dist) * direction_match

    def _compute_relative_angle(self, hero_loc, bullet_loc):
        """计算子弹相对英雄的8方位角"""
        dx = bullet_loc["x"] - hero_loc["x"]
        dz = bullet_loc["z"] - hero_loc["z"]
        if self.transform_camp2_to_camp1:
            dx = -dx
            dz = -dz
        angle = math.atan2(dz, dx)
        bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
        return bin_idx

    def _compute_fly_direction_angle(self, bullet):
        """计算子弹飞行方向的8方位角"""
        bullet_dir = self._get_bullet_direction(bullet)
        if bullet_dir is None:
            return 0
        angle = math.atan2(bullet_dir[1], bullet_dir[0])
        bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
        return bin_idx

    def _cache_bullet_positions(self, frame_state):
        self.prev_bullet_positions.clear()
        for bullet in frame_state.get("bullets", []):
            runtime_id = bullet.get("runtime_id")
            loc = bullet.get("location")
            if runtime_id is not None and loc is not None:
                self.prev_bullet_positions[runtime_id] = (loc["x"], loc["z"])

    def _cal_dist(self, pos1, pos2):
        return math.sqrt(
            (pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2
        )

    def _generate_bullet_features(self, bullets, count):
        vector_feature = []
        for i in range(count):
            if i < len(bullets):
                self._current_bullet = bullets[i]
                self._current_bullet_mask = 1
            else:
                self._current_bullet = None
                self._current_bullet_mask = 0

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

    # ============ 子弹特征函数 ============

    def get_bullet_dist(self, vector_feature):
        if self._current_bullet is None or self.main_hero_info is None:
            vector_feature.append(1)
            return
        dist = self._cal_dist(self.main_hero_info["location"], self._current_bullet["location"])
        vector_feature.append(min(dist / MAX_BULLET_DIST, 1.0))

    def get_bullet_relative_angle(self, vector_feature):
        if self._current_bullet is None or self.main_hero_info is None:
            vector_feature.append(0)
            return
        angle_idx = self._compute_relative_angle(
            self.main_hero_info["location"], self._current_bullet["location"]
        )
        vector_feature.append(angle_idx)

    def get_bullet_fly_direction(self, vector_feature):
        if self._current_bullet is None:
            vector_feature.append(0)
            return
        angle_idx = self._compute_fly_direction_angle(self._current_bullet)
        vector_feature.append(angle_idx)

    def get_bullet_source_type(self, vector_feature):
        if self._current_bullet is None:
            vector_feature.append(0)
            return
        vector_feature.append(self._get_source_type(self._current_bullet))

    def get_bullet_is_tracking(self, vector_feature):
        if self._current_bullet is None:
            vector_feature.append(0)
            return
        vector_feature.append(self._get_is_tracking(self._current_bullet))

    def get_bullet_speed_rate(self, vector_feature):
        if self._current_bullet is None:
            vector_feature.append(0)
            return
        speed = self._get_bullet_speed(self._current_bullet)
        vector_feature.append(min(speed / MAX_BULLET_SPEED, 1.0))

    def get_bullet_collision_risk(self, vector_feature):
        if self._current_bullet is None:
            vector_feature.append(0)
            return
        vector_feature.append(self._compute_collision_risk(self._current_bullet))

    def get_bullet_mask(self, vector_feature):
        vector_feature.append(self._current_bullet_mask)
