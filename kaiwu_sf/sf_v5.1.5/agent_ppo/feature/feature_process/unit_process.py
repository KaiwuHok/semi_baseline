#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Unit feature processing: 22 fixed slots x 27 dims per unit = 594 dims total.
Slot layout (target spec):
  0: 自身英雄 (Hero Self)
  1: 友方英雄 (Ally Hero, zero-padded in 1v1)
  2: 敌方英雄 (Enemy Hero)
  3-5: 友方小兵 (Ally Minions)
  6-13: 敌方小兵 (Enemy Minions)
  8 slots
  14-21: 其它单位 (Monsters, neutral creeps, etc.)
  8 slots

Each unit has 27 dims:
  - location.x / z: coordinates (2)
  - hp_rate: HP ratio (1)
  - camp: camp flag (1, 0=ally/1=enemy)
  - v_x / v_z: velocity from frame diff (2)
  - f_x / f_z: normalized forward direction (2)
  - 19 panel attributes (19)
"""

import math
from agent_ppo.conf.conf import GameConfig

GOLD_PRICE = GameConfig.GOLD_PRICE

# 19 panel attribute names and their economic normalization max values
# max = original_max * gold_price (ep/max_ep use 1.0 since not in GOLD_PRICE)
PANEL_ATTRS = [
    ("phy_atk", 12500),
    ("phy_def", 10000),
    ("mgc_atk", 12500),
    ("mgc_def", 10000),
    ("mov_spd", 2400),
    ("atk_spd", 6000),
    ("ep", 2000),
    ("max_ep", 2000),
    ("hp_recover", 1000),
    ("ep_recover", 500),
    ("phy_armor_hurt", 5000),
    ("mgc_armor_hurt", 5000),
    ("crit_rate", 2000),
    ("crit_effe", 6000),
    ("phy_vamp", 1500),
    ("mgc_vamp", 1500),
    ("cd_reduce", 600),
    ("ctrl_reduce", 900),
    ("attack_range", 2000),
]

UNIT_DIM = 27  # per-unit feature dims
NUM_SLOTS = 22  # fixed number of slots

SLOT_HERO_SELF = 0
SLOT_ALLY_HERO = 1
SLOT_ENEMY_HERO = 2
SLOT_ALLY_MINION_START = 3
SLOT_ALLY_MINION_END = 6
SLOT_ENEMY_MINION_START = 6
SLOT_ENEMY_MINION_END = 14
SLOT_OTHER_START = 14
SLOT_OTHER_END = 22


class UnitProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.prev_unit_positions = {}  # runtime_id -> (x, z) for velocity calc

    def reset(self):
        self.prev_unit_positions.clear()

    def process_units(self, frame_state):
        """Process all 22 unit slots, returning 594-dim flat feature vector."""
        hero_states = frame_state.get("hero_states", [])
        npc_states = frame_state.get("npc_states", [])
        cakes = frame_state.get("cakes", [])

        # Collect all units
        all_units = []
        main_hero = None

        for hero in hero_states:
            all_units.append(("hero", hero))
            if hero["camp"] == self.main_camp:
                main_hero = hero

        for npc in npc_states:
            all_units.append(("npc", npc))

        for cake in cakes:
            all_units.append(("cake", cake))

        # Sort into slot categories
        slot_self_hero = []
        slot_ally_hero = []
        slot_enemy_hero = []
        slot_ally_minions = []
        slot_enemy_minions = []
        slot_others = []

        main_hero_runtime_id = main_hero["runtime_id"] if main_hero else -1

        for unit_type, unit in all_units:
            runtime_id = unit.get("runtime_id", 0)
            unit_camp = unit.get("camp", 0)
            actor_type = unit.get("actor_type", 0)

            if runtime_id == main_hero_runtime_id:
                slot_self_hero.append(unit)
            elif actor_type == 0 and unit_camp == self.main_camp:
                # ACTOR_TYPE_HERO, ally hero (not self)
                slot_ally_hero.append(unit)
            elif actor_type == 0 and unit_camp != self.main_camp:
                # Enemy hero
                slot_enemy_hero.append(unit)
            elif unit_camp == self.main_camp and actor_type != 0:
                # Ally non-hero (minions, etc.)
                if unit_type == "npc":
                    sub_type = unit.get("sub_type", -1)
                    if sub_type == 1:  # ACTOR_SUB_SOLDIER
                        slot_ally_minions.append(unit)
                    else:
                        slot_others.append(unit)
                else:
                    slot_others.append(unit)
            elif unit_camp != self.main_camp and actor_type != 0:
                # Enemy non-hero
                if unit_type == "npc":
                    sub_type = unit.get("sub_type", -1)
                    if sub_type == 1:  # ACTOR_SUB_SOLDIER
                        slot_enemy_minions.append(unit)
                    else:
                        slot_others.append(unit)
                else:
                    slot_others.append(unit)
            else:
                slot_others.append(unit)

        # Sort by distance from main hero for minion/other slots
        def sort_by_dist(units):
            if not main_hero:
                return units
            hx, hz = main_hero["location"]["x"], main_hero["location"]["z"]

            def dist(u):
                ux, uz = u["location"]["x"], u["location"]["z"]
                return (ux - hx) ** 2 + (uz - hz) ** 2

            return sorted(units, key=dist)

        slot_ally_minions = sort_by_dist(slot_ally_minions)
        slot_enemy_minions = sort_by_dist(slot_enemy_minions)
        slot_others = sort_by_dist(slot_others)

        # Build 22-slot feature vector
        feature = []

        # Slot 0: self hero
        feature.extend(self._encode_unit(slot_self_hero[0] if slot_self_hero else None, main_hero, is_self=True))

        # Slot 1: ally hero (zero-pad in 1v1)
        feature.extend(self._encode_unit(slot_ally_hero[0] if slot_ally_hero else None, main_hero))

        # Slot 2: enemy hero
        feature.extend(self._encode_unit(slot_enemy_hero[0] if slot_enemy_hero else None, main_hero))

        # Slots 3-5: ally minions (up to 3)
        for i in range(3):
            feature.extend(self._encode_unit(slot_ally_minions[i] if i < len(slot_ally_minions) else None, main_hero))

        # Slots 6-13: enemy minions (up to 8)
        for i in range(8):
            feature.extend(self._encode_unit(slot_enemy_minions[i] if i < len(slot_enemy_minions) else None, main_hero))

        # Slots 14-21: others (up to 8)
        for i in range(8):
            feature.extend(self._encode_unit(slot_others[i] if i < len(slot_others) else None, main_hero))

        # Update velocity tracking
        self._update_velocity_tracking(all_units)

        return feature

    def _encode_unit(self, unit, main_hero, is_self=False):
        """Encode a single unit into 27-dim feature vector."""
        vec = [0.0] * UNIT_DIM

        if unit is None:
            return vec

        idx = 0

        # location.x / z (2): normalize to [-1, 1]
        loc_x = unit["location"]["x"]
        loc_z = unit["location"]["z"]
        if self.transform_camp2_to_camp1 and loc_x != 100000:
            loc_x = -loc_x
            loc_z = -loc_z
        vec[idx] = max(-1.0, min(1.0, loc_x / 60000.0))
        idx += 1
        vec[idx] = max(-1.0, min(1.0, loc_z / 60000.0))
        idx += 1

        # hp_rate (1): normalize to [0, 1]
        max_hp = unit.get("max_hp", 1)
        if max_hp > 0:
            vec[idx] = max(0.0, min(1.0, unit.get("hp", 0) / max_hp))
        idx += 1

        # camp (1): 0=ally, 1=enemy
        vec[idx] = 1.0 if unit.get("camp", 0) != self.main_camp else 0.0
        idx += 1

        # v_x / v_z (2): velocity from frame diff, clamped to [-3000, 3000]
        runtime_id = unit.get("runtime_id", 0)
        vx, vz = 0.0, 0.0
        if runtime_id in self.prev_unit_positions:
            px, pz = self.prev_unit_positions[runtime_id]
            vx = loc_x - px
            vz = loc_z - pz
        vec[idx] = max(-1.0, min(1.0, vx / 3000.0))
        idx += 1
        vec[idx] = max(-1.0, min(1.0, vz / 3000.0))
        idx += 1

        # f_x / f_z (2): normalized forward direction, [-1, 1]
        forward = unit.get("forward", {"x": 0, "z": 0})
        fx = forward.get("x", 0)
        fz = forward.get("z", 0)
        f_norm = math.sqrt(fx * fx + fz * fz)
        if f_norm > 0:
            fx /= f_norm
            fz /= f_norm
        vec[idx] = max(-1.0, min(1.0, fx))
        idx += 1
        vec[idx] = max(-1.0, min(1.0, fz))
        idx += 1

        # 19 panel attributes: economic value → min_max normalize to [0, 1]
        for attr_name, attr_max in PANEL_ATTRS:
            val = unit.get(attr_name, 0) * GOLD_PRICE.get(attr_name, 1.0)
            vec[idx] = max(0.0, min(1.0, val / attr_max))
            idx += 1

        return vec

    def _update_velocity_tracking(self, all_units):
        """Store current positions for next frame's velocity calculation."""
        self.prev_unit_positions.clear()
        for unit_type, unit in all_units:
            # ✅ 修复：跳过没有 location 字段 of the empty units to avoid crash
            if "location" not in unit:
                continue
            runtime_id = unit.get("runtime_id", 0)
            loc = unit["location"]
            lx, lz = loc["x"], loc["z"]
            if self.transform_camp2_to_camp1 and lx != 100000:
                lx = -lx
                lz = -lz
            self.prev_unit_positions[runtime_id] = (lx, lz)