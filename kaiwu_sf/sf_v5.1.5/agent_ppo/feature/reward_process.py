#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        # 用于 skill2atk_combo 的状态追踪
        self.m_main_last_skill_frame = -1000
        self.m_enemy_last_skill_frame = -1000
        self.m_main_combo_counted = True
        self.m_enemy_combo_counted = True
        # 用于累积计数器追踪 (kill_cnt, dead_cnt)
        self.m_main_last_kill_cnt = 0
        self.m_enemy_last_kill_cnt = 0
        self.m_main_last_dead_cnt = 0
        self.m_enemy_last_dead_cnt = 0
        # 经验归一化分母（到达15级的累计经验值）
        self.m_exp_norm_factor = 13610.0
        # 金币归一化分母
        self.m_money_norm_factor = 15000.0
        # 在初始化时构建经验等级表，避免每帧重复构建
        self._init_max_exp_of_each_hero()

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def _init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        frame_no = frame_data["frame_no"]

        # 检测新对局开始，重置逐局状态
        if frame_no < self.m_last_frame_no:
            self._reset_episode_state()

        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        self.m_last_frame_no = frame_no
        return self.m_reward_value

    # 重置逐局追踪状态
    def _reset_episode_state(self):
        self.m_main_last_skill_frame = -1000
        self.m_enemy_last_skill_frame = -1000
        self.m_main_combo_counted = True
        self.m_enemy_combo_counted = True
        self.m_main_last_kill_cnt = 0
        self.m_enemy_last_kill_cnt = 0
        self.m_main_last_dead_cnt = 0
        self.m_enemy_last_dead_cnt = 0
        # 重置逐局累加型奖励的计数器 (kill/death/last_hit/crab_kill/skill_hit_hero/skill2atk_combo)
        cumulative_keys = [
            "kill", "death", "last_hit", "crab_kill",
            "skill_hit_hero", "skill2atk_combo"
        ]
        for key in cumulative_keys:
            if key in self.m_main_calc_frame_map:
                self.m_main_calc_frame_map[key].cur_frame_value = 0.0
                self.m_main_calc_frame_map[key].last_frame_value = 0.0
            if key in self.m_enemy_calc_frame_map:
                self.m_enemy_calc_frame_map[key].cur_frame_value = 0.0
                self.m_enemy_calc_frame_map[key].last_frame_value = 0.0
        # 重置 crab_vision 的首次发现标记
        if "crab_vision" in self.m_main_calc_frame_map:
            self.m_main_calc_frame_map["crab_vision"].is_first_arrive_center = True
            self.m_main_calc_frame_map["crab_vision"].cur_frame_value = 0.0
            self.m_main_calc_frame_map["crab_vision"].last_frame_value = 0.0
        if "crab_vision" in self.m_enemy_calc_frame_map:
            self.m_enemy_calc_frame_map["crab_vision"].is_first_arrive_center = True
            self.m_enemy_calc_frame_map["crab_vision"].cur_frame_value = 0.0
            self.m_enemy_calc_frame_map["crab_vision"].last_frame_value = 0.0

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero = None
        enemy_hero = None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # Get both defense towers
        # 获取双方防御塔
        main_tower, enemy_tower = None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == 21:
                    main_tower = organ
            else:
                if organ_subtype == 21:
                    enemy_tower = organ

        frame_no = frame_data["frame_no"]
        frame_action = frame_data.get("frame_action", {})
        dead_actions = frame_action.get("dead_action", []) if frame_action else []

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = (main_hero["hp"] / main_hero["max_hp"]) ** 0.25
            elif reward_name == "money":
                reward_struct.cur_frame_value = main_hero.get("money", 0) / self.m_money_norm_factor
            elif reward_name == "exp":
                if main_hero.get("level", 0) < 15:
                    exp_val = main_hero.get("exp", 0)
                    # 累计经验归一化
                    level = main_hero.get("level", 1)
                    cumulative_exp = self._get_cumulative_exp(level, exp_val)
                    reward_struct.cur_frame_value = cumulative_exp / self.m_exp_norm_factor
                else:
                    reward_struct.cur_frame_value = 1.0
            elif reward_name == "ep_rate":
                ep = main_hero.get("ep", 0)
                max_ep = main_hero.get("max_ep", 1)
                reward_struct.cur_frame_value = ep / max(max_ep, 1)
            elif reward_name == "death":
                dead_cnt = main_hero.get("dead_cnt", 0)
                reward_struct.cur_frame_value = float(dead_cnt)
            elif reward_name == "kill":
                kill_cnt = main_hero.get("kill_cnt", 0)
                reward_struct.cur_frame_value = float(kill_cnt)
            elif reward_name == "last_hit":
                # 使用累加计数器，配合 get_reward 零和差分
                reward_struct.cur_frame_value += float(self._count_last_hit(main_hero, dead_actions))
            elif reward_name == "crab_kill":
                # 使用累加计数器，配合 get_reward 零和差分
                reward_struct.cur_frame_value += float(self._count_crab_kill(main_hero, dead_actions))
            elif reward_name == "crab_vision":
                reward_struct.cur_frame_value = self._check_crab_vision(
                    main_hero["camp"], npc_list, reward_struct
                )
            elif reward_name == "skill_hit_hero":
                # 使用累加计数器，配合 get_reward 零和差分
                reward_struct.cur_frame_value += float(
                    self._count_skill_hit_hero(main_hero, enemy_hero)
                )
            elif reward_name == "skill2atk_combo":
                # 使用累加计数器，配合 get_reward 零和差分
                reward_struct.cur_frame_value += float(
                    self._check_skill2atk_combo(main_hero, enemy_hero, frame_no, camp)
                )

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["hp"] / main_hero["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # 计算累计经验值用于归一化
    def _get_cumulative_exp(self, level, current_exp):
        cumulative = 0
        for lv in range(1, min(level, 15)):
            cumulative += self.m_each_level_max_exp.get(lv, 0)
        cumulative += current_exp
        return cumulative

    # 统计本帧补刀数（击杀小兵）
    # 通过 FrameAction.dead_action 检测己方英雄击杀的士兵
    # actor_type: 0=HERO, 1=MONSTER, 2=ORGAN
    # sub_type对organ: 21=防御塔; 对小兵用排除法识别
    def _count_last_hit(self, main_hero, dead_actions):
        hero_runtime_id = main_hero["runtime_id"]
        count = 0
        for dead_action in dead_actions:
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            actor_type = death.get("actor_type", -1)
            sub_type = death.get("sub_type", -1)
            # 排除英雄死亡(actor_type==0)、野怪死亡(actor_type==1，已有crab_kill)、防御塔死亡(sub_type==21)
            if actor_type != 0 and actor_type != 1 and sub_type != 21:
                if killer.get("runtime_id") == hero_runtime_id:
                    count += 1
        return count

    # 统计本帧河蟹击杀数
    # 通过 FrameAction.dead_action 检测己方英雄击杀的野怪(actor_type==1)
    def _count_crab_kill(self, main_hero, dead_actions):
        hero_runtime_id = main_hero["runtime_id"]
        count = 0
        for dead_action in dead_actions:
            killer = dead_action.get("killer", {})
            death = dead_action.get("death", {})
            if death.get("actor_type") == 1 and killer.get("runtime_id") == hero_runtime_id:
                count += 1
        return count

    # 检测本阵营首次发现河蟹（野怪）
    # 首次发现后 cur_frame_value 保持为 1，配合 get_reward 的零和差分仅在发现帧产生奖励
    def _check_crab_vision(self, camp, npc_list, reward_struct):
        camp_idx = 0 if camp == 1 else 1
        # 如果已经发现过，保持 1.0
        if not reward_struct.is_first_arrive_center:
            return 1.0
        # 检查是否本帧首次发现
        for npc in npc_list:
            # actor_type: 1=MONSTER(野怪/河蟹)
            if npc.get("actor_type") == 1:
                camp_visible = npc.get("camp_visible", [False, False])
                if len(camp_visible) > camp_idx and camp_visible[camp_idx]:
                    reward_struct.is_first_arrive_center = False
                    return 1.0
        return 0.0

    # 统计本帧技能命中敌方英雄次数
    def _count_skill_hit_hero(self, main_hero, enemy_hero):
        if enemy_hero is None:
            return 0
        enemy_runtime_id = enemy_hero["runtime_id"]
        count = 0
        hit_target_info = main_hero.get("hit_target_info", [])
        for hit in hit_target_info:
            if hit.get("hit_target") == enemy_runtime_id:
                count += 1
        return count

    # 检测技能后3秒内首次普攻命中敌方英雄
    def _check_skill2atk_combo(self, main_hero, enemy_hero, frame_no, camp):
        if enemy_hero is None:
            return 0

        hero_runtime_id = main_hero["runtime_id"]
        enemy_runtime_id = enemy_hero["runtime_id"]

        # 根据阵营选择对应的追踪变量
        if camp == self.main_hero_camp or self.main_hero_camp == -1:
            last_skill_frame = self.m_main_last_skill_frame
            combo_counted = self.m_main_combo_counted
        else:
            last_skill_frame = self.m_enemy_last_skill_frame
            combo_counted = self.m_enemy_combo_counted

        # 检测技能释放
        skill_used = False
        skill_state = main_hero.get("skill_state", {})
        slot_states = skill_state.get("slot_states", [])
        for slot in slot_states:
            if slot.get("succUsedInFrame", 0) > 0:
                skill_used = True
                break

        if skill_used:
            if camp == self.main_hero_camp or self.main_hero_camp == -1:
                self.m_main_last_skill_frame = frame_no
                self.m_main_combo_counted = False
            else:
                self.m_enemy_last_skill_frame = frame_no
                self.m_enemy_combo_counted = False
            return 0

        # 检测技能后3秒内的普攻命中
        if not combo_counted and (frame_no - last_skill_frame) <= 90:
            # 检查 real_cmd 中是否有攻击敌方英雄的普攻指令
            real_cmd = main_hero.get("real_cmd", [])
            for cmd in real_cmd:
                cmd_type = cmd.get("command_type", -1)
                if cmd_type == 2:
                    attack_common = cmd.get("attack_common", {})
                    if attack_common.get("actorID") == enemy_runtime_id:
                        if camp == self.main_hero_camp or self.main_hero_camp == -1:
                            self.m_main_combo_counted = True
                        else:
                            self.m_enemy_combo_counted = True
                        return 1

        return 0

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
