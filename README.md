# semi_baseline

腾讯开悟（KaiwuDRL）王者荣耀 1v1 复赛官方 baseline 源代码与开发指南。

本仓库基于官方发布包 `code-hok1v1-public-61.1.3-comp-normal-lite.26comp`，提供两套可直接训练的智能体代码（`agent_ppo` / `agent_diy`），以及对应的环境、特征、奖励、训练流程实现，便于在复赛环境中快速进行算法迭代。

## 项目背景

- 比赛环境：王者荣耀 1v1 自走棋对战，由开悟平台提供分布式训练框架 `kaiwudrl`（actor / learner / monitor）。
- 任务目标：实现一个能在 1v1 场景下自我对弈训练并击败基线对手的强化学习智能体。
- 算法基线：基于 PPO（带 GAE 优势估计、PPO clip、value MSE、entropy 正则）+ LSTM 时序建模，多头输出（按钮 / 方向 / 目标 / 技能等 6 个 action head）。
- 关键超参：`GAMMA=0.995`，`LAMDA=0.95`，`CLIP_PARAM=0.2`，`BETA_START=0.025`（entropy 系数），`LSTM_TIME_STEPS=16`，`LSTM_UNIT_SIZE=512`。

## 目录结构

```
semi_baseline/
├── README.md
├── git_operate.md                          # git 操作备忘
└── kaiwu_sf/
    ├── 腾讯开悟强化学习框架/                 # 框架综述与说明文档
    ├── DevGuide/                            # 项目简介、环境与智能体详述、数据协议
    ├── gif/                                 # legal action mask / sub action mask 示意图
    └── code-hok1v1-public-61.1.3-comp-normal-lite.26comp/   # 比赛代码主目录
        ├── kaiwu.json                       # 模型池配置
        ├── train_test.py                    # 本地快速训练自测脚本
        ├── conf/                            # 全局算法 / 应用配置（toml）
        ├── agent_ppo/                       # PPO 基线智能体（推荐起点）
        │   ├── agent.py                     # Agent 实现：动作采样、reward_manager 等
        │   ├── algorithm/algorithm.py       # learn(): forward + loss + backward + monitor 上报
        │   ├── model/model.py               # 网络结构 + compute_loss(value/policy/entropy)
        │   ├── feature/
        │   │   ├── definition.py            # 样本结构、GAE 计算、replay buffer 打包
        │   │   ├── reward_process.py        # 奖励函数（tower_hp、forward 等）
        │   │   └── feature_process/         # 英雄 / 防御塔特征工程与归一化
        │   ├── workflow/train_workflow.py   # 训练主循环：env step、采样、调度
        │   └── conf/                        # 算法超参、监控面板、训练环境配置
        └── agent_diy/                       # 自定义算法模板（结构同 agent_ppo，留给参赛者改）
```

## 关键模块说明

| 模块 | 作用 |
|---|---|
| `agent_ppo/model/model.py` | 定义 Actor-Critic 网络（共享主干 + 6 个 action head + value head），`compute_loss` 计算 PPO clip loss、value loss、entropy loss 并加权求和 |
| `agent_ppo/algorithm/algorithm.py` | learner 端 `learn(samples)`：前向、反传、梯度裁剪、optimizer.step、按 60s 间隔通过 `monitor.put_data` 上报指标 |
| `agent_ppo/feature/definition.py` | 采样侧逻辑：拼帧、`_calc_reward` 用 GAE 反向递推 advantage 与 V_target、按 LSTM_TIME_STEPS 切分样本 |
| `agent_ppo/feature/reward_process.py` | 奖励项定义与权重，决定智能体优化目标 |
| `agent_ppo/workflow/train_workflow.py` | 训练编排：环境交互、self-play、模型保存与切换 |
| `agent_ppo/conf/conf.py` | `Config` / `GameConfig` / `DimConfig`，所有超参的入口 |
| `agent_ppo/conf/monitor_builder.py` | 监控面板配置（reward / total_loss / value_loss / policy_loss / entropy_loss） |

## 快速开始

1. 阅读 `kaiwu_sf/DevGuide/` 下 3 篇开发指南，了解环境、智能体接口、数据协议。
2. 在 `agent_ppo` 上跑通基线，确认监控曲线（reward / loss）正常下降。
3. 自定义改动建议放到 `agent_diy/`，保留 `agent_ppo/` 作为对照参考。
4. 主要可调参数集中在：
   - `agent_ppo/conf/conf.py`：学习率、PPO clip、entropy 系数、GAE 参数
   - `agent_ppo/feature/reward_process.py`：奖励项与权重
   - `agent_ppo/conf/train_env_conf.toml`：评估间隔、对手类型、monitor_side

## 参考资料

- `kaiwu_sf/腾讯开悟强化学习框架/`：框架综述、智能体说明
- `kaiwu_sf/DevGuide/`：项目简介、环境与智能体详述、数据协议
- `git_operate.md`：分支与日常 git 操作备忘
