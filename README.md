---
title: "基于深度强化学习的智能推荐与决策优化"
author: "konghao"
date: 2026-11-1
---


# 基于深度强化学习的智能推荐与决策优化

> 以 DQN 为核心算法，马尔可夫决策过程（MDP）建模、深度 Q 网络训练与推荐场景。

项目包含两条主线：

1. **经典控制（验证算法，对应"决策优化"）**：CartPole 平衡杆——标准 MDP 环境，先验证 DQN 实现正确（学会后可持续拿满 500 分）；
2. **智能推荐（核心应用）**：把 MovieLens 100K 离线评分数据改造成交互式 MDP 环境（用户模拟器），训练 DQN 学会"给用户推荐合适的电影"，并与随机策略、流行度策略对比。

## 项目架构

```
MovieLens 100K ──> 用户模拟器(env/movie_env.py) ──> DQN(agent/dqn.py) ──> 评估(eval.py)
CartPole(内置) ──┘     实现 gymnasium 接口         经验回放+目标网络+ε-greedy   训练曲线/对比图(report/)
```

- **环境层** `env/`：用户模拟器，实现 gymnasium 的 `reset/step` 接口，把离线数据变成可交互的 MDP；
- **智能体层** `agent/`：Q 网络、目标网络、经验回放池、ε-greedy 探索；
- **训练/评估** `train_cartpole.py`、`train_recommend.py`、`eval.py`；
- **报告素材** `report/`：训练曲线、策略对比图。

## MDP 定义

| MDP 要素 | CartPole（验证环境） | MovieLens 推荐环境                                              |
|--------|---|-------------------------------------------------------------|
| 状态 S   | 车位置、车速、杆角度、角速度（4 维） | [用户偏好隐向量(20), 历史 embedding 均值(20), 候选电影特征 20×(20+1)]（460 维） |
| 动作 A   | 左推 / 右推（2 维） | 从候选集（每步随机召回 20 部）推荐 1 部                                     |
| 奖励 R   | 每存活一步 +1 | 模拟评分 r作为奖励                                                  |
| 转移 P   | 物理动力学（环境内置） | 推荐进入用户历史，状态更新                                               |
| 折扣率γ   | 0.99 | 0.6                                                         |

## 用到的知识点

**MDP / 值函数**
- 五元组 (S, A, P, R, γ)、回合式交互、折扣累计回报 G = Σ γ^t * r_{t+1}
- 状态/动作值函数 V(s)、Q(s,a)，Bellman 方程,即 V(s) = E[r + γ * V(s')] ,Q(s,a)=E[r + γ * Q(s', a')]
- Q-learning 的 TD 目标：y = r + γ · max_{a′} Q(s′, a′; θ⁻)

**DQN 三大稳定技巧**
1. 经验回放：打破相邻样本的时间相关性，提高样本利用率；
2. 目标网络：TD 目标用冻结的 θ⁻ 计算，每 C 步硬同步，防止"追着自己的尾巴跑"；
3. ε-greedy：训练初期多探索、后期多利用，ε 随步数线性衰减。

**加分实现**
- Double DQN：在线网络选动作、目标网络估值，缓解 max 操作带来的 Q 值高估（默认开启）；
- 损失函数用 SmoothL1（对离群 TD 误差更稳健），梯度裁剪 10.0。


## 目录结构

```
Deep-learn-group-work/
├── agent/               # 智能体：DQN 与经验回放
│   ├── dqn.py
│   └── replay_buffer.py
├── env/                 # 环境：MovieLens 用户模拟器
│   └── movie_env.py
├── utils/               # 绘图工具
│   └── plotting.py
├── scripts/
│   └── download_data.py # 下载 MovieLens 100K
├── train_cartpole.py    # 训练①：CartPole 验证 DQN
├── train_recommend.py   # 训练②：推荐环境主实验
├── eval.py              # 评估：DQN vs 随机 vs 流行度
├── data/                # 数据集（下载后生成，不入库）
├── models/              # 训练好的模型权重
├── report/              # 训练曲线与对比图
└── requirements.txt
```

> *执行根目录下 main.py*即可