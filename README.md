# 基于深度强化学习的智能推荐与决策优化

> 课程作业：以 DQN 为核心算法，覆盖马尔可夫决策过程（MDP）建模、深度 Q 网络训练与推荐场景落地。

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

| MDP 要素 | CartPole（验证环境） | MovieLens 推荐环境 |
|---|---|---|
| 状态 S | 车位置、车速、杆角度、角速度（4 维） | [用户偏好隐向量(20), 历史 embedding 均值(20), 候选电影特征 20×(20+1)]（460 维） |
| 动作 A | 左推 / 右推（2 维） | 从候选集（每步随机召回 20 部）推荐 1 部 |
| 奖励 R | 每存活一步 +1 | 模拟评分 rating∈[1,5]，reward = rating − 3（≥4 分视为喜欢） |
| 转移 P | 物理动力学（环境内置） | 推荐进入用户历史，状态更新（偏好漂移） |
| γ | 0.99 | 0.6 |

## 用到的知识点

**MDP / 值函数**
- 五元组 (S, A, P, R, γ)、回合式交互、折扣累计回报 G_t = Σ γᵏ r_{t+k}
- 状态/动作值函数 V(s)、Q(s,a)，Bellman 最优方程
- Q-learning 的 TD 目标：y = r + γ · max_{a′} Q(s′, a′; θ⁻)

**DQN 三大稳定化技巧**（报告中逐条说明动机）
1. 经验回放：打破相邻样本的时间相关性，提高样本利用率；
2. 目标网络：TD 目标用冻结的 θ⁻ 计算，每 C 步硬同步，防止"追着自己的尾巴跑"；
3. ε-greedy：训练初期多探索、后期多利用，ε 随步数线性衰减。

**加分实现**
- Double DQN：在线网络选动作、目标网络估值，缓解 max 操作带来的 Q 值高估（默认开启）；
- 损失函数用 SmoothL1（对离群 TD 误差更稳健），梯度裁剪 10.0。

**推荐场景特有**
- 离线数据集 → 交互式环境改造：评分模型 r̂ = μ + b_u + b_i + uᵀv + 噪声（截断 SVD 隐因子 + 偏差项）；
- 动作空间 = 候选集召回（对应工业界"召回→排序"两段式架构），候选特征必须进状态；
- Q 网络定制为**逐候选打分结构**：同一打分 MLP 对每个候选计算
  score(用户特征, 候选特征)，在所有候选间共享参数——普通 MLP 需要为
  "输出 a 对应候选块 a" 从零学位置路由，弱信号任务下学不动；
- 状态特征标准化：隐向量量级 ~20 是物品偏差 ~1 的 20 倍，统一尺度后
  打分函数更易学；
- 折扣因子取 0.6：推荐会话远期奖励不确定，且大 γ 下 max 自举会把
  近并列候选的 Q 值高估反复放大（诊断脚本实测 Q 极差膨胀到 34）；
- 评估指标：平均累计奖励、命中率（评分 ≥ 4 的推荐占比）。

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

## 快速开始

```bash
# 1. 安装依赖（venv 已建好；torch 用 CPU 版 + 国内镜像）
.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas matplotlib gymnasium
.venv\Scripts\python -m pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu

# 2. 下载数据集（约 4.7 MB）
.venv\Scripts\python scripts\download_data.py

# 3. CartPole 验证 DQN 实现（约 2~5 分钟，CPU）
.venv\Scripts\python train_cartpole.py

# 4. 推荐环境主实验（约 10 分钟，CPU）
.venv\Scripts\python train_recommend.py

# 5. 策略对比评估
.venv\Scripts\python eval.py
```

## 结果

### CartPole（验证 DQN 实现）
- 贪婪评估平均存活 **500.0 / 500 步**（满分），DQN 实现正确
- 曲线：`report/cartpole_reward.png`（训练后期有波动，属 DQN 硬同步目标的经典现象，可作讨论点）

### MovieLens 推荐环境（50 轮评估，每轮 30 次推荐）

| 策略 | 平均累计奖励 | 命中率（评分 ≥ 4） |
|---|---|---|
| DQN | **25.16** | **39.0%** |
| 随机策略 | 12.84 | 13.5% |
| 流行度策略 | 26.11 | 41.6% |

- DQN 比随机策略提升 **+96%**，与流行度基线基本持平（差异 0.95 处于评估噪声范围内，
  流行度利用了全局统计而 DQN 完全从交互奖励中学出候选打分）
- 曲线：`report/movie_reward.png`（训练奖励 + 每 50 轮贪婪评估）
- 对比图：`report/eval_compare.png`
