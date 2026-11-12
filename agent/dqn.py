"""DQN 智能体：Q 网络 + 目标网络 + ε-greedy 探索。

核心：TD 目标 y = r + γ · max_{a'} Q(s', a'; θ⁻)（θ⁻ 为目标网络参数）
     损失 L = SmoothL1( Q(s, a; θ) - y )，y 视为常量（不反传梯度）。

DQN 三大稳定化技巧（报告中逐条说明动机）：
1. 经验回放（ReplayBuffer）：打破样本时间相关性；
2. 目标网络：TD 目标用冻结的 θ⁻ 计算，每 C 步硬同步；
3. ε-greedy：探索/利用平衡，ε 随步数指数衰减（与总步数解耦）。

加分实现：Double DQN（默认开启）——用在线网络选动作、目标网络估值，
缓解 max 操作带来的 Q 值高估。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn


# 坐标
class QNetwork(nn.Module):
    """Q(s, a; θ)：输入状态 s，输出每个动作的 Q 值（小型 MLP）。"""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class CandidateQNetwork(nn.Module):
    """逐候选打分的 Q 网络（推荐任务定制结构）。

    状态 = [用户侧特征, 候选0特征块, 候选1特征块, ..., 候选(A-1)特征块]。
    对每个候选，把 [用户侧特征, 该候选特征块] 送进同一个打分 MLP，
    得到该候选的 Q 值。打分函数在所有候选间共享——网络学的是
    "用户 × 电影" 的打分规律，而不是为每个动作位置单独学一套权重，
    候选块在状态里的拼接顺序因此不影响结果。
    """

    def __init__(self, state_dim, action_dim, candidate_feat_dim, hidden=256):
        super().__init__()
        self.candidate_feat_dim = candidate_feat_dim
        self.user_dim = state_dim - action_dim * candidate_feat_dim
        self.scorer = nn.Sequential(
            nn.Linear(self.user_dim + candidate_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, state_dim) = [user (B, user_dim), 各候选块 (B, A*F)]
        batch = x.shape[0]
        user = x[:, :self.user_dim]
        cands = x[:, self.user_dim:].reshape(
            batch, -1, self.candidate_feat_dim)          # (B, A, F)
        a = cands.shape[1]
        user_rep = user.unsqueeze(1).expand(-1, a, -1)    # (B, A, user_dim)
        feats = torch.cat([user_rep, cands], dim=-1)      # (B, A, user_dim+F)
        scores = self.scorer(feats.reshape(batch * a, -1))
        return scores.reshape(batch, a)


class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_tau=3000,
                 target_update_freq=500, double=True, hidden=128,
                 soft_tau=None, candidate_feat_dim=None, device="cpu"):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.double = double
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_tau = epsilon_decay_tau
        self.target_update_freq = target_update_freq
        self.soft_tau = soft_tau

        # candidate_feat_dim 非空 → 逐候选打分结构（推荐任务）；否则普通 MLP
        if candidate_feat_dim is not None:
            net = CandidateQNetwork
            net_kwargs = dict(candidate_feat_dim=candidate_feat_dim)
        else:
            net = QNetwork
            net_kwargs = {}
        self.q_net = net(state_dim, action_dim, hidden=hidden,
                         **net_kwargs).to(self.device)
        self.target_net = net(state_dim, action_dim, hidden=hidden,
                              **net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.steps_done = 0  # 全局步数，用于 ε 衰减与目标网络同步

    @property
    def epsilon(self):
        """ε 随步数指数衰减：ε = ε_end + (ε_start - ε_end) · exp(-steps/τ)。"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
            * math.exp(-self.steps_done / self.epsilon_decay_tau)

    def act(self, state, eval_mode=False):
        """ε-greedy 选动作；eval_mode=True 时关闭探索，纯贪婪。"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self, batch):
        """从经验回放采样的一批数据上做一步梯度下降，返回 loss。"""
        states, actions, rewards, next_states, dones = batch
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            if self.double:
                # Double DQN：在线网络选动作，目标网络估值
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def maybe_sync_target(self):
        """同步目标网络：soft_tau 非空时每步软更新 θ⁻ ← (1-τ)θ⁻ + τθ
        （更平稳，适合弱信号任务）；否则每 C 步硬复制。"""
        if self.soft_tau is not None:
            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(),
                                 self.q_net.parameters()):
                    tp.data.mul_(1.0 - self.soft_tau).add_(p.data, alpha=self.soft_tau)
        elif self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
