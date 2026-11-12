"""MovieLens 用户模拟器：把离线评分数据改造成交互式 MDP 环境。

MDP 要素在推荐场景中的定义：
- 状态 S：拼接 [用户静态偏好向量 u (k 维), 最近 history_len 部电影 embedding 均值 (k 维),
          当前候选集特征 (每部候选电影: 隐向量 k 维 + 物品偏差 1 维)]
  候选集特征必须进状态——候选集每步随机重抽，若状态不含候选信息，
  动作编号就毫无意义，Q 网络无法学会为候选打分。
- 动作 A：从每步随机采样的 n_candidates 部候选电影中推荐 1 部（对应"召回"环节）
- 奖励 R：模拟评分 rating ∈ [1,5]，reward = rating - 3（≥4 分视为"喜欢"，奖励为正）
- 转移 P：被推荐的电影进入用户历史，状态随之更新（用户偏好随交互漂移）
- 折扣 γ：0.6（推荐会话中远期奖励不确定、兴趣漂移快，取较小折扣；
  大 γ + 多近并列候选会经 max 自举放大 Q 值高估，训练不稳）

评分模拟：在训练集上做截断 SVD 分解，r̂ = μ + b_u + b_i + uᵀv + N(0, noise²)，
其中 μ 为全局均值，b_u/b_i 为用户/物品偏差（带收缩），uᵀv 为 k 维隐因子内积。
"""
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class MovieLensEnv(gym.Env):
    def __init__(self, data_dir="data/ml-100k", k=20, n_candidates=20,
                 history_len=10, episode_len=30, noise=0.3, seed=42):
        super().__init__()
        self.k = k
        self.n_candidates = n_candidates
        self.history_len = history_len
        self.episode_len = episode_len
        self.noise = noise
        self.rng = np.random.default_rng(seed) #随机种子

        self._load_data(data_dir)
        self._fit_svd()

        # 状态维度 = 用户偏好 k + 历史均值 k + 候选特征 n_candidates*(k+1)
        state_dim = 2 * k + n_candidates * (k + 1)
        self.action_space = spaces.Discrete(n_candidates)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.user_id = None
        self.history = []
        self.t = 0
        self.candidates = None

    # ---------- 数据加载与分解 ----------
    def _load_data(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t",
                         names=["user_id", "item_id", "rating", "timestamp"])
        self.n_users = int(df["user_id"].max())
        self.n_items = int(df["item_id"].max())
        # 按交互随机划分 80% 训练 / 20% 测试（固定随机种子保证可复现）
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        split = int(len(df) * 0.8)
        self.train_df = df.iloc[:split]
        self.test_df = df.iloc[split:]
        # 物品流行度（训练集交互次数），供流行度基线使用
        counts = self.train_df["item_id"].value_counts()
        self.item_popularity = np.zeros(self.n_items + 1)
        self.item_popularity[counts.index] = counts.values

    def _fit_svd(self):
        """μ + 偏差 + 隐因子：对残差矩阵做截断 SVD，得到用户/物品向量。"""
        train = self.train_df
        self.mu = float(train["rating"].mean())
        shrink = 25.0  # 收缩系数：样本少的用户/物品偏差向 0 收缩

        def biases(group_col, n):
            stats = train.groupby(group_col)["rating"].agg(["sum", "count"])
            b = np.zeros(n + 1)
            b[stats.index] = (stats["sum"] - self.mu * stats["count"]) \
                             / (stats["count"] + shrink)
            return b

        self.b_u = biases("user_id", self.n_users)
        self.b_i = biases("item_id", self.n_items)

        # 残差矩阵（未观测处为 0），截断 SVD 得到 k 维隐向量
        R = np.zeros((self.n_users + 1, self.n_items + 1))
        u_idx = train["user_id"].to_numpy()
        i_idx = train["item_id"].to_numpy()
        R[u_idx, i_idx] = (train["rating"].to_numpy() - self.mu
                          - self.b_u[u_idx] - self.b_i[i_idx])
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        # 奇异值开方对半分给用户/物品两侧，使 u·v 重构出截断残差矩阵
        scale = np.sqrt(S[:self.k])
        self.user_vec = U[:, :self.k] * scale       # (n_users+1, k)
        self.item_vec = Vt[:self.k, :].T * scale    # (n_items+1, k)
        # 特征标准化参数：隐向量量级 ~20 而偏差 ~1，统一尺度后打分函数更易学
        self.u_std = float(self.user_vec.std()) or 1.0
        self.v_std = float(self.item_vec.std()) or 1.0
        self.b_std = float(self.b_i.std()) or 1.0

    # ---------- 环境接口（gymnasium） ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.user_id = int(self.rng.choice(self.n_users) + 1)
        self.history = []
        self.t = 0
        self.candidates = self._sample_candidates()
        return self._get_state(), {}

    def step(self, action):
        item = self.candidates[action]
        u = self.user_id
        # 模拟用户评分：偏差 + 隐因子内积 + 噪声
        pred = (self.mu + self.b_u[u] + self.b_i[item]
                + float(self.user_vec[u] @ self.item_vec[item])
                + self.rng.normal(0.0, self.noise))
        rating = float(np.clip(pred, 1.0, 5.0))
        reward = rating - 3.0  # ≥4 分才产生正奖励

        self.history.append(self.item_vec[item])
        if len(self.history) > self.history_len:
            self.history.pop(0)
        self.t += 1
        terminated = truncated = self.t >= self.episode_len
        self.candidates = self._sample_candidates()
        info = {"user_id": u, "item_id": int(item), "rating": rating}
        return self._get_state(), reward, terminated, truncated, info

    # ---------- 内部工具 ----------
    def _sample_candidates(self):
        """每步随机采样候选集，缩小动作空间（对应推荐系统"召回"环节）。"""
        return self.rng.choice(self.n_items, size=self.n_candidates,
                               replace=False) + 1

    def make_state(self, user_id, candidates, history):
        """构造标准化状态：用户侧 [u/σu, 历史均值/σv] + 候选 [v/σv, b/σb]。"""
        user_feat = self.user_vec[user_id] / self.u_std
        if history:
            hist_feat = np.mean(history, axis=0) / self.v_std
        else:
            hist_feat = np.zeros(self.k)
        # 候选集特征：每部候选电影 [隐向量, 物品偏差]，供 Q 网络为每个动作打分
        cand_feats = np.concatenate(
            [self.item_vec[candidates] / self.v_std,
             self.b_i[candidates, None] / self.b_std], axis=1)  # (A, k+1)
        return np.concatenate(
            [user_feat, hist_feat, cand_feats.ravel()]).astype(np.float32)

    def _get_state(self):
        return self.make_state(self.user_id, self.candidates, self.history)
