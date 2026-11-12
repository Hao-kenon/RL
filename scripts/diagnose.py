"""诊断脚本：解剖训练好的 Q 网络在推荐环境里到底学到了什么。

对比三种"选候选"策略每步的期望奖励（无噪声）：
- DQN(Q)：Q 值 argmax（训练出的模型）
- oracle：真实分数 b_i + u·v 的 argmax（理论上界）
- 流行度：b_i 的 argmax

同时输出 Q 值跨候选的极差（判别度）、以及用监督回归训练同一个打分网络
能拿到什么水平（隔离"网络结构不行"与"强化学习训不动"）。
"""
# 泰隆
import numpy as np
import torch

from agent.dqn import DQN, CandidateQNetwork
from env.movie_env import MovieLensEnv


def eval_argmax_policies(env, agent, n_eps=30):
    totals = {"DQN(Q)": [], "oracle": [], "流行度": []}
    q_ranges = []
    for _ in range(n_eps):
        state, _ = env.reset()
        u = env.user_id
        base = env.mu + env.b_u[u] - 3.0
        for _ in range(env.episode_len):
            cands = env.candidates
            true_score = env.b_i[cands] + env.user_vec[u] @ env.item_vec[cands].T
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q = agent.q_net(s).numpy().ravel()
            q_ranges.append(float(q.max() - q.min()))
            a_q = int(q.argmax())
            totals["DQN(Q)"].append(base + true_score[a_q])
            totals["oracle"].append(base + true_score[int(true_score.argmax())])
            totals["流行度"].append(base + true_score[int(env.b_i[cands].argmax())])
            state, _, _, _, _ = env.step(a_q)  # 沿 Q 策略推进，让历史真实演化
    return totals, np.mean(q_ranges)


def supervised_probe(env, steps=8000, lr=1e-3, batch=256, hidden=256):
    """用监督回归训练同一个打分结构，看网络本身能否学出打分函数。"""
    net = CandidateQNetwork(state_dim=env.observation_space.shape[0],
                            action_dim=env.n_candidates,
                            candidate_feat_dim=env.k + 1, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        u = np.random.randint(1, env.n_users + 1)
        cands = np.random.choice(env.n_items, size=env.n_candidates,
                                 replace=False) + 1
        state = env.make_state(u, cands, [])
        target = (env.b_i[cands]
                  + env.user_vec[u] @ env.item_vec[cands].T).astype(np.float32)
        s = torch.as_tensor(state).unsqueeze(0)
        t = torch.as_tensor(target).unsqueeze(0)
        pred = net(s)
        loss = loss_fn(pred, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def main():
    env = MovieLensEnv()
    agent = DQN(state_dim=env.observation_space.shape[0],
                action_dim=env.n_candidates, hidden=256,
                candidate_feat_dim=env.k + 1)
    agent.load("models/movie_dqn.pt")

    totals, q_range = eval_argmax_policies(env, agent)
    print("=== 训练好的 DQN：每步期望奖励（30 轮 × 30 步） ===")
    for name, vals in totals.items():
        print(f"  {name:<6}: {np.mean(vals):.3f}")
    print(f"  Q 值跨候选极差均值: {q_range:.3f}（判别度）")

    net = supervised_probe(env)
    print("=== 监督回归（同一网络结构，8000 步） ===")
    totals_sup, q_range_sup = eval_argmax_policies(env, agent, n_eps=30)
    # 换成监督网络做 argmax
    totals_sup = {"监督回归": []}
    for _ in range(30):
        state, _ = env.reset()
        u = env.user_id
        base = env.mu + env.b_u[u] - 3.0
        for _ in range(env.episode_len):
            cands = env.candidates
            true_score = env.b_i[cands] + env.user_vec[u] @ env.item_vec[cands].T
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q = net(s).numpy().ravel()
            totals_sup["监督回归"].append(base + true_score[int(q.argmax())])
            state, _, _, _, _ = env.step(int(q.argmax()))
    print(f"  监督回归: 每步期望奖励 {np.mean(totals_sup['监督回归']):.3f}")


if __name__ == "__main__":
    main()
