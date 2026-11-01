"""评估脚本：加载训练好的模型，对比 DQN / 随机策略 / 流行度策略。

指标（推荐环境）：
- 平均累计奖励：一轮 30 次推荐的总奖励，越大越好；
- 命中率：推荐评分 ≥ 4 的电影占比（即用户"喜欢"的比例）。

CartPole 环境：报告贪婪策略的平均存活步数（满分 500）。
"""
import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agent.dqn import DQN
from env.movie_env import MovieLensEnv
from utils.plotting import SERIES_COLORS, MUTED, save_fig


def eval_cartpole(agent, n_episodes=10):
    env = gym.make("CartPole-v1")
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def run_policy(env, policy, n_episodes):
    """跑 n_episodes 轮，返回 (平均累计奖励, 平均命中率)。"""
    rewards, hits = [], []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total, n_hit = 0.0, 0
        for _ in range(env.episode_len):
            action = policy(state)
            state, reward, _, _, info = env.step(action)
            total += reward
            if info["rating"] >= 4:
                n_hit += 1
        rewards.append(total)
        hits.append(n_hit / env.episode_len)
    return float(np.mean(rewards)), float(np.mean(hits))


def plot_compare(metrics, path):
    """三种策略对比：颜色固定跟随策略实体，不随数值排名变化。"""
    names = list(metrics)
    rewards = [metrics[n][0] for n in names]
    hits = [metrics[n][1] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, values, label in zip(
            axes, [rewards, hits], ["平均累计奖励", "命中率（评分 ≥ 4）"]):
        ax.bar(names, values, color=SERIES_COLORS[:len(names)])
        ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=9, color=MUTED)
        ax.set_ylabel(label)
    fig.suptitle("MovieLens 环境策略对比：DQN vs 随机 vs 流行度")
    fig.tight_layout()
    save_fig(fig, path)


def main(args):
    env = MovieLensEnv(data_dir=args.data_dir, k=args.k,
                       n_candidates=args.n_candidates, seed=args.seed)
    agent = DQN(state_dim=env.observation_space.shape[0],
                action_dim=env.n_candidates, hidden=256,
                candidate_feat_dim=args.k + 1)
    if os.path.exists(args.movie_model):
        agent.load(args.movie_model)
        print("已加载模型:", args.movie_model)
    else:
        print("警告: 未找到", args.movie_model, "，DQN 使用未训练权重（结果应接近随机）")

    metrics = {}
    metrics["DQN"] = run_policy(
        env, lambda s: agent.act(s, eval_mode=True), args.episodes)
    metrics["随机策略"] = run_policy(
        env, lambda s: env.action_space.sample(), args.episodes)
    metrics["流行度策略"] = run_policy(
        env, lambda s: int(np.argmax(env.item_popularity[env.candidates])),
        args.episodes)

    print(f"\n{'策略':<8}{'平均累计奖励':>12}{'命中率(≥4分)':>12}")
    for name, (avg_r, hit) in metrics.items():
        print(f"{name:<8}{avg_r:>12.2f}{hit:>12.2%}")
    plot_compare(metrics, args.plot_path)

    if os.path.exists(args.cartpole_model):
        agent_cp = DQN(state_dim=4, action_dim=2)
        agent_cp.load(args.cartpole_model)
        print(f"\nCartPole 贪婪评估平均存活: "
              f"{eval_cartpole(agent_cp):.1f} / 500 步")
    else:
        print("\n未找到", args.cartpole_model, "，跳过 CartPole 评估")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="策略对比评估")
    parser.add_argument("--data_dir", default="data/ml-100k")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50, help="评估轮数")
    parser.add_argument("--movie_model", default="models/movie_dqn.pt")
    parser.add_argument("--cartpole_model", default="models/cartpole_dqn.pt")
    parser.add_argument("--plot_path", default="report/eval_compare.png")
    main(parser.parse_args())
