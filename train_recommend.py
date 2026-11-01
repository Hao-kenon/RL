"""DQN 在 MovieLens 用户模拟器上的训练（对应"智能推荐"主实验）。

先运行 scripts/download_data.py 准备数据。
每 eval_freq 轮做一次贪婪评估（关闭探索），观察泛化性能。
"""
import argparse
import os

import numpy as np

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer
from env.movie_env import MovieLensEnv
from utils.plotting import plot_reward_curve


def evaluate_greedy(env, agent, n_episodes=10):
    """关闭探索的评估：每轮从新用户开始，累计贪婪策略的奖励。"""
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        for _ in range(env.episode_len):
            action = agent.act(state, eval_mode=True)
            state, reward, _, _, _ = env.step(action)
            total += reward
        totals.append(total)
    return float(np.mean(totals))


def main(args):
    env = MovieLensEnv(data_dir=args.data_dir, k=args.k,
                       n_candidates=args.n_candidates,
                       history_len=args.history_len,
                       episode_len=args.episode_len, seed=args.seed)
    state_dim = env.observation_space.shape[0]
    print(f"环境就绪: {env.n_users} 用户 × {env.n_items} 电影 | "
          f"状态 {state_dim} 维 | 动作 {args.n_candidates} 维")

    agent = DQN(state_dim=state_dim, action_dim=args.n_candidates,
                lr=args.lr, gamma=args.gamma,
                epsilon_decay_tau=args.epsilon_decay_tau,
                target_update_freq=args.target_update_freq,
                hidden=256, soft_tau=args.soft_tau,
                candidate_feat_dim=args.k + 1)
    buffer = ReplayBuffer(args.replay_capacity)

    rewards, eval_episodes, eval_rewards = [], [], []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(env.episode_len):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state,
                        terminated or truncated)
            agent.steps_done += 1
            agent.maybe_sync_target()
            # 每步多次梯度更新：推荐任务奖励信号弱（每对用户-电影平均只
            # 被采样十余次），需要多次过数据才能学出打分函数
            if len(buffer) >= args.batch_size:
                for _ in range(args.updates_per_step):
                    agent.update(buffer.sample(args.batch_size, agent.device))
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        if ep % args.eval_freq == 0:
            eval_reward = evaluate_greedy(env, agent)
            eval_episodes.append(ep)
            eval_rewards.append(eval_reward)
            print(f"[{ep:4d}/{args.episodes}] 训练奖励 {ep_reward:6.1f} | "
                  f"近10轮均值 {np.mean(rewards[-10:]):6.1f} | "
                  f"贪婪评估 {eval_reward:6.1f} | ε={agent.epsilon:.3f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    agent.save(args.save_path)
    plot_reward_curve(rewards, eval_episodes, eval_rewards, path=args.plot_path,
                      title="MovieLens 推荐环境：DQN 训练曲线")
    print("完成，模型保存到", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推荐环境 DQN 主实验")
    parser.add_argument("--data_dir", default="data/ml-100k")
    parser.add_argument("--k", type=int, default=20, help="隐向量维度")
    parser.add_argument("--n_candidates", type=int, default=20, help="每步候选集大小")
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--updates_per_step", type=int, default=2,
                        help="每步环境交互后做的梯度更新次数")
    parser.add_argument("--epsilon_decay_tau", type=float, default=5000,
                        help="ε 指数衰减时间常数（步）")
    parser.add_argument("--soft_tau", type=float, default=0.005,
                        help="目标网络软更新系数")
    parser.add_argument("--target_update_freq", type=int, default=300)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_path", default="models/movie_dqn.pt")
    parser.add_argument("--plot_path", default="report/movie_reward.png")
    main(parser.parse_args())
