"""DQN 在 CartPole 上的验证训练（对应"决策优化"）。

CartPole 是教科书级 MDP 环境：状态 = (车位置, 车速, 杆角度, 角速度)，
动作 = 左推/右推，奖励 = 每存活一步 +1（满分 500）。
先在它上面验证 DQN 实现正确，再迁移到推荐环境。
每 eval_freq 轮做一次贪婪评估（关闭探索），观察泛化性能。
"""

# 模型评估
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer
from utils.plotting import plot_reward_curve


def evaluate_greedy(env, agent, n_episodes=10):
    """关闭探索的评估：贪婪策略的平均存活步数。"""
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
    return float(np.mean(totals))


def main(args):
    env = gym.make("CartPole-v1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("训练设备:", device)

    agent = DQN(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=args.lr, gamma=0.99,
        epsilon_decay_tau=args.epsilon_decay_tau,
        target_update_freq=args.target_update_freq,
        device=device,
    )
    buffer = ReplayBuffer(args.replay_capacity)

    rewards, eval_episodes, eval_rewards = [], [], []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            agent.steps_done += 1
            agent.maybe_sync_target()
            if len(buffer) >= args.batch_size:
                agent.update(buffer.sample(args.batch_size, agent.device))
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        if ep % args.eval_freq == 0:
            eval_reward = evaluate_greedy(env, agent)
            eval_episodes.append(ep)
            eval_rewards.append(eval_reward)
            avg = np.mean(rewards[-10:])
            print(f"[{ep:4d}/{args.episodes}] 近10轮平均奖励 {avg:6.1f} | "
                  f"贪婪评估 {eval_reward:6.1f} | ε={agent.epsilon:.3f}")
            if avg >= args.early_stop_reward:
                print(f"近10轮平均已达 {args.early_stop_reward}，提前结束训练")
                break

    env.close()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    agent.save(args.save_path)
    plot_reward_curve(rewards, eval_episodes, eval_rewards, path=args.plot_path,
                      title="CartPole：DQN 训练奖励曲线",
                      eval_label="贪婪评估平均存活")
    print(f"完成。最终近10轮平均: {np.mean(rewards[-10:]):.1f} / 500，"
          f"模型保存到 {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole 验证 DQN")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay_capacity", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilon_decay_tau", type=float, default=3000,
                        help="ε 指数衰减时间常数（步）")
    parser.add_argument("--target_update_freq", type=int, default=200)
    parser.add_argument("--early_stop_reward", type=float, default=475)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_path", default="models/cartpole_dqn.pt")
    parser.add_argument("--plot_path", default="report/cartpole_reward.png")
    main(parser.parse_args())
