"""DQN 推荐系统可视化交互界面（Gradio）。

三个功能模块：
1. 手动玩一局推荐：用户从候选电影中点选 → 模拟评分 → 看 DQN Q 值变化
2. 参数调节 & 自动运行：拖滑杆调 ε、选 checkpoint，实时跑一局看奖励曲线
3. 策略对比评估：DQN vs 随机 vs 流行度，柱状图对比

启动方式：python app.py
"""
import os
import glob

import gradio as gr
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# 中文字体配置（解决乱码）
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Segoe UI"],
    "axes.unicode_minus": False,
})

from agent.dqn import DQN
from env.movie_env import MovieLensEnv

# ============ 全局初始化 ============
DATA_DIR = "data/ml-100k"
K = 20
N_CANDIDATES = 20
HISTORY_LEN = 10
EPISODE_LEN = 30
HIDDEN = 256
CANDIDATE_FEAT_DIM = K + 1

# 配色
COLOR_DQN = "#2a78d6"
COLOR_RANDOM = "#eb6834"
COLOR_POPULAR = "#1baf7a"
COLOR_BG = "#fcfcfb"


def build_env():
    """构建推荐环境实例。"""
    return MovieLensEnv(data_dir=DATA_DIR, k=K, n_candidates=N_CANDIDATES,
                        history_len=HISTORY_LEN, episode_len=EPISODE_LEN)


def build_agent(model_path=None):
    """构建 DQN 智能体，可选加载权重。"""
    env_tmp = build_env()
    state_dim = env_tmp.observation_space.shape[0]
    agent = DQN(state_dim=state_dim, action_dim=N_CANDIDATES,
                hidden=HIDDEN, candidate_feat_dim=CANDIDATE_FEAT_DIM)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    return agent


def get_model_files():
    """扫描 models/ 目录下所有 .pt 文件。"""
    files = glob.glob("models/*.pt")
    if not files:
        return ["（无可用模型）"]
    return sorted(files)


# ============ Tab 1: 手动玩一局推荐 ============
class ManualSession:
    """管理手动交互的会话状态。"""

    def __init__(self):
        self.env = None
        self.agent = None
        self.state = None
        self.step_count = 0
        self.history_log = []  # (step, item_id, rating, reward, q_values)

    def reset(self, model_path):
        self.env = build_env()
        self.agent = build_agent(model_path)
        self.state, _ = self.env.reset()
        self.step_count = 0
        self.history_log = []
        return self._get_display()

    def pick_movie(self, action_idx):
        """用户选择推荐第 action_idx 个候选电影。"""
        if self.env is None or self.step_count >= EPISODE_LEN:
            return self._get_display()

        # 计算当前所有候选的 Q 值
        with torch.no_grad():
            s = torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)
            q_values = self.agent.q_net(s).squeeze(0).numpy()

        # 环境交互
        next_state, reward, terminated, truncated, info = self.env.step(action_idx)
        self.state = next_state
        self.step_count += 1
        self.history_log.append({
            "step": self.step_count,
            "item_id": info["item_id"],
            "user_id": info["user_id"],
            "rating": info["rating"],
            "reward": reward,
            "q_values": q_values.copy(),
            "chosen_idx": action_idx,
        })
        return self._get_display()

    def auto_pick_best(self):
        """DQN 自动选最优动作。"""
        if self.env is None or self.step_count >= EPISODE_LEN:
            return self._get_display()
        with torch.no_grad():
            s = torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)
            q_values = self.agent.q_net(s).squeeze(0).numpy()
        best_action = int(np.argmax(q_values))
        return self.pick_movie(best_action)

    def _get_display(self):
        """返回当前状态的展示数据。"""
        if self.env is None:
            return "请先点击「开始新的一局」", "", None, ""

        # 候选电影列表 + Q 值
        with torch.no_grad():
            s = torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)
            q_values = self.agent.q_net(s).squeeze(0).numpy()

        candidates = self.env.candidates
        lines = []
        for i, (cid, q) in enumerate(zip(candidates, q_values)):
            marker = " ⭐" if i == int(np.argmax(q_values)) else ""
            lines.append(f"  [{i:2d}] 电影 #{cid:<5d}  Q值={q:+.3f}{marker}")
        candidate_text = "\n".join(lines)

        # Q 值柱状图
        fig, ax = plt.subplots(figsize=(10, 3.5))
        colors = [COLOR_DQN if i == int(np.argmax(q_values)) else "#aab4c2"
                  for i in range(len(q_values))]
        ax.bar(range(len(q_values)), q_values, color=colors, edgecolor="none")
        ax.set_xlabel("候选电影编号")
        ax.set_ylabel("Q 值")
        ax.set_title(f"第 {self.step_count + 1} 步 — DQN 对各候选电影的打分"
                     f"（用户 #{self.env.user_id}）")
        ax.set_facecolor(COLOR_BG)
        fig.tight_layout()

        # 历史记录
        if self.history_log:
            hist_lines = [f"{'步':>3} | {'电影ID':>6} | {'评分':>5} | {'奖励':>6} | 说明"]
            hist_lines.append("-" * 55)
            for h in self.history_log[-10:]:
                emoji = "😊" if h["rating"] >= 4 else ("😐" if h["rating"] >= 3 else "😞")
                hist_lines.append(
                    f"{h['step']:>3} | #{h['item_id']:<5d} | {h['rating']:.2f} | "
                    f"{h['reward']:+.2f} | {emoji}")
            history_text = "\n".join(hist_lines)
        else:
            history_text = "（尚无推荐记录）"

        status = (f"🎬 用户 #{self.env.user_id} | "
                  f"步数 {self.step_count}/{EPISODE_LEN} | "
                  f"累计奖励 {sum(h['reward'] for h in self.history_log):+.2f}")
        if self.step_count >= EPISODE_LEN:
            status += "\n✅ 本轮已结束！点击「开始新的一局」重新体验。"

        return status, candidate_text, fig, history_text


# 全局会话（Gradio 单用户模式）
session = ManualSession()


def tab1_start(model_path):
    result = session.reset(model_path)
    return result


def tab1_pick(action_idx):
    idx = int(action_idx)
    if idx < 0 or idx >= N_CANDIDATES:
        return session._get_display()
    return session.pick_movie(idx)


def tab1_auto():
    return session.auto_pick_best()


# ============ Tab 2: 参数调节 & 自动运行 ============
def run_episode_with_params(model_path, epsilon_override, gamma, seed):
    """用指定参数自动跑一局，返回奖励曲线图和统计信息。"""
    env = build_env()
    agent = build_agent(model_path)

    # 覆盖 ε（固定值，不衰减）
    rewards_per_step = []
    q_history = []
    state, _ = env.reset(seed=int(seed))

    for t in range(EPISODE_LEN):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = agent.q_net(s).squeeze(0).numpy()
        q_history.append(q_vals.max())

        # ε-greedy
        if np.random.random() < epsilon_override:
            action = np.random.randint(N_CANDIDATES)
        else:
            action = int(np.argmax(q_vals))

        state, reward, _, _, info = env.step(action)
        rewards_per_step.append(reward)

    cumulative = np.cumsum(rewards_per_step)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(COLOR_BG)

    # 每步奖励
    ax = axes[0]
    colors = [COLOR_DQN if r > 0 else COLOR_RANDOM for r in rewards_per_step]
    ax.bar(range(len(rewards_per_step)), rewards_per_step, color=colors)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xlabel("推荐步数")
    ax.set_ylabel("即时奖励 (rating - 3)")
    ax.set_title("每步奖励")
    ax.set_facecolor(COLOR_BG)

    # 累计奖励
    ax = axes[1]
    ax.plot(cumulative, color=COLOR_DQN, lw=2)
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15, color=COLOR_DQN)
    ax.set_xlabel("推荐步数")
    ax.set_ylabel("累计奖励")
    ax.set_title("累计奖励曲线")
    ax.set_facecolor(COLOR_BG)

    # max Q 值变化
    ax = axes[2]
    ax.plot(q_history, color=COLOR_POPULAR, lw=2)
    ax.set_xlabel("推荐步数")
    ax.set_ylabel("max Q(s,a)")
    ax.set_title("Q 值变化趋势")
    ax.set_facecolor(COLOR_BG)

    fig.tight_layout()

    hit_rate = sum(1 for r in rewards_per_step if r >= 1.0) / len(rewards_per_step)
    stats = (f"📊 统计结果\n"
             f"  • 累计奖励: {cumulative[-1]:+.2f}\n"
             f"  • 平均即时奖励: {np.mean(rewards_per_step):+.3f}\n"
             f"  • 命中率 (评分≥4): {hit_rate:.1%}\n"
             f"  • 探索率 ε: {epsilon_override:.2f}\n"
             f"  • 折扣因子 γ: {gamma}\n"
             f"  • 随机种子: {int(seed)}")
    return fig, stats


def run_multi_episodes(model_path, epsilon_override, n_episodes, seed):
    """跑多局，返回奖励曲线对比图。"""
    env = build_env()
    agent = build_agent(model_path)
    all_cumulative = []

    for ep in range(int(n_episodes)):
        state, _ = env.reset(seed=int(seed) + ep)
        ep_rewards = []
        for t in range(EPISODE_LEN):
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = agent.q_net(s).squeeze(0).numpy()
            if np.random.random() < epsilon_override:
                action = np.random.randint(N_CANDIDATES)
            else:
                action = int(np.argmax(q_vals))
            state, reward, _, _, _ = env.step(action)
            ep_rewards.append(reward)
        all_cumulative.append(np.cumsum(ep_rewards))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)

    cmap = plt.cm.Blues
    for i, cum in enumerate(all_cumulative):
        alpha = 0.3 + 0.7 * (i / max(len(all_cumulative) - 1, 1))
        ax.plot(cum, color=cmap(alpha + 0.2), lw=1.5, alpha=0.8,
                label=f"Episode {i+1}")

    mean_curve = np.mean(all_cumulative, axis=0)
    ax.plot(mean_curve, color=COLOR_RANDOM, lw=3, label="平均")
    ax.set_xlabel("推荐步数")
    ax.set_ylabel("累计奖励")
    ax.set_title(f"{int(n_episodes)} 局推荐奖励曲线对比（ε={epsilon_override:.2f}）")
    if n_episodes <= 10:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    avg_total = np.mean([c[-1] for c in all_cumulative])
    stats = (f"📊 {int(n_episodes)} 局平均累计奖励: {avg_total:+.2f}\n"
             f"  • 最高: {max(c[-1] for c in all_cumulative):+.2f}\n"
             f"  • 最低: {min(c[-1] for c in all_cumulative):+.2f}\n"
             f"  • 标准差: {np.std([c[-1] for c in all_cumulative]):.2f}")
    return fig, stats


# ============ Tab 3: 策略对比评估 ============
def run_comparison(model_path, n_episodes, seed):
    """对比 DQN / 随机 / 流行度三种策略。"""
    env = build_env()
    agent = build_agent(model_path)
    n_ep = int(n_episodes)

    results = {"DQN": {"rewards": [], "hits": []},
               "随机策略": {"rewards": [], "hits": []},
               "流行度策略": {"rewards": [], "hits": []}}

    for ep in range(n_ep):
        s = int(seed) + ep
        # DQN
        state, _ = env.reset(seed=s)
        total, hits = 0.0, 0
        for _ in range(EPISODE_LEN):
            action = agent.act(state, eval_mode=True)
            state, reward, _, _, info = env.step(action)
            total += reward
            if info["rating"] >= 4:
                hits += 1
        results["DQN"]["rewards"].append(total)
        results["DQN"]["hits"].append(hits / EPISODE_LEN)

        # 随机
        state, _ = env.reset(seed=s)
        total, hits = 0.0, 0
        for _ in range(EPISODE_LEN):
            action = env.action_space.sample()
            state, reward, _, _, info = env.step(action)
            total += reward
            if info["rating"] >= 4:
                hits += 1
        results["随机策略"]["rewards"].append(total)
        results["随机策略"]["hits"].append(hits / EPISODE_LEN)

        # 流行度
        state, _ = env.reset(seed=s)
        total, hits = 0.0, 0
        for _ in range(EPISODE_LEN):
            action = int(np.argmax(env.item_popularity[env.candidates]))
            state, reward, _, _, info = env.step(action)
            total += reward
            if info["rating"] >= 4:
                hits += 1
        results["流行度策略"]["rewards"].append(total)
        results["流行度策略"]["hits"].append(hits / EPISODE_LEN)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    names = list(results.keys())
    colors = [COLOR_DQN, COLOR_RANDOM, COLOR_POPULAR]

    # 平均累计奖励
    ax = axes[0]
    avg_rewards = [np.mean(results[n]["rewards"]) for n in names]
    bars = ax.bar(names, avg_rewards, color=colors, edgecolor="none")
    ax.bar_label(bars, fmt="%.2f", fontsize=10)
    ax.set_ylabel("平均累计奖励")
    ax.set_title("累计奖励对比")
    ax.set_facecolor(COLOR_BG)

    # 命中率
    ax = axes[1]
    avg_hits = [np.mean(results[n]["hits"]) for n in names]
    bars = ax.bar(names, [h * 100 for h in avg_hits], color=colors, edgecolor="none")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=10)
    ax.set_ylabel("命中率 (%)")
    ax.set_title("命中率对比（评分 ≥ 4）")
    ax.set_facecolor(COLOR_BG)

    # 奖励分布箱线图
    ax = axes[2]
    data = [results[n]["rewards"] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("累计奖励")
    ax.set_title("奖励分布")
    ax.set_facecolor(COLOR_BG)

    fig.tight_layout()

    # 文字报告
    report_lines = [f"{'策略':<10} {'平均奖励':>10} {'命中率':>10} {'标准差':>10}"]
    report_lines.append("-" * 45)
    for n in names:
        avg_r = np.mean(results[n]["rewards"])
        avg_h = np.mean(results[n]["hits"])
        std_r = np.std(results[n]["rewards"])
        report_lines.append(f"{n:<10} {avg_r:>+10.2f} {avg_h:>10.1%} {std_r:>10.2f}")

    return fig, "\n".join(report_lines)


# ============ 构建 Gradio 界面 ============
def create_app():
    with gr.Blocks(title="DQN 推荐系统可视化") as app:
        gr.Markdown("""
        # 🎬 DQN 智能推荐系统 — 交互可视化
        基于 MovieLens-100K 数据集的深度强化学习推荐演示平台
        """)

        model_choices = get_model_files()
        default_model = "models/movie_dqn.pt" if "models/movie_dqn.pt" in model_choices \
            else (model_choices[0] if model_choices else None)

        # ---- Tab 1: 手动推荐 ----
        with gr.Tab("🎮 手动玩一局推荐"):
            gr.Markdown("""
            ### 玩法说明
            系统为随机用户抽取 20 部候选电影，DQN 给出每部电影的 Q 值打分。
            你可以**手动选择**推荐哪部电影，观察模拟评分和 Q 值变化；
            也可以让 DQN **自动选最优**。每局 30 步。
            """)
            with gr.Row():
                t1_model = gr.Dropdown(choices=model_choices, value=default_model,
                                       label="选择模型 Checkpoint")
                t1_start_btn = gr.Button("🔄 开始新的一局", variant="primary")
            with gr.Row():
                t1_status = gr.Textbox(label="状态", interactive=False, lines=2)
            with gr.Row():
                t1_qplot = gr.Plot(label="DQN Q 值分布")
            with gr.Row():
                t1_candidates = gr.Textbox(label="候选电影 & Q 值", interactive=False,
                                           lines=12)
                t1_history = gr.Textbox(label="推荐历史（最近 10 步）",
                                        interactive=False, lines=12)
            with gr.Row():
                t1_action = gr.Number(label="输入候选编号 (0-19) 选择推荐", value=0,
                                      precision=0)
                t1_pick_btn = gr.Button("🎯 推荐这部电影", variant="primary")
                t1_auto_btn = gr.Button("🤖 DQN 自动选最优")

            t1_outputs = [t1_status, t1_candidates, t1_qplot, t1_history]
            t1_start_btn.click(fn=tab1_start, inputs=[t1_model], outputs=t1_outputs)
            t1_pick_btn.click(fn=tab1_pick, inputs=[t1_action], outputs=t1_outputs)
            t1_auto_btn.click(fn=tab1_auto, inputs=[], outputs=t1_outputs)

        # ---- Tab 2: 参数调节 ----
        with gr.Tab("⚙️ 参数调节 & 自动运行"):
            gr.Markdown("""
            ### 调参演示
            拖动滑杆调整探索率 ε，选择不同 checkpoint，实时跑推荐看效果。
            - **ε = 0**：纯贪婪（完全利用 DQN 学到的知识）
            - **ε = 1**：纯随机（完全探索）
            - **ε = 0.1**：90% 贪婪 + 10% 随机
            """)
            with gr.Row():
                t2_model = gr.Dropdown(choices=model_choices, value=default_model,
                                       label="模型 Checkpoint")
                t2_epsilon = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                       label="探索率 ε")
                t2_gamma = gr.Slider(0.1, 1.0, value=0.6, step=0.05,
                                     label="折扣因子 γ（展示用）")
                t2_seed = gr.Number(value=42, label="随机种子", precision=0)

            with gr.Row():
                t2_run_btn = gr.Button("▶️ 跑一局（30步）", variant="primary")
                t2_n_ep = gr.Slider(1, 20, value=5, step=1, label="多局对比数量")
                t2_multi_btn = gr.Button("📈 跑多局对比")

            t2_plot = gr.Plot(label="运行结果")
            t2_stats = gr.Textbox(label="统计信息", interactive=False, lines=6)

            t2_run_btn.click(
                fn=run_episode_with_params,
                inputs=[t2_model, t2_epsilon, t2_gamma, t2_seed],
                outputs=[t2_plot, t2_stats])
            t2_multi_btn.click(
                fn=run_multi_episodes,
                inputs=[t2_model, t2_epsilon, t2_n_ep, t2_seed],
                outputs=[t2_plot, t2_stats])

        # ---- Tab 3: 策略对比 ----
        with gr.Tab("📊 策略对比评估"):
            gr.Markdown("""
            ### 三种策略对比
            在相同随机种子下，对比：
            - **DQN**：训练好的深度强化学习策略（贪婪模式）
            - **随机策略**：均匀随机选电影（基线）
            - **流行度策略**：总是选最热门的电影（经典基线）
            """)
            with gr.Row():
                t3_model = gr.Dropdown(choices=model_choices, value=default_model,
                                       label="DQN 模型 Checkpoint")
                t3_episodes = gr.Slider(10, 200, value=50, step=10,
                                        label="评估轮数")
                t3_seed = gr.Number(value=42, label="随机种子", precision=0)
                t3_run_btn = gr.Button("🚀 开始评估", variant="primary")

            t3_plot = gr.Plot(label="策略对比图")
            t3_report = gr.Textbox(label="评估报告", interactive=False, lines=8)

            t3_run_btn.click(
                fn=run_comparison,
                inputs=[t3_model, t3_episodes, t3_seed],
                outputs=[t3_plot, t3_report])

        # ---- 页脚 ----
        gr.Markdown("""
        ---
        **技术栈**: PyTorch + Gymnasium + Gradio + MovieLens-100K |
        **模型**: Double DQN + 逐候选打分网络 | **课程项目演示用**
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False,
               inbrowser=True,
               theme=gr.themes.Soft(),
               css="""
               .gradio-container { max-width: 1200px !important; }
               footer { display: none !important; }
               """)
