"""绘图工具：统一配色与中文字体，输出报告用 PNG。

配色说明：分类色按固定顺序使用（蓝/橙/青），采用经过色盲可读性
（CVD）验证的色板，同一实体在所有图中颜色固定不变。
"""
# oUo
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 分类色板（固定顺序，勿打乱：蓝色=DQN 主曲线，橙色=评估曲线/第二系列，青色=第三系列）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Segoe UI", "SimHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.6,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
})


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图已保存:", path)


def plot_reward_curve(rewards, eval_points=None, eval_values=None,
                      window=10, path="report/reward.png",
                      title="DQN 训练奖励曲线", eval_label="贪婪评估平均奖励"):
    """训练累计奖励曲线（含滑动平均），可选叠加定期贪婪评估曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rewards, color=SERIES_COLORS[0], alpha=0.25, lw=1,
            label="每轮累计奖励")
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma, color=SERIES_COLORS[0],
                lw=2, label=f"滑动平均（窗口 {window}）")
    if eval_points:
        ax.plot(eval_points, eval_values, color=SERIES_COLORS[1], lw=2,
                marker="o", ms=5, label=eval_label)
    ax.set_xlabel("训练轮数 episode")
    ax.set_ylabel("累计奖励")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, path)
