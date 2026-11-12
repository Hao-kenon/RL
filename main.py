
import subprocess
import sys
import os

PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, desc):
    print(f"\n{'='*50}")
    print(f"▶ {desc}")
    print(f"{'='*50}")
    result = subprocess.run([PYTHON, script], cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f" 失败：{desc}")
        sys.exit(1)
    print(f"完成：{desc}\n")


def main():
    # 1. 确保数据存在
    if not os.path.exists("data/ml-100k/u.data"):
        run("scripts/download_data.py", "下载 MovieLens-100K 数据集")

    # 2. 训练模型
    run("train_cartpole.py",  "CartPole DQN 训练")
    run("train_recommend.py", "推荐环境 DQN 训练")

    # 3. 评估对比
    run("eval.py", "策略对比评估")

    # 4. 启动可视化界面
    print(f"\n{'='*50}")
    print("▶ 启动 Gradio 可视化界面")
    print("  浏览器将自动打开 http://localhost:7860")
    print(f"{'='*50}\n")
    subprocess.run([PYTHON, "app.py"], cwd=PROJECT_DIR)


if __name__ == "__main__":
    main()