import numpy as np
import matplotlib.pyplot as plt
import json

metadata_path = "/home/tc0786/Project/train-oasis/data/maze/metadata.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)

for idx, item in enumerate(metadata['validation'][:20]):  # 仅查看前20个文件
    path = item["file"]
    data = np.load(path, allow_pickle=True)
    origin_layout = data["maze_layout"][0]

    # ---------- 在此之后追加 ----------
    # 可视化 agent 在迷宫上的轨迹
    agent_positions = data["agent_pos"]  # 假设形状为 (T, 2)，格式 [row, col]

    plt.figure(figsize=(6, 6))
    # 显示迷宫布局，1 为墙，0 为通道，用灰度反转
    plt.imshow(origin_layout, cmap='gray')
    # 拆分坐标
    xs = agent_positions[:, 0] - 0.5
    ys = agent_positions[:, 1] - 0.5
    # 画出轨迹
    plt.plot(xs, ys, '-o', color='blue', markersize=4, label='trajectory')
    # 标记起点和终点
    plt.scatter(xs[0], ys[0], color='green', s=80, label='start')
    plt.scatter(xs[-1], ys[-1], color='red', s=80, label='end')

    plt.title("Agent Trajectory on Maze")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"data/maze/visualization/agent_trajectory_{idx}.png")
    plt.close()