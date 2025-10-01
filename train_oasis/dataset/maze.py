import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
import os
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

class MazeDataset(torch.utils.data.Dataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.split = split
        if split == "validation":
            return
        self.n_frames = cfg.n_frames
        self.pre_load = cfg.pre_load
        self.action_types = cfg.action_types

        self.metadata_paths = cfg.metadata
        self.limit_video_lengths = cfg.limit_video_length
        self.actions = []
        self.videos = []
        self.paths = []
        self.lengths = []
        for metadata_path, limit_video_length in zip(self.metadata_paths, self.limit_video_lengths):
            if not Path(metadata_path).exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            metadata = json.load(open(metadata_path, "r"))[split]
            if limit_video_length is not None:
                metadata = metadata[:limit_video_length]
            data_paths = [Path(x["file"]) for x in metadata]
            lengths = [x["length"] for x in metadata]
            self.lengths.extend(lengths)

            if not self.pre_load:
                self.paths.extend(data_paths)
            else:
                for data_path in tqdm(data_paths, desc="Pre-loading data"):
                    if not data_path.exists():
                        raise FileNotFoundError(f"Action file not found: {data_path}")
                    data = np.load(data_path, allow_pickle=True)
                    actions = self.get_action(data)
                    actions = torch.from_numpy(actions).float()
                    self.actions.append(actions)
                    video = data["image"]
                    video = torch.from_numpy(video).float() / 255.0
                    video = rearrange(video, "t h w c -> t c h w")
                    self.videos.append(video)
                    if video.shape[0] != actions.shape[0]:
                        raise ValueError(f"Video and action lengths do not match: {video.shape[0]} != {actions.shape[0]}")
                    assert not np.any(data['terminal']), f"Terminal found in pre-loaded data: {data_path}"

        self.lengths = np.array(self.lengths)
        self.clips_per_video = np.clip(self.lengths - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)

    def get_action(self, data, limit_range=None):
        all_actions = []
        for action_type in self.action_types:
            action = data[action_type]
            if limit_range is not None:
                action = action[limit_range[0] : limit_range[1]]
            all_actions.append(action)
        return np.concatenate(all_actions, axis=-1)

    def __len__(self):
        if self.split == "training":
            return self.clips_per_video.sum()
        else:
            return 0

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def getitem(self, idx):
        file_idx, frame_idx = self.split_idx(idx)

        if self.pre_load:
            actions = self.actions[file_idx][frame_idx : frame_idx + self.n_frames]
            video = self.videos[file_idx][frame_idx : frame_idx + self.n_frames]
        else:
            path = self.paths[file_idx]
            data = np.load(path, allow_pickle=True)
            actions = self.get_action(data, limit_range=(frame_idx, frame_idx + self.n_frames))
            actions = torch.from_numpy(actions).float()
            video = data["image"][frame_idx : frame_idx + self.n_frames]
            video = torch.from_numpy(video).float() / 255.0
            video = rearrange(video, "t h w c -> t c h w")
        nonterminal = np.ones(self.n_frames)

        assert actions.shape[0] == self.n_frames
        assert video.shape[0] == self.n_frames, f"video.shape[0]={video.shape[0]} != self.n_frames"

        return (
            video,
            actions,
            nonterminal
        )

def handle_metadata():
    dir_path = "data/maze"
    metadata_path = "data/maze/metadata.json"

    metadata = {}
    for split in ["train", 'eval']:
        split_dir = os.path.join(dir_path, split)
        data_files = [f for f in os.listdir(split_dir)]
        split_name = "training" if split == "train" else "validation"
        metadata[split_name] = [
            {
                "file": os.path.abspath(os.path.join(split_dir, f)),
                "length": 1001,  # All videos are 1001 frames long
            } for f in data_files
        ]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def show_example():
    path = "data/maze/eval/20220920T045048-1000.npz"

    data = np.load(path, allow_pickle=True)
    print(data.files)
    for file in data.files:
        if file == 'image':
            continue
        print(file)
        print(data[file].shape, data[file].dtype)
        print(data[file][0])
        print()

def check_terminal():
    from tqdm import tqdm
    with open("data/maze/metadata.json", "r") as f:
        metadata = json.load(f)

    for d in tqdm(metadata['train']):
        path = d["file"]

        data = np.load(path, allow_pickle=True)
        if np.any(data['terminal']):
            print(f"Terminal found in {path}")

def visualize():
    path = "/home/tc0786/Project/train-oasis/data/maze/eval/20220920T045206-1000.npz"
    data = np.load(path, allow_pickle=True)

    print(data.files)
    print(data["maze_layout"].shape)
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
    plt.savefig("data/maze/agent_trajectory.png")

if __name__ == "__main__":
    handle_metadata()