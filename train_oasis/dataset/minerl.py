import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
from torchvision.io import read_video
from train_oasis.utils import parse_VPT_action

class MinerlDataset(torch.utils.data.Dataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.h = cfg.h
        self.w = cfg.w
        self.external_cond_dim = cfg.external_cond_dim
        self.n_frames = (
            cfg.n_frames * cfg.frame_skip
            if split == "training"
            else cfg.n_frames * cfg.frame_skip * cfg.validation_multiplier
        )
        self.frame_skip = cfg.frame_skip
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.split_dir = self.save_dir / f"{split}"

        self.metadata_path = self.save_dir / "metadata.json"

        if not self.metadata_path.exists():
            # Build dataset
            print(f"Creating dataset in {self.save_dir}...")
            json.dump(
                {
                    "training": self.get_data_lengths("training"),
                    "validation": self.get_data_lengths("validation"),
                },
                open(self.metadata_path, "w"),
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        self.data_paths = self.get_data_paths(self.split)
        self.clips_per_video = np.clip(np.array(self.metadata[split]) - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)
        # self.transform = transforms.Resize((self.resolution, self.resolution), antialias=True)

        # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
        random.seed(0)
        self.idx_remap = list(range(self.__len__()))
        random.shuffle(self.idx_remap)

    def __len__(self):
        return self.clips_per_video.sum()

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = list(data_dir.glob("*.jsonl"))
        return paths

    def get_data_lengths(self, split):
        paths = self.get_data_paths(split)
        lengths = []
        for path in paths:
            with open(path, "r") as f:
                lines = f.readlines()
                lengths.append(len(lines)+1)
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        action_path = self.data_paths[file_idx]
        video_path = action_path.with_suffix(".mp4")
        start = frame_idx / 20
        end = (frame_idx + self.n_frames - 1) / 20
        video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        video = video.contiguous().numpy()
        if self.external_cond_dim > 0:
            with open(action_path, "r") as f:
                lines = f.readlines()
            if frame_idx == 0:
                # First frame will be set to zero
                actions = [parse_VPT_action(lines[0])] + [parse_VPT_action(line) for line in lines[frame_idx : frame_idx + self.n_frames - 1]]
                actions = np.array(actions)
            else:
                actions = [parse_VPT_action(line) for line in lines[frame_idx - 1 : frame_idx + self.n_frames - 1]]
                actions = np.array(actions)

        # video = video[frame_idx : frame_idx + self.n_frames]  # (t, h, w, 3)
        pad_len = self.n_frames - len(video)

        nonterminal = np.ones(self.n_frames)
        # if len(video) < self.n_frames:
        #     video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
        #     nonterminal[-pad_len:] = 0
        # if self.external_cond_dim > 0 and len(actions) < self.n_frames:
        #     pad_len = self.n_frames - len(actions)
        #     actions = np.pad(actions, ((0, pad_len),))
        assert len(video) == self.n_frames, f"len(video)={len(video)} != self.n_frames={self.n_frames}, file_idx={file_idx}, frame_idx={frame_idx}"

        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()

        if self.external_cond_dim > 0:
            return (
                video[:: self.frame_skip],
                torch.tensor(actions[:: self.frame_skip], dtype=torch.float32),
                nonterminal[:: self.frame_skip],
            )
        else:
            return (
                video[:: self.frame_skip],
                nonterminal[:: self.frame_skip],
            )