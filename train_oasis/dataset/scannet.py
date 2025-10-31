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
from torchvision.io import read_video
from fractions import Fraction

class ScannetDataset(torch.utils.data.Dataset):
    """
    Scannet dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.n_frames = cfg.n_frames
        self.pre_load = cfg.pre_load

        self.metadata_paths = cfg.metadata
        self.limit_video_lengths = cfg.limit_video_length
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
                raise NotImplementedError("Pre-loading not implemented for ScannetDataset yet.")

        self.lengths = np.array(self.lengths)
        self.clips_per_video = np.clip(self.lengths - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)

    def __len__(self):
        return self.clips_per_video.sum()

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

        action_path = self.paths[file_idx]
        video_path = action_path.with_suffix(".mp4")
        start = Fraction(frame_idx, self.cfg.fps)
        end = Fraction((frame_idx + self.n_frames - 1), self.cfg.fps)
        video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        video = video.contiguous().float() / 255.0
        video = video.permute(0, 3, 1, 2).contiguous()
        assert video.shape[0] == self.n_frames, f"video.shape[0]={video.shape[0]} != self.n_frames"

        # load npy data
        actions = np.load(action_path, allow_pickle=True)
        actions = actions[frame_idx: frame_idx + self.n_frames]
        actions = torch.from_numpy(actions).float()
        assert actions.shape[0] == self.n_frames, f"actions.shape[0]={actions.shape[0]} != self.n_frames"
        nonterminal = np.ones(self.n_frames)

        return (
            video,
            actions,
            nonterminal
        )
