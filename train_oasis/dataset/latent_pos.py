import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
from torchvision.io import read_video
import os
from tqdm import tqdm
from fractions import Fraction


class LatentPosDataset(torch.utils.data.Dataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.split = split
        # Validation/test use the same latent index as training so Lightning gets a
        # non-empty val_dataloader (GRPO logs val videos from the val batch).
        # ``limit_batch`` in the experiment caps how many val batches actually run.
        self.n_frames = cfg.n_frames
        self.pre_load = cfg.pre_load
        self.action_type = cfg.action_type

        self.metadata_paths = cfg.metadata
        self.limit_video_lengths = cfg.limit_video_length
        self.actions = []
        self.videos = []
        self.lengths = []
        for metadata_path, limit_video_length in zip(self.metadata_paths, self.limit_video_lengths):
            if not Path(metadata_path).exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            metadata = json.load(open(metadata_path, "r"))
            if limit_video_length is not None:
                metadata = metadata[:limit_video_length]
            data_paths = [Path(x["file"]) for x in metadata]
            lengths = [x["length"] for x in metadata]
            self.lengths.extend(lengths)

            for action_path in data_paths:
                if not action_path.exists():
                    raise FileNotFoundError(f"Action file not found: {action_path}")
                video_path = action_path.with_suffix(".pt")
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                if self.pre_load:
                    actions = np.load(action_path)["actions"]
                    actions = torch.from_numpy(actions).float()
                    self.actions.append(actions)
                    video = torch.load(video_path, map_location="cpu", weights_only=True)
                    self.videos.append(video)
                    if video.shape[0] != actions.shape[0]:
                        raise ValueError(f"Video and action lengths do not match: {video.shape[0]} != {actions.shape[0]}")
                else:
                    self.actions.append(action_path)
                    self.videos.append(video_path)

        self.lengths = np.array(self.lengths)
        self.clips_per_video = np.clip(self.lengths - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)

    def __len__(self):
        if self.split in ("training", "validation", "test"):
            return int(self.clips_per_video.sum())
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
            action_path = self.actions[file_idx]
            video_path = self.videos[file_idx]
            actions = np.load(action_path)["actions"]
            actions = torch.from_numpy(actions).float()
            actions = actions[frame_idx : frame_idx + self.n_frames]
            video = torch.load(video_path, map_location="cpu", weights_only=True)
            video = video[frame_idx : frame_idx + self.n_frames]
        actions[:, 4:] = actions[:, 4:] - actions[0, 4:]
        nonterminal = np.ones(self.n_frames)

        assert actions.shape == (self.n_frames, 8), f"actions.shape={actions.shape} != (self.n_frames - 1, self.external_cond_dim), file_idx={file_idx}, frame_idx={frame_idx}"
        assert video.shape[0] == self.n_frames, f"video.shape[0]={video.shape[0]} != self.n_frames"

        if self.action_type == "action":
            actions = actions[:, :4]
        elif self.action_type == "pos":
            actions = actions[:, 4:]
        elif self.action_type == "both":
            actions = actions
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
        return (
            video,
            actions,
            nonterminal
        )