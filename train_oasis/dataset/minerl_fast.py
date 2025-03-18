import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
from torchvision.io import read_video
from train_oasis.utils import parse_VPT_action
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class MinerlFastDataset(torch.utils.data.Dataset):
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
                indent=4,
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        # self.data_paths = self.get_data_paths(self.split)
        self.data_paths = [Path(x["file"]) for x in self.metadata[self.split]]
        lengths = [x["length"] for x in self.metadata[self.split]]
        lengths = np.array(lengths)
        # self.clips_per_video = np.clip(np.array(lengths) - self.n_frames + 1, a_min=1, a_max=None).astype(
        #     np.int32
        # )

        # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
        random.seed(0)

        self.sec_cum_lengths = []
        self.idx_remaps = [] #list(range(self.__len__()))
        self.total_len = []

        for i in range(self.n_frames):
            sl = (lengths - i) // self.n_frames # clips number for each video
            self.total_len.append(sl.sum())
            idx_remap = list(range(len(sl)))
            random.shuffle(idx_remap)
            self.idx_remaps.append(idx_remap)
            sl = [sl[idx] for idx in idx_remap]
            sl = np.array(sl)
            sl = np.cumsum(sl)
            self.sec_cum_lengths.append(sl)

        self.total_len = np.array(self.total_len)
        self.total_cum = np.cumsum(self.total_len)

    def __len__(self):
        return self.total_len.sum()

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = list(data_dir.glob("*.jsonl"))
        return paths

    def get_data_lengths(self, split):
        paths = self.get_data_paths(split)
        total_files = len(paths)

        def process_file(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)  # Count lines incrementally
                return str(path), line_count + 1  # Add 1 to mimic original logic
            except Exception as e:
                print(f"Skipping file {path} due to error: {e}")
                return None

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Wrap the executor.map with tqdm for progress bar
            results = list(tqdm(executor.map(process_file, paths), total=total_files, desc="Processing files"))

        # Collect valid lengths
        lengths = [ {
            "file": result[0],
            "length": result[1]
        } for result in results if result is not None]
        return lengths

    def split_idx(self, idx):
        sec_idx = np.argmax(self.total_cum > idx)
        idx_in_sec = idx - np.pad(self.total_cum, (1, 0))[sec_idx]
        cum_length = self.sec_cum_lengths[sec_idx]
        video_idx = np.argmax(cum_length > idx_in_sec)
        frame_idx = idx_in_sec - np.pad(cum_length, (1, 0))[video_idx]
        frame_idx = sec_idx + self.n_frames * frame_idx
        video_idx = self.idx_remaps[sec_idx][video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            return self.getitem(random.randint(0, self.__len__() - 1))

    def getitem(self, idx):
        file_idx, frame_idx = self.split_idx(idx)
        action_path = self.data_paths[file_idx]
        assert action_path.exists(), f"File {action_path} does not exist"
        video_path = action_path.with_suffix(".mp4")
        start = frame_idx / 20
        end = (frame_idx + self.n_frames - 1) / 20
        video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        video = video.contiguous().numpy()
        assert len(video) == self.n_frames, f"len(video)={len(video)} != self.n_frames={self.n_frames}, file_idx={file_idx}, frame_idx={frame_idx}"
        if self.external_cond_dim > 0:
            with open(action_path, "r") as f:
                lines = f.readlines()
            assert frame_idx <= len(lines), f"frame_idx {frame_idx} need to be less than {len(lines)}"
            if frame_idx == 0:
                # First frame will be set to zero
                actions = [parse_VPT_action(lines[0])] + [parse_VPT_action(line) for line in lines[frame_idx : frame_idx + self.n_frames - 1]]
                actions = np.array(actions)
            else:
                actions = [parse_VPT_action(line) for line in lines[frame_idx - 1 : frame_idx + self.n_frames - 1]]
                actions = np.array(actions)
            assert actions.shape == (self.n_frames, self.external_cond_dim), f"actions.shape={actions.shape} != (self.n_frames - 1, self.external_cond_dim), file_idx={file_idx}, frame_idx={frame_idx}"

        nonterminal = np.ones(self.n_frames)
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