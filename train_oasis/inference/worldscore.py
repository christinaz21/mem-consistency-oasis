# worldscore.py
import os
import sys
import shutil
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


# Expect a local folder "droid_slam" next to this file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_SCRIPT_DIR, "droid_slam"))
# sys.path.insert(0, "/n/fs/videogen/DROID-SLAM/droid_slam")  # <- your compiled DROID-SLAM checkout

from droid import Droid  # noqa: E402


def _image_stream(image_list: List[str], stride: int, calib: Sequence[float]):
    """Yields (t, image_tensor[1,3,H,W], intrinsics_tensor[4]) for DROID."""
    fx, fy, cx, cy = calib

    image_list = image_list[::stride]
    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        if image is None:
            continue

        h0, w0, _ = image.shape
        # resize so that (h1*w1) approx 512*512, then crop to multiple of 8
        scale = np.sqrt((512 * 512) / (h0 * w0))
        h1 = int(h0 * scale)
        w1 = int(w0 * scale)

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - (h1 % 8), : w1 - (w1 % 8)]
        image = torch.as_tensor(image).permute(2, 0, 1)  # CHW, uint8-ish

        intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def _write_video_frames_to_dir(
    video_thwc_u8: np.ndarray,
    out_dir: str,
    resize_long_side: Optional[int] = None,
) -> List[str]:
    """
    video_thwc_u8: (T,H,W,C=3) uint8 RGB
    Writes PNG frames and returns sorted file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    frame_paths = []
    T = video_thwc_u8.shape[0]

    for t in range(T):
        rgb = video_thwc_u8[t]
        if resize_long_side is not None:
            h, w = rgb.shape[:2]
            long_side = max(h, w)
            if long_side > resize_long_side:
                scale = resize_long_side / float(long_side)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        # cv2 wants BGR
        bgr = rgb[..., ::-1]
        fp = os.path.join(out_dir, f"{t:06d}.png")
        ok = cv2.imwrite(fp, bgr)
        if ok:
            frame_paths.append(fp)

    return frame_paths


class DroidReprojectionScorer:
    """
    Self-contained reprojection-error scorer using DROID-SLAM.

    Returns:
      - mean_reprojection_error (lower is better)
      - reward = -mean_error (higher is better), convenient for BoN argmax
    """

    def __init__(
        self,
        weights_path: str,
        calib: Tuple[float, float, float, float] = (500.0, 500.0, 256.0, 256.0),
        stride: int = 2,
        buffer: int = 512,
        filter_thresh: float = 0.01,
        upsample: bool = True,
        quiet: bool = True,
        resize_long_side: Optional[int] = 256,
        max_frames: int = 200,
    ):
        self.weights_path = weights_path
        self.calib = calib
        self.stride = int(stride)
        self.buffer = int(buffer)
        self.filter_thresh = float(filter_thresh)
        self.upsample = bool(upsample)
        self.quiet = bool(quiet)
        self.resize_long_side = resize_long_side
        self.max_frames = int(max_frames)

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    def mean_reprojection_error_from_frames(self, frame_paths: List[str]) -> Optional[float]:
        if len(frame_paths) == 0:
            return None

        droid_args = type("Args", (), {})()
        droid_args.t0 = 0
        droid_args.stride = self.stride
        droid_args.weights = self.weights_path
        droid_args.buffer = self.buffer
        droid_args.beta = 0.3
        droid_args.filter_thresh = self.filter_thresh
        droid_args.warmup = 8
        droid_args.keyframe_thresh = 4.0
        droid_args.frontend_thresh = 16.0
        droid_args.frontend_window = 25
        droid_args.frontend_radius = 2
        droid_args.frontend_nms = 1
        droid_args.backend_thresh = 22.0
        droid_args.backend_radius = 2
        droid_args.backend_nms = 3
        droid_args.upsample = self.upsample
        droid_args.stereo = False
        droid_args.calib = list(self.calib)

        droid = None
        stream = _image_stream(frame_paths, droid_args.stride, droid_args.calib)

        iterator = tqdm(stream, disable=self.quiet)

        def _run():
            nonlocal droid
            for (t, image, intrinsics) in iterator:
                if droid is None:
                    droid_args.image_size = [image.shape[2], image.shape[3]]
                    droid = Droid(droid_args)
                droid.track(t, image, intrinsics=intrinsics)

            # terminate expects a fresh stream
            stream2 = _image_stream(frame_paths, droid_args.stride, droid_args.calib)
            traj_est, valid_errors = droid.terminate(stream2)
            return valid_errors

        if self.quiet:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                valid_errors = _run()
        else:
            valid_errors = _run()

        if valid_errors is None or len(valid_errors) == 0:
            return None

        mean_err = valid_errors.mean().item()
        if not np.isfinite(mean_err):
            return None
        return float(mean_err)

    def mean_reprojection_error_from_video(self, video_thwc: torch.Tensor) -> Optional[float]:
        """
        video_thwc: (T,H,W,C) float in [0,1] (CPU or GPU)
        """
        v = torch.clamp(video_thwc, 0, 1)
        if v.is_cuda:
            v = v.detach().cpu()
        v = (v * 255).byte().numpy()  # uint8 RGB

        # trim
        if v.shape[0] > self.max_frames:
            v = v[: self.max_frames]

        tmpdir = tempfile.mkdtemp(prefix="droid_eval_")
        try:
            frame_paths = _write_video_frames_to_dir(v, tmpdir, resize_long_side=self.resize_long_side)
            return self.mean_reprojection_error_from_frames(frame_paths)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def reward_from_video(self, video_thwc: torch.Tensor) -> float:
        """
        Returns higher-is-better reward = -mean_reprojection_error.
        If DROID fails, returns a large negative number.
        """
        try:
            err = self.mean_reprojection_error_from_video(video_thwc)
            if err is None:
                return -1e9
            return -err
        except Exception:
            return -1e9