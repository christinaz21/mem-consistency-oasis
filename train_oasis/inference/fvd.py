import numpy as np
import torch
from tqdm import tqdm

import os
import math
import torch.nn.functional as F
from typing import Tuple
from scipy.linalg import sqrtm
import json
from torchvision.io import read_video
from fractions import Fraction
# https://github.com/universome/fvd-comparison


def load_i3d_pretrained(device=torch.device('cpu')):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = "/home/tc0786/Project/train-oasis/models/fvd_detector/styleganv.pt"
    print('Loading model from: %s'%filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    return i3d
    

def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(x=torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, i3d, bs, device, only_final=False):
    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    # i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device, bs=bs)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device, bs=bs)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device, bs=bs)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device, bs=bs)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    i3d = load_i3d_pretrained(device=device)
    result = calculate_fvd(videos1, videos2, i3d, 1, device, only_final=False)
    print("[fvd-styleganv]", result["value"])

@torch.no_grad()
def all():
    model_names = ["yarn", "historical_buffer", "rag", "infini_attn", "vanilla_10", "vanilla_20"]
    gt_file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    video_clip_length = 32
    save_path = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/fvd.json"

    with open(gt_file_path, "r") as f:
        gt_paths = json.load(f)["random"]
    total_length = 1200
    device = torch.device("cpu")
    i3d = load_i3d_pretrained(device=device)
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)
    all_fvd = {}
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/"
        assert os.path.exists(save_dir), f"Save directory {save_dir} does not exist."
        gt_feats = []
        fake_feats = []
        for info in tqdm(gt_paths, desc=f"Evaluating {model_name}"):
            gt_video_path = info["video_path"]
            save_relative_path = info["save_relative_path"]
            fake_save_path = os.path.join(save_dir, save_relative_path)
            for start_idx in range(100, total_length, video_clip_length):
                if start_idx + video_clip_length > total_length:
                    break
                end_idx = start_idx + video_clip_length
                start_sec = Fraction(start_idx, 20)
                end_sec = Fraction(end_idx, 20)
                gt_video = read_video(gt_video_path, start_sec, end_sec, pts_unit="sec")[0]
                fake_video = read_video(fake_save_path, start_sec, end_sec, pts_unit="sec")[0]

                # THWC -> CTHW
                gt_video = gt_video.permute(3, 0, 1, 2)
                fake_video = fake_video.permute(3, 0, 1, 2)
                feats = np.empty((0, 400))
                gt_feat, fake_feat = np.vstack([feats, i3d(x=torch.stack([preprocess_single(gt_video), preprocess_single(fake_video)]).to(device), **detector_kwargs).detach().cpu().numpy()])
                gt_feats.append(gt_feat)
                fake_feats.append(fake_feat)
        gt_feats = np.array(gt_feats)
        fake_feats = np.array(fake_feats)
        gt_feats = gt_feats.reshape(-1, 400)
        fake_feats = fake_feats.reshape(-1, 400)
        fvd = frechet_distance(fake_feats, gt_feats)
        all_fvd[model_name] = fvd
    with open(save_path, "w") as f:
        json.dump(all_fvd, f, indent=4)

if __name__ == "__main__":
    all()