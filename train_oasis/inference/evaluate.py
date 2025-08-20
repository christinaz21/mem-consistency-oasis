import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from typing import Optional, Dict
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
    peak_signal_noise_ratio
)
from train_oasis.utils import FrechetVideoDistance
import json
from torchvision.io import read_video
from einops import rearrange
from train_oasis.model.image_discriminator import (
    ImageDiscriminator,
    ImageDiscriminatorResNet
)
from train_oasis.model.vae import VAE_models
from safetensors.torch import load_model
from tqdm import tqdm


@torch.no_grad()
def uiqi_frame(pred, gt):
    uiqi = universal_image_quality_index(pred, gt, reduction="none")
    return uiqi.mean(dim=[1, 2, 3])

@torch.no_grad()
def mse_frame(pred, gt):
    mse = (pred - gt) ** 2
    return mse.mean(dim=[1, 2, 3])

@torch.no_grad()
def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: lpips.LPIPS,
    fid_model: FrechetInceptionDistance,
    image_discriminators: Dict[str, torch.nn.Module],
    vae_model,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    observation_gt = observation_gt * 2 - 1
    observation_hat = observation_hat * 2 - 1
    observation_gt = observation_gt.contiguous()
    observation_hat = observation_hat.contiguous()
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    # assert frame >= 9, "FVD requires at least 9 frames"

    # output_dict["fvd"] = fvd_model.compute(
    #     torch.clamp(observation_hat, -1.0, 1.0),
    #     torch.clamp(observation_gt, -1.0, 1.0),
    # )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mse_frame(observation_hat, observation_gt).cpu().tolist()
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0, reduction="none", dim=[1, 2, 3]).cpu().tolist()
    ssim_batch_size = 200
    output_dict["ssim"] = []
    output_dict["uiqi"] = []
    for i in range(0, observation_hat.shape[0], ssim_batch_size):
        observation_hat_batch = observation_hat[i : i + ssim_batch_size]
        observation_gt_batch = observation_gt[i : i + ssim_batch_size]
        ssim = structural_similarity_index_measure(observation_hat_batch, observation_gt_batch, data_range=2.0, reduction="none").cpu().tolist()
        output_dict["ssim"].extend(ssim)
        uiqi = uiqi_frame(observation_hat_batch, observation_gt_batch).cpu().tolist()
        output_dict["uiqi"].extend(uiqi)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    output_dict["lpips"] = []
    lpips_batch_size = 64
    for i in range(0, observation_hat.shape[0], lpips_batch_size):
        observation_hat_batch = observation_hat[i : i + lpips_batch_size]
        observation_gt_batch = observation_gt[i : i + lpips_batch_size]
        lpips = lpips_model(observation_hat_batch, observation_gt_batch).flatten().cpu().tolist()
        output_dict["lpips"].extend(lpips)

    observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
    observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
    fid_model.update(observation_gt_uint8, real=True)
    fid_model.update(observation_hat_uint8, real=False)
    fid = fid_model.compute()
    output_dict["fid"] = fid.item()
    # Reset the states of non-functional metrics
    fid_model.reset()

    scaling_factor = 0.07843137255
    for name, image_discriminator in image_discriminators.items():
        output_dict[name] = []
        for i in range(0, observation_hat.shape[0], 128):
            observation_hat_batch = observation_hat[i : i + 128]
            observation_hat_batch = vae_model.encode(observation_hat_batch).mean * scaling_factor
            observation_hat_batch = rearrange(observation_hat_batch, "b (h w) c -> b c h w", h=18, w=32)
            output_dict[name].extend(image_discriminator(observation_hat_batch).cpu().tolist())
    return output_dict

@torch.no_grad()
def evaluate():
    device = "cuda"
    dtype = torch.float32
    validation_fid_model = FrechetInceptionDistance(feature=64).to(device)
    # validation_lpips_model = LearnedPerceptualImagePatchSimilarity().to(device)
    validation_lpips_model = lpips.LPIPS(net="vgg").to(device)
    # validation_fvd_model = FrechetVideoDistance().to(device)

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    image_discriminators = {
        "full_step_resnet": ImageDiscriminatorResNet(),
        "small_dit": ImageDiscriminator(depth=4, gradient_checkpointing=False),
        "1000_step_resnet": ImageDiscriminatorResNet(),
    }
    image_discriminators_ckpt = {
        "1000_step_resnet": "/home/tc0786/Project/train-oasis/models/discriminator/1000_step_resnet.pth",
        "small_dit": "/home/tc0786/Project/train-oasis/models/discriminator/small_dit.pth",
        "full_step_resnet": "/home/tc0786/Project/train-oasis/models/discriminator/full_step_resnet.pth",
    }
    for name, model in image_discriminators.items():
        ckpt_path = image_discriminators_ckpt[name]
        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist."
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model = model.to(device).to(dtype).eval()
        image_discriminators[name] = model

    model_names = ["rag_pred_pose", "rag_multi_pred_pose"] # ["vanilla_20_longer", "vanilla_40_direct_extrapolate", "frame_pack", "yarn", "historical_buffer", "rag", "infini_attn", "vanilla_10", "vanilla_20", "world_coordinate"]
    eval_split = ["memory", "random"]

    gt_file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(gt_file_path, "r") as f:
        gt_paths = json.load(f)

    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}/"
        assert os.path.exists(save_dir), f"Save directory {save_dir} does not exist."
        for split in eval_split:
            evaluation_output_save_path = os.path.join(save_dir, f"{split}.json")
            gt_split_paths = gt_paths[split]
            all_outputs = []
            for info in tqdm(gt_split_paths, desc=f"Evaluating {split} split"):
                gt_video_path = info["video_path"]
                save_relative_path = info["save_relative_path"]
                fake_save_path = os.path.join(save_dir, save_relative_path)
                assert os.path.exists(fake_save_path), f"Fake video path {fake_save_path} does not exist."
                gt_video, _, _ = read_video(gt_video_path, pts_unit="sec")
                gt_video = gt_video.contiguous().numpy()
                gt_video = torch.from_numpy(gt_video).float() / 255.0
                gt_video = gt_video.to(dtype).to(device)
                gt_video = gt_video.permute(0, 3, 1, 2)
                fake_video, _, _ = read_video(fake_save_path, pts_unit="sec")
                fake_video = fake_video.contiguous().numpy()
                fake_video = torch.from_numpy(fake_video).float() / 255.0
                fake_video = fake_video.to(dtype).to(device)
                fake_video = fake_video.permute(0, 3, 1, 2)
                min_frames = min(gt_video.shape[0], fake_video.shape[0])
                gt_video = gt_video[:min_frames]
                fake_video = fake_video[:min_frames]
                fake_video = fake_video.unsqueeze(1).contiguous()
                gt_video = gt_video.unsqueeze(1).contiguous()

                output_dict = get_validation_metrics_for_videos(
                    fake_video,
                    gt_video,
                    validation_lpips_model,
                    validation_fid_model,
                    image_discriminators,
                    vae,
                )
                all_outputs.append({
                    "save_path": fake_save_path,
                    "output_dict": output_dict,
                })
            with open(evaluation_output_save_path, "w") as f:
                json.dump(all_outputs, f, indent=4)
            print(f"Evaluation results saved to {evaluation_output_save_path}")

if __name__ == "__main__":
    evaluate()