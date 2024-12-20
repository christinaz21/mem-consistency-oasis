from vae import VAE_models
import torch
from safetensors.torch import load_model
from torchvision.io import read_video, write_png
from torchvision.transforms.functional import resize
import os
from einops import rearrange

def check_vae():
    device = "cuda:0"
    save_dir = "outputs/test_vae"
    ckpt_path = "/data/taiye/Project/open-oasis/models/oasis500m/vit-l-20.safetensors"

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={ckpt_path}...")
    if ckpt_path.endswith(".pt"):
        vae_ckpt = torch.load(ckpt_path, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif ckpt_path.endswith(".safetensors"):
        load_model(vae, ckpt_path)
    vae = vae.to(device).eval()

    video_path = "/data/taiye/Project/open-oasis/data/VPT/training/bumpy-pumpkin-dunker-5c7c23a408bf-20220104-193828.mp4"
    prompt = read_video(video_path, pts_unit="sec")[0]
    img = prompt[0]
    print(img.shape)
    x = rearrange(img, "h w c -> c h w")
    write_png(x, os.path.join(save_dir, "before.png"))
    x = x.float() / 255
    x = x.to(device)
    os.makedirs(save_dir, exist_ok=True)
    scaling_factor = 0.07843137255
    with torch.no_grad():
        # (c, h, w)
        x = rearrange(x, "c h w -> 1 c h w") # (1, c, h, w)
        x = vae.encode(x * 2 - 1).mean * scaling_factor # (1 (h w) c)
        x = (vae.decode(x / scaling_factor) + 1) / 2 # (1 c h w)
        x = rearrange(x, "1 c h w -> c h w")
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte().cpu()
    write_png(x, os.path.join(save_dir, "after.png"))

def check_load():
    from pytorchvideo.data.encoded_video import EncodedVideo
    path = "data/VPT/training/bumpy-pumpkin-dunker-f153ac423f61-20220203-230948.mp4"
    video = EncodedVideo.from_path(path, decode_audio=False)
    video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
    print(video.shape)

if __name__ == "__main__":
    print(os.cpu_count())