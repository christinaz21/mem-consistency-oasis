import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    A ResNet-based image discriminator.
    """
    def __init__(self, hidden_dim=256):
        super(ResNet, self).__init__()

        # (16, 18, 32) -> (16, 9, 16)
        self.layer1 = ResNetBlock(16, 32, stride=2)
        # (16, 9, 16) -> (32, 5, 8)
        self.layer2 = ResNetBlock(32, 64, stride=2)
        # (32, 5, 8) -> (64, 3, 4)
        self.layer3 = ResNetBlock(64, 128, stride=2)
        # (64, 3, 4) -> (128, 2, 2)
        self.layer4 = ResNetBlock(128, 256, stride=2)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, hidden_dim)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Predictor, self).__init__()
        self.image_handler = ResNet(hidden_dim)
        self.yaw_dim = 1
        self.pose_dim = 4
        self.action_dim = 4
        self.action_handler = nn.Sequential(
            nn.Linear(self.action_dim + self.yaw_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pose_dim)
        )

    def forward(self, image, action, yaw):
        image_features = self.image_handler(image) # (B, hidden_dim)
        action = torch.cat([action, yaw], dim=-1)
        action_features = self.action_handler(action) # (B, hidden_dim)
        x = torch.cat([image_features, action_features], dim=-1)
        x = self.mlp(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.metadata_file = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/20k.json"
        with open(self.metadata_file, 'r') as f:
            self.data_paths = json.load(f)[:4000]

        lengths = [x["length"] for x in self.data_paths]
        self.lengths = np.array(lengths)
        self.clips_per_video = np.clip(self.lengths - 2 + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)
        assert sum((data["length"] - 1) for data in self.data_paths) == self.cum_clips_per_video[-1]

    def __len__(self):
        return int(self.cum_clips_per_video[-1] * 1.0)
    
    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        file_idx, frame_idx = self.split_idx(idx)
        data_path = self.data_paths[file_idx]["file"]
        video_path = Path(data_path).with_suffix(".pt")
        video = torch.load(video_path, map_location="cpu", weights_only=True)
        video = video[frame_idx]
        actions = np.load(data_path)["actions"]
        input_action = actions[frame_idx+1][:4]
        input_pose = actions[frame_idx][4:]
        output_pose = actions[frame_idx + 1][4:]
        pred_pose = output_pose - input_pose
        input_action = torch.from_numpy(input_action).float()
        pred_pose = torch.from_numpy(pred_pose).float()
        input_pose = torch.from_numpy(input_pose).float()
        if pred_pose[3] > 180:
            pred_pose[3] -= 360
        elif pred_pose[3] < -180:
            pred_pose[3] += 360
        assert abs(pred_pose[3]) <= 180, f"Invalid angle: {pred_pose[3]}"
        pred_pose = pred_pose  / 20
        return video.to(torch.float32), input_action, input_pose[3:4], pred_pose



def train():
    print(f"Using device: {device}")
    save_model_interval = 300
    model = Predictor(hidden_dim=256)
    dataset = Dataset()
    dataloader = tqdm(torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, drop_last=True), desc="Training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(dataloader),  # Assuming 10 epochs
        num_cycles=0.5,
    )

    model.to(device)
    model.train()

    save_dir = "/home/tc0786/Project/train-oasis/outputs/df/predictor"
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    for images, actions, poses, targets in dataloader:
        images = images.to(device)
        actions = actions.to(device)
        poses = poses.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images, actions, poses)
        loss = criterion(outputs, targets)
        loss.backward()
        # print gradients norm
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        dataloader.set_description(f"Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")
        if step % save_model_interval == 0:
            save_path = os.path.join(save_dir, f"model_step_{step}.pth")
            torch.save(model.state_dict(), save_path)
        step += 1

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))

def regenerate_action():
    from train_oasis.model.vae import VAE_models
    from safetensors.torch import load_model
    from torchvision.io import read_video
    from torch import autocast
    from einops import rearrange
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/home/tc0786/Project/train-oasis/outputs/df/predictor/model_final.pth"
    model = Predictor(hidden_dim=256)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)

    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    old_metadata_file = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    new_metadata_file = "/home/tc0786/Project/train-oasis/data/eval_data/pred_pose_paths.json"
    with open(old_metadata_file, 'r') as f:
        data_paths = json.load(f)

    new_metadata = {}
    for split in ["memory", "random"]:
        new_metadata[split] = []
        for data in data_paths[split]:
            video_path = data["video_path"]
            action_path = data["action_path"]
            save_relative_path = data["save_relative_path"]
            video, _, _ = read_video(video_path, pts_unit='sec')
            video = video.permute(0, 3, 1, 2).to(device).float() / 255.0 # [B, C, H, W]
            video = video.to(device)
            vae_batch_size = 128
            scaling_factor = 0.07843137255
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, video.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = video[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                video = torch.cat(all_frames, dim=0) # [B, C, H, W]
            video = rearrange(video, "b (h w) c -> b c h w", h=18, w=32)
            video = video.to(device).to(torch.float32)
            actions = np.load(action_path)["actions"]
            input_actions = actions[:, :4]
            input_poses = actions[:, 4:]
            pred_poses = []
            pred_poses.append(input_poses[0])
            pred_poses.append(input_poses[1])
            for i in range(2, input_poses.shape[0]):
                input_action = torch.from_numpy(input_actions[i]).float().to(device)
                input_pose = torch.from_numpy(pred_poses[-1][3:4]).float().to(device)
                with torch.no_grad():
                    pred_pose = model(video[i-1].unsqueeze(0), input_action.unsqueeze(0), input_pose.unsqueeze(0))
                pred_pose = pred_pose.squeeze(0).cpu().numpy()
                pred_pose[3] *= 20
                pred_pose += pred_poses[-1]
                if pred_pose[3] > 180:
                    pred_pose[3] -= 360
                elif pred_pose[3] < -180:
                    pred_pose[3] += 360
                pred_poses.append(pred_pose)

            pred_poses = np.array(pred_poses)
            pred_poses = np.concatenate([input_actions, pred_poses], axis=-1)
            save_path = action_path.replace("memory", "pred/memory").replace("random", "pred/random")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, actions=pred_poses)
            new_metadata[split].append({
                "video_path": video_path,
                "action_path": save_path,
                "save_relative_path": save_relative_path
            })

    with open(new_metadata_file, 'w') as f:
        json.dump(new_metadata, f, indent=4)


if __name__ == "__main__":
    # train()
    regenerate_action()
    # dataset = Dataset()
    # for i in range(100):
    #     video, input_action, input_pose, pred_pose = dataset[i]
    #     print(input_action, input_pose, pred_pose)