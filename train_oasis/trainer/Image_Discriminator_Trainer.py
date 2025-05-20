import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

from train_oasis.model.image_discriminator import ImageDiscriminator, ImageDiscriminatorResNet
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from train_oasis.model.vae import VAE_models

import torch
import json
import numpy as np
from tqdm import tqdm
from einops import rearrange
import wandb
from safetensors.torch import load_model

class DatasetPro:
    def __init__(self, dtype=torch.bfloat16, data=None, labels=None):
        if data is not None and labels is not None:
            self.data = data
            self.labels = labels
            return
        self.root_dir = "/data/taiye/Project/train-oasis/outputs/sample"
        self.metadata_file = os.path.join(self.root_dir, "metadata.json")
        self.prompt_num = 5
        if not os.path.exists(self.metadata_file):
            temp_data_paths = []
            sub_dirs = os.listdir(self.root_dir)
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(self.root_dir, sub_dir)
                if os.path.isdir(sub_dir_path):
                    pt_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.pt')]
                    for pt_file in pt_files:
                        pt_file_path = os.path.join(sub_dir_path, pt_file)
                        temp_data_paths.append(pt_file_path)
            self.data_paths = []
            for pt_file_path in tqdm(temp_data_paths, desc="Loading data paths"):
                try:
                    pt_data = torch.load(pt_file_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(f"Error loading {pt_file_path}: {e}")
                    raise e
                if "gt" in pt_file_path:
                    l = pt_data.shape[0]
                    d = {
                        "path": pt_file_path,
                        "length": l,
                        "label": 1
                    }
                else:
                    b = pt_data.shape[0]
                    t = pt_data.shape[1]
                    d = {
                        "path": pt_file_path,
                        "length": b * (t - self.prompt_num),
                        "label": 0
                    }
                self.data_paths.append(d)
                del pt_data
            with open(self.metadata_file, 'w') as f:
                json.dump(self.data_paths, f, indent=4)
        else:
            with open(self.metadata_file, 'r') as f:
                self.data_paths = json.load(f)
        
        lengths = [data["length"] for data in self.data_paths]
        lengths = np.array(lengths)
        self.dtype = dtype

        self.data = []
        self.labels = []
        for path in self.data_paths:
            video_data = torch.load(path["path"], map_location="cpu", weights_only=True)
            if path["label"] == 1:
                video = video_data
            else:
                video = rearrange(video_data[:, self.prompt_num:], "b t c h w -> (b t) c h w")
            self.data.append(video)
            labels = torch.full((video.shape[0],), path["label"], dtype=torch.int64)
            self.labels.append(labels)
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        # Shuffle self.data and self.labels
        indices = torch.randperm(self.data.shape[0])
        self.data = self.data[indices]
        self.labels = self.labels[indices]

        self.data = self.data.to(self.dtype)
        self.labels = self.labels.to(self.dtype)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train():
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    model_names = ["yarn", "historical_buffer", "rag", "infini_attn", "vanilla_10", "vanilla_20", "world_coordinate"]
    train_split = ["random"]

    

    lr = 5e-5
    RUN_NAME = f'lr_{lr}_512_resnet_1000step'
    PROJECT_NAME = 'ImageDiscriminator'
    WANDB_ONLINE = True # turn this on to pipe experiment to cloud
    save_dir = f'./outputs/ImageDiscriminator/{RUN_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=PROJECT_NAME, dir=save_dir, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()

    batch_size = 512
    warmup_step = 200
    num_epochs = 1
    eval_step = 100
    train_limit = 512000
    val_limit = 5120

    # dtype = torch.bfloat16
    dtype = torch.float32
    dataset = DatasetPro(dtype=dtype)
    train_dataset, val_dataset = dataset.split(train_limit=train_limit,val_limit=val_limit)  # Split into train and validation datasets
    del dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    device = "cuda"
    # model = ImageDiscriminator(depth=4, dtype=dtype, gradient_checkpointing=False).to(device).to(dtype)
    model = ImageDiscriminatorResNet().to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=num_epochs * len(train_dataloader),
        num_cycles=1,
    )

    def eval():
        model.eval()
        val_loss = 0
        val_acc = 0
        total = 0
        with torch.no_grad():
            for val_video, val_label in tqdm(val_dataloader, desc="Validation", leave=False):
                val_video = val_video.to(device)
                val_label = val_label.to(device).to(dtype)
                val_output = model(val_video)
                # Mask out NaN values in val_output and corresponding val_label
                nan_mask = ~torch.isnan(val_output)
                val_output = val_output[nan_mask]
                val_label = val_label[nan_mask]
                if val_output.shape[0] == 0:
                    print("No valid output for this batch, skipping...")
                    continue
                total += val_output.shape[0]
                # Calculate loss and accuracy
                val_loss += criterion(val_output, val_label).item() * val_output.shape[0]
                val_acc += ((val_output > 0.5).float() == val_label).float().sum().item()
        val_loss /= total
        val_acc /= total
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        model.train()

    pbar = tqdm(range(num_epochs * len(train_dataloader)), desc="Training")
    eval()
    for epoch in range(num_epochs):
        for step, (video, label) in enumerate(train_dataloader):
            video = video.to(device)
            label = label.to(device).to(dtype)

            optimizer.zero_grad()
            output = model(video)
            if torch.isnan(output).any():
                print("NaN detected in output, skipping this batch.")
                del output
            else:
                acc = (output > 0.5).float()
                acc = (acc == label).float().mean()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                log_loss = loss.item()
                log_acc = acc.item()
            scheduler.step()

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            pbar.set_postfix({"loss": log_loss, "acc": log_acc})
            wandb.log({"loss": log_loss, "acc": log_acc, "lr": optimizer.param_groups[0]["lr"]})

            # Validation every 1000 steps
            if step > 0 and step % eval_step == 0:
                eval()
        
        eval()
        # Save the model
        os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "ckpt", f"model_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    train()