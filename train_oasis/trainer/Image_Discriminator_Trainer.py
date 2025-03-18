"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""
from typing import Any, Union, Sequence, Optional
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    FrechetVideoDistance, 
    extract, 
    sigmoid_beta_schedule, 
    convert_zero_ckpt_into_state_dict,
    WarmUpScheduler
)

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch.distributed as dist
from train_oasis.model.dit import DiT
from train_oasis.model.image_discriminator import ImageDiscriminator
from train_oasis.model.vae import AutoencoderKL
from safetensors.torch import load_model

class ImageDiscriminatorTrainer(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, model_ckpt: str = None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.x_shape = cfg.x_shape
        # if self.cfg.vae_name == "oasis":
        #     self.x_shape = [16, 18, 32]
        # elif self.cfg.vae_name == "flappy_bird":
        #     self.x_shape = [4, 64, 36]
        self.vae_name = cfg.vae_name
        self.external_cond_dim = cfg.external_cond_dim

        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        self.stabilization_level = cfg.diffusion.stabilization_level

        self.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** cfg.frame_skip

        self.validation_step_outputs = []
        self.metrics = cfg.metrics
        self.n_frames = cfg.n_frames  # number of max tokens for the model

        self.snr_clip = cfg.diffusion.snr_clip
        self.scaling_factor = cfg.scaling_factor
        self.predict_v = cfg.diffusion.predict_v
        self.inference_context_length = cfg.inference_context_length
        self.inference_length = cfg.inference_length
        assert self.sampling_timesteps % (self.n_frames-1) == 0, "The number of frames should be divisible by the number of frames."
        self.ddim_one_step = self.sampling_timesteps // (self.n_frames-1)
        
        weight = (self.inference_length - self.inference_context_length) / self.inference_length
        self.loss_function = nn.BCEWithLogitsLoss(weight=torch.tensor(weight))
        
        self._build_model(model_ckpt)
        self._build_buffer()

    def register_data_mean_std(
        self, mean: Union[str, float, Sequence], std: Union[str, float, Sequence], namespace: str = "data"
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))


    def _build_model(self, model_ckpt):
        self.diffusion_model = DiT(
            input_h=self.model_cfg.input_h,
            input_w=self.model_cfg.input_w,
            patch_size=self.model_cfg.patch_size,
            in_channels=self.model_cfg.in_channels,
            hidden_size=self.model_cfg.hidden_size,
            depth=self.model_cfg.depth,
            num_heads=self.model_cfg.num_heads,
            mlp_ratio=self.model_cfg.mlp_ratio,
            external_cond_dim=self.external_cond_dim,
            max_frames=self.cfg.n_frames,
            gradient_checkpointing=self.model_cfg.gradient_checkpointing,
            dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
        )
        assert model_ckpt, "Model checkpoint is required for the diffusion model."
        print(f"Loading Diffusion model from {model_ckpt}")
        state_dict = convert_zero_ckpt_into_state_dict(model_ckpt)
        self.diffusion_model.load_state_dict(state_dict, strict=True)
        self.diffusion_model.eval()
        
        self.vae = AutoencoderKL(
            latent_dim=16,
            patch_size=20,
            enc_dim=1024,
            enc_depth=6,
            enc_heads=16,
            dec_dim=1024,
            dec_depth=12,
            dec_heads=16,
            input_height=360,
            input_width=640,
        )
        assert self.cfg.vae_ckpt, "VAE checkpoint is required for oasis VAE."
        load_model(self.vae, self.cfg.vae_ckpt)
        self.vae.eval()

        self.image_discriminator = ImageDiscriminator()
        
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None

    def _build_buffer(self):
        global_nan_number = torch.tensor(0, dtype=torch.float)
        self.register_buffer("global_nan_number", global_nan_number)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        betas = sigmoid_beta_schedule(self.timesteps).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        register_buffer("alphas_cumprod", alphas_cumprod)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()
        register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        clipped_snr = self.snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)
        register_buffer("clipped_snr", clipped_snr)
        
        
        # The first frame is stabilized, noise level for the last 9 frames
        noise_level_matrix = torch.zeros((self.ddim_one_step + 1, (self.n_frames-1)), dtype=torch.long)
        noise_range = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1, dtype=torch.long)
        for i in range(self.n_frames-1):
            for step in range(self.ddim_one_step+1):
                noise_level_matrix[step, i] = noise_range[step + i*self.ddim_one_step]

        self.register_buffer("noise_level_matrix", noise_level_matrix)

    def configure_optimizers(self):
        params = tuple(self.image_discriminator.parameters())
        if self.cfg.strategy == "ddp":
            optimizer_dynamics = torch.optim.AdamW(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
            )
            scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        elif self.cfg.strategy == "deepspeed":
            optimizer_dynamics = DeepSpeedCPUAdam(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
            )
            scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        else:
            raise ValueError(f"Unsupported strategy {self.cfg.strategy}.")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(step=self.trainer.global_step)

    def q_sample(self, x_start, t, noise=None):
        # t random(0, timestep)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_v_from_x(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        xs = self.vae_encode(xs)
        xs_gt = xs.clone()
        noise = torch.randn_like(xs)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noise_levels = self._generate_noise_levels(xs, masks)
        noised_x = self.q_sample(x_start=xs, t=noise_levels, noise=noise)
        with torch.no_grad():
            model_pred = self.diffusion_model(
                x=rearrange(noised_x, "t b ... -> b t ..."),
                t=rearrange(noise_levels, "t b -> b t"),
                external_cond=rearrange(conditions, "t b ... -> b t ...") if conditions is not None else None,
            )
        model_pred = rearrange(model_pred, "b t ... -> t b ...")
        nan_number = torch.isnan(model_pred).sum()
        dist.all_reduce(nan_number, op=dist.ReduceOp.SUM)
        if nan_number != 0:
            loss = torch.tensor(0.0, dtype=xs_gt.dtype, requires_grad=True, device=self.device)
            self.global_nan_number += 1
            self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=True)
            output_dict = {
                "loss": loss,
            }
            return output_dict
        if self.predict_v:
            model_pred = self.predict_start_from_v(noised_x, noise_levels, model_pred)
        prompt = rearrange(xs, "t b ... -> b t ...")[:, :self.inference_context_length]
        sample_pred = self.sample_step(prompt, rearrange(conditions, "t b ... -> b t ...") if conditions is not None else None)
        sample_pred = rearrange(sample_pred, "b t ... -> t b ...")
        positive = rearrange(xs_gt, "t b ... -> (t b) ...")
        negative = torch.cat([rearrange(model_pred, "t b ... -> (t b) ..."), rearrange(sample_pred[1:, ...], "t b ... -> (t b) ...")], 0)
        all_labels = torch.cat([torch.ones(positive.shape[0]), torch.zeros(negative.shape[0])], 0).to(self.device)
        all_images = torch.cat([positive, negative], 0)
        dis_pred = self.image_discriminator(all_images)
        loss = self.loss_function(dis_pred, all_labels)

        # log the loss
        self.log("training/loss", loss, sync_dist=True, prog_bar=True)

        output_dict = {
            "loss": loss,
        }
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        print("Validation step")
        return None

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    @torch.no_grad()
    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
    ):
        B, T, *x_shape = x.shape
        assert T == self.inference_context_length, "The length of the context should be equal to the inference context length."
        # sampling loop
        for i in range(self.inference_context_length, self.inference_length):
            chunk = torch.randn((B, 1, *x.shape[-3:]), dtype=x.dtype).to(x.device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -self.clip_noise, +self.clip_noise)
            x = torch.cat([x, chunk], dim=1)
            # print(x[0][-1])
            start_frame = max(0, i + 1 - self.n_frames)
            horizon = min(self.n_frames, i + 1)

            for noise_idx in reversed(range(1, self.ddim_one_step + 1)):
                t = torch.full((B, horizon), self.stabilization_level - 1, dtype=torch.long, device=x.device)
                t[:, 1:] = self.noise_level_matrix[noise_idx, self.n_frames-horizon:]
                t_next = torch.full((B, horizon), self.stabilization_level - 1, dtype=torch.long, device=x.device)
                t_next[:, 1:] = self.noise_level_matrix[noise_idx-1, self.n_frames-horizon:]
                t_next = torch.where(t_next < 0, t, t_next)

                # sliding window
                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    v = self.diffusion_model(x_curr, t, external_cond[:, start_frame : i + 1])

                if self.predict_v:
                    x_start = self.predict_start_from_v(x_curr, t, v)
                else:
                    x_start = v
                x_noise = self.predict_noise_from_start(x_curr, t, x_start)

                # get frame prediction
                alpha_next = self.alphas_cumprod[t_next]
                alpha_next = self.add_shape_channels(alpha_next)
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1 and horizon == self.n_frames:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, start_frame+1:] = x_pred[:, -(horizon-1):]
            
        # handle last max_frame - 1 frames
        for i in range(self.inference_length, self.inference_length + self.n_frames - 2):
            start_frame = max(0, i + 1 - self.n_frames)
            horizon = min(self.n_frames, self.n_frames + self.inference_length - 1 - i)

            for noise_idx in reversed(range(1, self.ddim_one_step + 1)):
                t = torch.full((B, horizon), self.stabilization_level - 1, dtype=torch.long, device=x.device)
                t[:, 1:] = self.noise_level_matrix[noise_idx, :horizon-1]
                t_next = torch.full((B, horizon), self.stabilization_level - 1, dtype=torch.long, device=x.device)
                t_next[:, 1:] = self.noise_level_matrix[noise_idx-1, :horizon-1]
                t_next = torch.where(t_next < 0, t, t_next)

                # sliding window
                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    v = self.diffusion_model(x_curr, t, external_cond[:, start_frame : i + 1])

                if self.predict_v:
                    x_start = self.predict_start_from_v(x_curr, t, v)
                else:
                    x_start = v
                x_noise = self.predict_noise_from_start(x_curr, t, x_start)

                # get frame prediction
                alpha_next = self.alphas_cumprod[t_next]
                alpha_next = self.add_shape_channels(alpha_next)
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, start_frame+1:] = x_pred[:, -(horizon-1):]
        return x

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = ~masks.bool()
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def _generate_scheduling_matrix(self, horizon: int):
        height = self.sampling_timesteps + int((horizon - 1) * self.sampling_timesteps) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * self.sampling_timesteps) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    @torch.no_grad()
    def vae_encode(self, x):
        batch_size, n_frames, c, h, w = x.shape # the order of the first two dimensions can be ignored
        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.vae.encode(x).mean * self.scaling_factor
        x = rearrange(x, "(b t) (h w) c -> b t c h w", b=batch_size, t=n_frames, h=18, w=32, c=16)
        return x

    @torch.no_grad()
    def vae_decode(self, x):
        # input: (b, t, c, h, w)
        batch_size, n_frames, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) (h w) c")
        x = self.vae.decode(x / self.scaling_factor)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames)
        return x

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames, c, h, w = xs.shape

        masks = torch.ones(n_frames, batch_size).to(self.device)

        if self.external_cond_dim > 0:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b t d -> t b d").contiguous()
        else:
            conditions = None

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b t c ... -> t b c ...").contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        xs = (xs - mean) / std
        return xs

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        xs = xs * std + mean
        return xs
