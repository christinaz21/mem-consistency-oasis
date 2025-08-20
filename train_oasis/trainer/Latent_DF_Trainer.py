"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""
from typing import Any, Union, Sequence, Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    FrechetVideoDistance, 
    get_validation_metrics_for_videos, 
    log_video, 
    extract, 
    sigmoid_beta_schedule, 
    convert_zero_ckpt_into_state_dict,
    WarmUpScheduler
)

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

class LatentDFVideo(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, model_ckpt: str = None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.x_shape = cfg.x_shape
        self.external_cond_dim = cfg.external_cond_dim
        self.scheduler = cfg.scheduler

        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        self.stabilization_level = cfg.diffusion.stabilization_level

        self.cum_snr_decay = self.cfg.diffusion.cum_snr_decay

        self.validation_step_outputs = []
        self.metrics = cfg.metrics
        self.n_frames = cfg.n_frames  # number of max tokens for the model

        self.snr_clip = cfg.diffusion.snr_clip
        self.predict_v = cfg.diffusion.predict_v
        
        self._build_model(model_ckpt)
        self._build_buffer()

    def _build_model(self, model_ckpt):
        if self.model_cfg.architecture == "oasis_dit":
            from train_oasis.model.dit import DiT
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
        elif self.model_cfg.architecture == "action_dit":
            from train_oasis.model.action_dit import DiT
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
        else:
            raise ValueError(f"Unsupported architecture {self.model_cfg.architecture}.")
        
        if model_ckpt:
            print(f"Loading Diffusion model from {model_ckpt}")
            state_dict = convert_zero_ckpt_into_state_dict(model_ckpt)
            self.diffusion_model.load_state_dict(state_dict, strict=True)

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

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        if self.cfg.strategy == "ddp":
            optimizer_dynamics = torch.optim.AdamW(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
            )
            if self.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer_dynamics,
                    num_warmup_steps=self.cfg.warmup_steps,
                    num_training_steps=self.trainer.estimated_stepping_batches,
                )
                if self.trainer.is_global_zero:
                    print(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
            elif self.scheduler == "warmup":
                scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            else:
                raise ValueError(f"Unsupported scheduler {self.scheduler}.")
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        elif self.cfg.strategy == "deepspeed":
            optimizer_dynamics = DeepSpeedCPUAdam(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
            )
            if self.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer_dynamics,
                    num_warmup_steps=self.cfg.warmup_steps,
                    num_training_steps=self.trainer.estimated_stepping_batches,
                )
                if self.trainer.is_global_zero:
                    print(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
            elif self.scheduler == "warmup":
                scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            else:
                raise ValueError(f"Unsupported scheduler {self.scheduler}.")
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        else:
            raise ValueError(f"Unsupported strategy {self.cfg.strategy}.")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

    def lr_scheduler_step(self, scheduler, metric):
        if self.scheduler == "cosine":
            scheduler.step()
        elif self.scheduler == "warmup":
            scheduler.step(self.trainer.global_step)
        else:
            raise ValueError(f"Unsupported scheduler {self.scheduler}.")

    def compute_loss_weights(self, noise_levels: torch.Tensor):
        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[0]):
            if t == 0:
                cum_snr[t] = normalized_clipped_snr[t]
            else:
                cum_snr[t] = self.cum_snr_decay * cum_snr[t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]

        cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        if self.predict_v:
            fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)
            return clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
        else:
            return clipped_fused_snr * self.snr_clip

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
        xs_gt = xs.clone()
        noise = torch.randn_like(xs)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noise_levels = self._generate_noise_levels(xs, masks)
        noised_x = self.q_sample(x_start=xs, t=noise_levels, noise=noise)
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
        else:
            if self.predict_v:
                target = self.predict_v_from_x(xs, noise_levels, noise)
            else:
                target = xs
            loss = F.mse_loss(model_pred, target.detach(), reduction="none")
            loss_weight = self.compute_loss_weights(noise_levels)
            loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
            loss = loss * loss_weight
            loss = self.reweight_loss(loss, masks)
        
        # log the loss
        self.log("training/loss", loss, sync_dist=True, prog_bar=True)
        self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=False)
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True, prog_bar=False)

        output_dict = {
            "loss": loss,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
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
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
    ):
        real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device).long()

        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        curr_noise_level = real_steps[curr_noise_level]
        next_noise_level = real_steps[next_noise_level]

        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        c = (1 - alpha_next).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)

        model_pred = self.diffusion_model(
            x=rearrange(x, "t b ... -> b t ..."),
            t=rearrange(clipped_curr_noise_level, "t b -> b t"),
            external_cond=rearrange(external_cond, "t b ... -> b t ...") if external_cond is not None else None,
        )
        model_pred = rearrange(model_pred, "b t ... -> t b ...")
        
        if self.predict_v:
            x_start = self.predict_start_from_v(x, clipped_curr_noise_level, model_pred)
        else:
            x_start = model_pred
        pred_noise = self.predict_noise_from_start(x, clipped_curr_noise_level, x_start)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        return

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

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

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "t b ... -> t b ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames, c, h, w = xs.shape

        masks = torch.ones(n_frames, batch_size).to(self.device)

        conditions = batch[1]
        conditions = rearrange(conditions, "b t d -> t b d").contiguous()
        
        xs = rearrange(xs, "b t c ... -> t b c ...").contiguous()

        return xs, conditions, masks
