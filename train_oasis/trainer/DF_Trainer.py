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
import os
import math

class DiffusionForcingVideo(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, model_ckpt: str = None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.x_shape = cfg.x_shape
        self.vae_name = cfg.vae_name
        self.max_vae_batch = cfg.max_vae_batch
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
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
        elif self.model_cfg.architecture == "sora_dit":
            from train_oasis.model.open_sora_dit import DiT
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
                use_causal_mask=self.model_cfg.use_causal_mask,
                gradient_checkpointing=self.model_cfg.gradient_checkpointing,
                dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
            )
        elif self.model_cfg.architecture == "mla_dit":
            from train_oasis.model.mla_dit import DiT
            self.diffusion_model = DiT(
                input_h=self.model_cfg.input_h,
                input_w=self.model_cfg.input_w,
                patch_size=self.model_cfg.patch_size,
                in_channels=self.model_cfg.in_channels,
                hidden_size=self.model_cfg.hidden_size,
                lora_rank=self.model_cfg.lora_rank,
                depth=self.model_cfg.depth,
                num_heads=self.model_cfg.num_heads,
                mlp_ratio=self.model_cfg.mlp_ratio,
                external_cond_dim=self.external_cond_dim,
                max_frames=self.cfg.n_frames,
                gradient_checkpointing=self.model_cfg.gradient_checkpointing,
                dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
            )
        elif self.model_cfg.architecture == "lstm_dit":
            from train_oasis.model.lstm_dit import DiT
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
                lstm_dropout=self.model_cfg.lstm_dropout,
                lstm_layer_num=self.model_cfg.lstm_layer_num,
                inner_window_size=self.model_cfg.inner_window_size,
                gradient_checkpointing=self.model_cfg.gradient_checkpointing,
                dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
            )
        elif self.model_cfg.architecture == "mamba_dit":
            from train_oasis.model.mamba_dit import DiT
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
                mamba_d_state=self.model_cfg.mamba_d_state,
                mamba_d_conv=self.model_cfg.mamba_d_conv,
                mamba_expand=self.model_cfg.mamba_expand,
                gradient_checkpointing=self.model_cfg.gradient_checkpointing,
                dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
            )
        elif self.model_cfg.architecture == "rnn_chunk":
            from train_oasis.model.rnn_chunk_dit import DiT
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
                max_frames=self.model_cfg.inner_window_size,
                rnn_config=self.model_cfg.rnn_config,
                gradient_checkpointing=self.model_cfg.gradient_checkpointing,
                dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
            )
        else:
            raise ValueError(f"Unsupported model {self.model_cfg._name}.")
        
        if model_ckpt:
            print(f"Loading Diffusion model from {model_ckpt}")
            sft_rnn = self.model_cfg.get("sft_rnn", False)
            if self.model_cfg.architecture == "rnn_chunk" and sft_rnn:
                strict_load = False
            else:
                strict_load = True
            if os.path.isdir(model_ckpt):
                state_dict = convert_zero_ckpt_into_state_dict(model_ckpt)
                self.diffusion_model.load_state_dict(state_dict, strict=strict_load)
            elif model_ckpt.endswith(".ckpt"):
                ckpt = torch.load(model_ckpt, map_location="cpu")
                state_dict = {}
                for key, value in ckpt['state_dict'].items():
                    if key.startswith("diffusion_model."):
                        state_dict[key[16:]] = value
                self.diffusion_model.load_state_dict(state_dict, strict=strict_load)
            elif model_ckpt.endswith(".bin"):
                state_dict = torch.load(model_ckpt, map_location="cpu")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("diffusion_model."):
                        new_state_dict[key[16:]] = value
                self.diffusion_model.load_state_dict(new_state_dict, strict=strict_load)
            else:
                state_dict = torch.load(model_ckpt, map_location="cpu")
                self.diffusion_model.load_state_dict(state_dict, strict=strict_load)
            if self.model_cfg.architecture == "rnn_chunk" and sft_rnn:
                trainable_param_names = ["rnn", "r_norm", "r_adaLN_modulation", "combine_action_proj"]
                for name, param in self.diffusion_model.named_parameters():
                    if any([tp_name in name for tp_name in trainable_param_names]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                
                if self.global_rank == 0:
                    print("Only finetuning the RNN parameters.")
                    for name, param in self.diffusion_model.named_parameters():
                        if param.requires_grad:
                            print(f"Trainable parameter: {name}, shape: {param.shape}")
                        else:
                            print(f"Frozen parameter: {name}, shape: {param.shape}")
        
        if self.cfg.vae_name == "oasis":
            from train_oasis.model.vae import AutoencoderKL
            from safetensors.torch import load_model
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
        elif self.cfg.vae_name == "sd_vae":
            assert self.cfg.vae_ckpt
            from diffusers.models import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_ckpt)
            self.vae.eval()
        else:
            self.vae = None
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

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
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
        xs = self.vae_encode(xs)
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
        if dist.is_initialized():
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

        output_dict = {
            "loss": loss,
        }
        if batch_idx % self.cfg.save_video_every_n_step == 0 and self.logger and self.global_rank == 0 and False:
            xs_gt = self._unnormalize_x(xs_gt)
            if self.predict_v:
                model_pred = self.predict_start_from_v(noised_x, noise_levels, model_pred)
            model_pred = self.vae_decode(model_pred)
            model_pred = self._unnormalize_x(model_pred)
            log_video(
                model_pred[:, :8],
                xs_gt[:, :8],
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        xs_gt = xs.clone()
        xs = self.vae_encode(xs)
        n_frames, batch_size, *_ = xs.shape
        curr_frame = 0

        # context
        n_context_frames = self.context_frames
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_frames, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_shape), device=self.device, dtype=xs_pred.dtype)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_frames frames
            start_frame = max(0, curr_frame + horizon - self.n_frames)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon] if conditions is not None else None,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)


        xs_gt = self._unnormalize_x(xs_gt)
        xs_pred = self.vae_decode(xs_pred)
        xs_pred = self._unnormalize_x(xs_pred)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs_gt.detach().cpu()))

        return loss

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

    def sample_step_with_logprob(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        *,
        eta: float,
        sigma_min: float = 1e-3,
        prev_sample: Optional[torch.Tensor] = None,
        model=None,
    ):
        """
        Stochastic DDIM-style step with per-transition log-prob.

        Args:
            x:               (T, B, C, H, W)
            external_cond:   (T, B, D) or None
            curr_noise_level:(T, B) in sampling-step index space [0, sampling_timesteps]
            next_noise_level:(T, B) in sampling-step index space [0, sampling_timesteps]
            eta:             > 0 for stochastic transition
            sigma_min:       lower bound for stddev
            prev_sample:     if provided, score this exact transition instead of sampling anew
            model:           policy to use; defaults to self.diffusion_model

        Returns:
            x_next:          sampled / provided next latent, same shape as x
            log_prob:        (T, B) transition log-prob
            active_mask:     (T, B) where curr != next
            clipped_curr:    (T, B) real timestep indices used for model forward
        """
        if eta <= 0:
            raise ValueError("eta must be > 0 for stochastic GRPO log-probs")

        if model is None:
            model = self.diffusion_model

        # map scheduler indices [0..sampling_timesteps] -> real train diffusion indices [-1..timesteps-1]
        real_steps = torch.linspace(
            -1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device
        ).long()

        curr_real = real_steps[curr_noise_level]
        next_real = real_steps[next_noise_level]

        # same stabilization handling as your existing sample_step
        clipped_curr = torch.where(
            curr_real < 0,
            torch.full_like(curr_real, self.stabilization_level - 1, dtype=torch.long),
            curr_real,
        )

        orig_x = x
        scaled_context = self.q_sample(
            x,
            clipped_curr,
            noise=torch.zeros_like(x),
        )
        x_in = torch.where(self.add_shape_channels(curr_real < 0), scaled_context, orig_x)

        model_pred = model(
            x=rearrange(x_in, "t b ... -> b t ..."),
            t=rearrange(clipped_curr, "t b -> b t"),
            external_cond=(
                rearrange(external_cond, "t b ... -> b t ...")
                if external_cond is not None
                else None
            ),
        )
        model_pred = rearrange(model_pred, "b t ... -> t b ...")

        if self.predict_v:
            x_start = self.predict_start_from_v(x_in, clipped_curr, model_pred)
        else:
            x_start = model_pred

        pred_noise = self.predict_noise_from_start(x_in, clipped_curr, x_start)

        # active positions are the ones whose noise level actually changes
        active_mask = curr_real != next_real

        # alpha_t and alpha_prev in real diffusion-time index space
        alpha_t = torch.ones_like(curr_real, dtype=torch.float32, device=x.device)
        alpha_prev = torch.ones_like(next_real, dtype=torch.float32, device=x.device)

        pos_curr = curr_real >= 0
        pos_prev = next_real >= 0
        alpha_t[pos_curr] = self.alphas_cumprod[curr_real[pos_curr]]
        alpha_prev[pos_prev] = self.alphas_cumprod[next_real[pos_prev]]

        # variance only matters on active positions
        variance = torch.zeros_like(alpha_t)
        active_idx = active_mask & pos_curr
        if active_idx.any():
            at = alpha_t[active_idx]
            ap = alpha_prev[active_idx]
            variance[active_idx] = ((1.0 - ap) / (1.0 - at)) * (1.0 - at / ap)
            variance[active_idx] = torch.clamp(variance[active_idx], min=1e-20)

        bc = (1,) * (x.ndim - 2)
        alpha_prev_b = alpha_prev.view(*alpha_prev.shape, *bc)

        sigma = eta * variance.sqrt()
        sigma = torch.clamp(sigma, min=0.0)
        sigma_b = sigma.view(*sigma.shape, *bc)

        dir_coeff = torch.clamp(1.0 - alpha_prev_b - sigma_b * sigma_b, min=0.0).sqrt()
        prev_mean = x_start * alpha_prev_b.sqrt() + pred_noise * dir_coeff

        if prev_sample is None:
            z = torch.randn_like(x)
            sampled = prev_mean + sigma_b * z
        else:
            sampled = prev_sample

        # unchanged positions should stay identical
        x_next = torch.where(self.add_shape_channels(active_mask), sampled, orig_x)

        # log prob only defined / needed on active positions
        log_prob = torch.zeros_like(curr_real, dtype=torch.float32)
        if active_idx.any():
            sigma_eval = torch.clamp(sigma[active_idx], min=sigma_min)
            mean_eval = prev_mean[active_idx]
            sample_eval = sampled[active_idx]

            expand = (None,) * (sample_eval.ndim - 1)
            sigma_b = sigma_eval[(slice(None),) + expand]

            lp = (
                -((sample_eval.detach().float() - mean_eval.float()) ** 2)
                / (2.0 * sigma_b ** 2)
                - torch.log(sigma_b)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_prob[active_idx] = lp.mean(dim=tuple(range(1, lp.ndim)))

        return x_next, log_prob, active_mask, clipped_curr
        
    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return
        xs_pred = []
        xs = []
        for pred, gt in self.validation_step_outputs:
            xs_pred.append(pred)
            xs.append(gt)
        xs_pred = torch.cat(xs_pred, 1)
        xs = torch.cat(xs, 1)

        if self.logger:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.validation_step_outputs.clear()

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

    def _generate_scheduling_matrix(self, horizon: int):
        height = self.sampling_timesteps + int((horizon - 1) * self.sampling_timesteps) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * self.sampling_timesteps) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

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

    @torch.no_grad()
    def vae_encode(self, x):
        if not self.vae:
            return x
        elif self.vae_name == "oasis":
            batch_size, n_frames, c, h, w = x.shape # the order of the first two dimensions can be ignored
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.vae.encode(x).mean * self.scaling_factor
            x = rearrange(x, "(b t) (h w) c -> b t c h w", b=batch_size, t=n_frames, h=18, w=32, c=16)
            return x
        elif self.vae_name == "sd_vae":
            batch_size, n_frames, c, h, w = x.shape # the order of the first two dimensions can be ignored
            x = rearrange(x, "b t ... -> (b t) ...")
            all_outputs = []
            for start_idx in range(0, x.shape[0], self.max_vae_batch):
                end_idx = min(start_idx + self.max_vae_batch, x.shape[0])
                x_chunk = x[start_idx:end_idx]
                x_chunk = self.vae.encode(x_chunk).latent_dist.sample() * self.vae.config.scaling_factor
                all_outputs.append(x_chunk)
            x = torch.cat(all_outputs, 0)
            x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
            return x
        else:
            raise ValueError(f"Unsupported VAE {self.vae_name}.")
    
    @torch.no_grad()
    def vae_decode(self, x):
        # input: (b, t, c, h, w)
        if not self.vae:
            return x
        elif self.vae_name == "oasis":
            batch_size, n_frames, c, h, w = x.shape
            x = rearrange(x, "b t c h w -> (b t) (h w) c")
            x = self.vae.decode(x / self.scaling_factor)
            x = rearrange(x, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames)
            return x
        elif self.vae_name == "sd_vae":
            batch_size, n_frames, c, h, w = x.shape
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.vae.decode(x / self.vae.config.scaling_factor).sample
            x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
            return x
        else:
            raise ValueError(f"Unsupported VAE {self.vae_name}.")

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
