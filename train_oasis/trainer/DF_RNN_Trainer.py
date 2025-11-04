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
import time
import os

class DiffusionForcingRNNVideo(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, model_ckpt: str = None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.x_shape = cfg.x_shape
        self.vae_name = cfg.vae_name
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

        # rnn only
        self.clean_frame_noise_range = cfg.diffusion.clean_frame_noise_range
        self.training_chunk_num = cfg.training_chunk_num
        self.model_inner_window_size = model_cfg.inner_window_size
        self.select_start_frame = cfg.get("select_start_frame", 0)
        self.auxiliary_loss_coeff = cfg.get("auxiliary_loss_coeff", 0.0)

        self.lstm_mini_batch_size = cfg.lstm_mini_batch_size
        self.training_mini_batch_size = cfg.training_mini_batch_size
        
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
        from train_oasis.model.rnn_dit import DiT
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
        
        if model_ckpt:
            print(f"Loading Diffusion model from {model_ckpt}")
            sft_rnn = self.model_cfg.get("sft_rnn", False)
            strict_load = True if not sft_rnn else False
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
            else:
                state_dict = torch.load(model_ckpt, map_location="cpu")
                self.diffusion_model.load_state_dict(state_dict, strict=strict_load)
            if sft_rnn and False:
                trainable_param_names = ["rnn", "r_norm", "r_adaLN_modulation", "combine_action_proj", "external_cond"]
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

    def concat_hidden_states(self, hidden_states_list, batch_size, C, H, W):
        """
        Concatenate hidden states from a list of hidden states.
        LSTM:
            hidden_states_list: (num_frames, layer_num, ), each element is (h, c) for LSTM or h for GRU.
                h: (l, BHW, D)
            return: (layer_num, (BHW, T, D)) for LSTM.
        Mamba2:
            pass
        TTT:
            hidden_states_list: (num_frames, layer_num, ), each element is dict() for TTT.
                dict(): {
                    "W1_states": (BHW, D, D, D),
                    "b1_states": (BHW, D, 1, D),
                    "W1_grad": (BHW, D, D, D),
                    "b1_grad": (BHW, D, 1, D),
                }
            return: (layer_num, (BHW, T, D)) for TTT.
        """
        if self.model_cfg.rnn_config.rnn_type == "LSTM":
            return [
                (
                    rearrange(torch.cat([
                        rearrange(hidden_states_list[frame][layer][0], "l (b h w) d -> l b h w d", b=batch_size, h=H // self.model_cfg.patch_size, w=W // self.model_cfg.patch_size, l=self.model_cfg.rnn_config.num_layers)
                        for frame in range(len(hidden_states_list))
                    ], dim=1), "l b h w d -> l (b h w) d", b=batch_size * len(hidden_states_list), h=H // self.model_cfg.patch_size, w=W // self.model_cfg.patch_size, l=self.model_cfg.rnn_config.num_layers),
                    rearrange(torch.cat([
                        rearrange(hidden_states_list[frame][layer][1], "l (b h w) d -> l b h w d", b=batch_size, h=H // self.model_cfg.patch_size, w=W // self.model_cfg.patch_size, l=self.model_cfg.rnn_config.num_layers)
                        for frame in range(len(hidden_states_list))
                    ], dim=1), "l b h w d -> l (b h w) d", b=batch_size * len(hidden_states_list), h=H // self.model_cfg.patch_size, w=W // self.model_cfg.patch_size, l=self.model_cfg.rnn_config.num_layers),
                )
                for layer in range(len(hidden_states_list[0]))
            ]
        elif self.model_cfg.rnn_config.rnn_type == "Mamba2":
            return [
                (
                    torch.cat([
                        hidden_states_list[frame][layer][0] for frame in range(len(hidden_states_list))
                    ], dim=0),
                    torch.cat([
                        hidden_states_list[frame][layer][1] for frame in range(len(hidden_states_list))
                    ], dim=0),
                ) for layer in range(len(hidden_states_list[0]))
            ]
        elif self.model_cfg.rnn_config.rnn_type == "TTT":
            assert hidden_states_list[0][0]["W1_states"].shape[0] == (H // self.model_cfg.patch_size) * (W // self.model_cfg.patch_size) * batch_size, \
                f"Expected hidden state shape {(H // self.model_cfg.patch_size) * (W // self.model_cfg.patch_size) * batch_size, self.model_cfg.rnn_config.hidden_size}, but got {hidden_states_list[0][0]['W1_states'].shape}"
            return [
                {
                    "W1_states": torch.cat([
                        hidden_states_list[frame][layer]["W1_states"] for frame in range(len(hidden_states_list))
                    ], dim=0), # (T * BHW, D, D, D)
                    "b1_states": torch.cat([hidden_states_list[frame][layer]["b1_states"] for frame in range(len(hidden_states_list))], dim=0), # (T * BHW, D, 1, D)
                    "W1_grad": torch.cat([hidden_states_list[frame][layer]["W1_grad"] for frame in range(len(hidden_states_list))], dim=0), # (T * BHW, D, D, D)
                    "b1_grad": torch.cat([hidden_states_list[frame][layer]["b1_grad"] for frame in range(len(hidden_states_list))], dim=0), # (T * BHW, D, 1, D)
                }
                for layer in range(len(hidden_states_list[0]))
            ]
        else:
            raise NotImplementedError(f"RNN type {self.model_cfg.rnn_config.rnn_type} not implemented.")


    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        DEBUG = False
        if self.global_rank == 0 and DEBUG:
            print("Start training step")
            start_time = time.time()
        xs, conditions, masks = self._preprocess_batch(batch)
        xs_gt = xs.clone()
        xs = self.vae_encode(xs)
        if self.global_rank == 0 and DEBUG:
            print(f"Encoded VAE in {time.time() - start_time:.2f} seconds")
            start_time = time.time()
        num_frames, batch_size, C, H, W = xs.shape
        noise = torch.randn_like(xs)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noise_levels = self._generate_noise_levels(xs, masks)
        noised_x = self.q_sample(x_start=xs, t=noise_levels, noise=noise)

        # get clean frame
        with torch.no_grad():
            clean_noise_levels = torch.randint(self.clean_frame_noise_range[0], self.clean_frame_noise_range[1], (num_frames, batch_size), device=xs.device)
            clean_noise = torch.clamp(torch.randn_like(xs), -self.clip_noise, self.clip_noise)
            clean_noised_x = self.q_sample(x_start=xs, t=clean_noise_levels, noise=clean_noise)
            if self.auxiliary_loss_coeff > 0.0:
                if self.predict_v:
                    auxiliary_target = self.predict_v_from_x(xs, clean_noise_levels, clean_noise)[:num_frames - self.model_inner_window_size]
                else:
                    auxiliary_target = xs[:num_frames - self.model_inner_window_size]
                auxiliary_loss_weights = self.compute_loss_weights(clean_noise_levels[:num_frames - self.model_inner_window_size])
            clean_noise_levels = torch.full_like(clean_noise_levels, self.stabilization_level - 1, device=xs.device)

        if self.global_rank == 0 and DEBUG:
            print(f"Generated noise in {time.time() - start_time:.2f} seconds")
            start_time = time.time()
        
        # get all hidden states
        all_selected_indices = torch.randperm(num_frames - self.model_inner_window_size - self.select_start_frame, device=noised_x.device)[:self.training_chunk_num] + 1 + self.select_start_frame # do not select the first frame
        # sort indices in ascending order
        all_selected_indices, _ = torch.sort(all_selected_indices)
        all_hidden_states = self.diffusion_model.window_size_1_forward(
            x=rearrange(clean_noised_x[:num_frames - self.model_inner_window_size], "t b ... -> b t ..."),
            t=rearrange(clean_noise_levels[:num_frames - self.model_inner_window_size], "t b -> b t"),
            external_cond=rearrange(conditions[:num_frames - self.model_inner_window_size], "t b ... -> b t ...") if conditions is not None else None,
            hidden_states=None,
            mini_batch_size=self.lstm_mini_batch_size,
            target_hidden_states=all_selected_indices,
            get_return=(self.auxiliary_loss_coeff > 0.0),
        ) # (t - window_size + 1, )
        if self.auxiliary_loss_coeff > 0.0:
            auxiliary_outputs, all_hidden_states = all_hidden_states

        if self.global_rank == 0 and DEBUG:
            print(f"Got hidden states in {time.time() - start_time:.2f} seconds")
            start_time = time.time()

        all_loss = []
        nan_number = torch.tensor(0, dtype=torch.float, device=self.device)

        for i in range(0, self.training_chunk_num, self.training_mini_batch_size):
            selected_indices = all_selected_indices[i : i + self.training_mini_batch_size] # (chunk_num, )
            selected_hidden_states = all_hidden_states[i : i + self.training_mini_batch_size] # (chunk_num, layer_num, )
            batched_hidden_states = self.concat_hidden_states(selected_hidden_states, batch_size, C, H, W)

            batched_noised_x = torch.cat([
                noised_x[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) # (t, b * chunk_num, c, h, w)
            batched_noise_levels = torch.cat([
                noise_levels[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) # (t, b * chunk_num)
            batched_conditions = torch.cat([
                conditions[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) if conditions is not None else None
            batched_masks = torch.cat([
                masks[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) if masks is not None else None
            batched_targets = torch.cat([
                xs[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) # (t, b * chunk_num, c, h, w)
            batched_noise = torch.cat([
                noise[idx : idx + self.model_inner_window_size] for idx in selected_indices
            ], dim=1) # (t, b * chunk_num, c, h, w)

            model_pred, _ = self.diffusion_model(
                x=rearrange(batched_noised_x, "t b ... -> b t ..."),
                t=rearrange(batched_noise_levels, "t b -> b t"),
                external_cond=rearrange(batched_conditions, "t b ... -> b t ...") if conditions is not None else None,
                hidden_states=batched_hidden_states,
            )
            model_pred = rearrange(model_pred, "b t ... -> t b ...")
            nan_number += torch.sum(torch.isnan(model_pred))
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
                    target = self.predict_v_from_x(batched_targets, batched_noise_levels, batched_noise)
                else:
                    target = batched_targets
                loss = F.mse_loss(model_pred, target.detach(), reduction="none")
                loss_weight = self.compute_loss_weights(batched_noise_levels)
                loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
                loss = loss * loss_weight
                loss = self.reweight_loss(loss, batched_masks)
                all_loss.append(loss)
        loss = torch.stack(all_loss).mean()
        if self.auxiliary_loss_coeff > 0.0:
            auxiliary_outputs = rearrange(auxiliary_outputs, "b t ... -> t b ...")
            auxiliary_target = auxiliary_target[:auxiliary_outputs.shape[0]]
            auxiliary_loss_weights = auxiliary_loss_weights[:auxiliary_outputs.shape[0]]
            auxiliary_loss = F.mse_loss(
                auxiliary_outputs,
                auxiliary_target.detach(),
                reduction="none",
            )
            auxiliary_loss_weights = auxiliary_loss_weights.view(*auxiliary_loss_weights.shape, *((1,) * (auxiliary_loss.ndim - 2)))
            auxiliary_loss = auxiliary_loss * auxiliary_loss_weights
            auxiliary_loss = self.reweight_loss(auxiliary_loss, masks[:auxiliary_outputs.shape[0]])
            loss = loss + self.auxiliary_loss_coeff * auxiliary_loss

        if self.global_rank == 0 and DEBUG:
            print(f"Got loss in {time.time() - start_time:.2f} seconds")
        
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
        output_dict = self.training_step(batch, batch_idx)
        loss = output_dict["loss"]
        self.log(
            f"{namespace}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        return loss

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

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
            x = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
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
