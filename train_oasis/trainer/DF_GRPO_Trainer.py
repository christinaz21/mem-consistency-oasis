"""
GRPO (Group Relative Policy Optimization) trainer for diffusion forcing video.

Adapts GRPO to diffusion models: generates multiple candidate videos per prompt,
scores them with a non-differentiable reward (DROID reprojection error), and
applies advantage-weighted diffusion loss with KL regularization against a
frozen reference model.

References:
    - GRPO: https://arxiv.org/abs/2402.03300
    - DDPO: https://arxiv.org/abs/2305.13301
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import copy
import sys
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.distributed as dist
from torchvision.io import write_video

from train_oasis.trainer.DF_Trainer import DiffusionForcingVideo
from train_oasis.utils import get_validation_metrics_for_videos, log_video

_INFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"
)
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)


class DiffusionForcingGRPO(DiffusionForcingVideo):
    """
    GRPO fine-tuning of a pre-trained diffusion-forcing video model.

    Each training step:
      1. Generates G candidate videos from the prompt + actions (no grad).
      2. Decodes and scores each candidate with DroidReprojectionScorer.
      3. Computes group-relative advantages: A_i = (r_i - mean) / std.
      4. Computes advantage-weighted diffusion loss with KL regularization
         against a frozen reference copy of the original model.

    Designed for DDP strategy. DeepSpeed stage-3 is not recommended because
    the frozen reference model is stored as a plain ``deepcopy``.
    A pre-trained checkpoint should be supplied via ``experiment.model_ckpt``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_cfg: DictConfig,
        model_ckpt: str = None,
    ):
        super().__init__(cfg, model_cfg, model_ckpt)

        grpo = cfg.grpo
        self.group_size: int = grpo.group_size
        self.kl_coeff: float = grpo.kl_coeff
        self.grpo_max_frames: int = grpo.get("max_gen_frames", self.n_frames)
        # Inference-speed knobs for GRPO rollout generation.
        self.gen_substeps_per_frame: int = int(grpo.get("gen_substeps_per_frame", 4))
        model_max_frames = int(getattr(self.diffusion_model, "max_frames", self.n_frames))
        self.gen_window_frames: int = int(grpo.get("gen_window_frames", model_max_frames))
        self.grpo_loss_frames: int = int(grpo.get("loss_frames", self.gen_window_frames))
        self.use_checkpoint_in_grpo_loss: bool = bool(
            grpo.get("use_checkpoint_in_grpo_loss", False)
        )
        self.gen_substeps_per_frame = max(1, self.gen_substeps_per_frame)
        self.gen_window_frames = max(1, self.gen_window_frames)
        self.grpo_loss_frames = max(1, self.grpo_loss_frames)
        self.profile_timing: bool = bool(grpo.get("profile_timing", False))
        self._backward_start_time: Optional[float] = None
        self._step_perf_t0: Optional[float] = None
        self._prof_gen_s: float = 0.0
        self._prof_reward_s: float = 0.0
        self._prof_logprob_s: float = 0.0

        self.ref_model = copy.deepcopy(self.diffusion_model)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        # Re-enable grad on the policy model after deepcopy
        self.diffusion_model.requires_grad_(True)

        # Scoring VAE: used to decode latents to pixels for DROID scoring.
        # Loaded independently so the trainer works with pre-encoded latent
        # datasets (vae_name=null) while still being able to score in pixel space.
        self.scoring_vae = None
        self.scoring_scaling_factor = cfg.scaling_factor
        sv = grpo.get("scoring_vae", None)
        if sv is not None and sv.get("ckpt", None) is not None:
            from train_oasis.model.vae import AutoencoderKL
            from safetensors.torch import load_model as load_safetensors

            self.scoring_vae = AutoencoderKL(
                latent_dim=sv.get("latent_dim", 16),
                patch_size=sv.get("patch_size", 20),
                enc_dim=sv.get("enc_dim", 1024),
                enc_depth=sv.get("enc_depth", 6),
                enc_heads=sv.get("enc_heads", 16),
                dec_dim=sv.get("dec_dim", 1024),
                dec_depth=sv.get("dec_depth", 12),
                dec_heads=sv.get("dec_heads", 16),
                input_height=sv.get("input_height", 360),
                input_width=sv.get("input_width", 640),
            )
            load_safetensors(self.scoring_vae, sv.ckpt)
            self.scoring_vae.eval()
            for p in self.scoring_vae.parameters():
                p.requires_grad = False

        self.reward_type: str = str(grpo.get("reward", "droid"))
        if self.reward_type == "droid":
            from worldscore import DroidReprojectionScorer

            d = grpo.droid
            self.scorer = DroidReprojectionScorer(
                weights_path=d.weights,
                calib=tuple(d.calib),
                stride=d.get("stride", 2),
                buffer=d.get("buffer", 512),
                filter_thresh=d.get("filter_thresh", 0.01),
                upsample=d.get("upsample", True),
                quiet=True,
                resize_long_side=d.get("resize", 256),
                max_frames=d.get("max_frames", 200),
            )
        elif self.reward_type == "spatial_distance":
            self.scorer = None
        else:
            raise ValueError(
                f"Unknown cfg.grpo.reward={self.reward_type!r} (expected 'droid' or 'spatial_distance')."
            )

    def _sync_perf_counter(self) -> float:
        """Return a wall-clock timestamp after synchronizing CUDA work."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _reduce_max_time(self, seconds: float) -> float:
        """Return max timing across ranks (or local value in single-process)."""
        t = torch.tensor(seconds, dtype=torch.float32, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def _prepare_scoring_vae_for_decode(self) -> None:
        """
        Lightning bf16-mixed can cast submodules to bf16; Oasis VAE decode can hit
        illegal CUDA accesses under bf16/AMP (worse under multi-GPU). Keep scoring
        weights in fp32 on the active device before decode.
        """
        if self.scoring_vae is None:
            return
        self.scoring_vae.eval()
        p0 = next(self.scoring_vae.parameters())
        if p0.device != self.device or p0.dtype != torch.float32:
            self.scoring_vae.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_video(
        self,
        prompt_latents: torch.Tensor,
        conditions: Optional[torch.Tensor],
        n_target: int,
        seed: int,
    ) -> torch.Tensor:
        """
        Autoregressively generate ``n_target`` latent frames (including the
        prompt).

        This uses the explicit DDIM loop structure used by
        `inference/fast_inference_droid.py` (noise-level matrix + sliding
        window) and **generates strictly frame-by-frame**: each iteration
        appends exactly one new latent frame and denoises it using a sliding
        window context of the last ``self.n_frames`` tokens.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Work in (B, T, ...) like the inference script.
        x = rearrange(prompt_latents.clone(), "t b ... -> b t ...").contiguous()
        B, T0 = x.shape[0], x.shape[1]
        if n_target <= T0:
            return rearrange(x[:, :n_target], "b t ... -> t b ...").contiguous()

        # Always frame-by-frame sliding-window generation, independent of
        # divisibility assumptions in inference-time noise ladder construction.
        xs = prompt_latents.clone()
        curr = xs.shape[0]
        batch_size = xs.shape[1]
        while curr < n_target:
            horizon = 1
            # Keep rollout cost aligned with fast_inference_droid:
            # use a small, explicit number of denoising substeps per new frame.
            substeps = self.gen_substeps_per_frame
            sched_vals = np.linspace(
                self.sampling_timesteps, 0, num=substeps + 1, dtype=np.int64
            )
            sched = sched_vals[:, None]  # (substeps + 1, 1)

            noise_chunk = torch.randn(
                (horizon, *xs.shape[1:]), device=self.device, dtype=xs.dtype
            )
            noise_chunk = torch.clamp(noise_chunk, -self.clip_noise, self.clip_noise)
            xs = torch.cat([xs, noise_chunk], dim=0)

            # Inference-style sliding window (typically model.max_frames, e.g. 10),
            # instead of full training n_frames (e.g. 100).
            start = max(0, curr + horizon - self.gen_window_frames)

            for m in range(sched.shape[0] - 1):
                from_nl = np.concatenate((np.zeros(curr, dtype=np.int64), sched[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_nl = np.concatenate((np.zeros(curr, dtype=np.int64), sched[m + 1]))[
                    :, None
                ].repeat(batch_size, axis=1)

                from_nl = torch.from_numpy(from_nl).to(self.device)
                to_nl = torch.from_numpy(to_nl).to(self.device)

                xs[start:] = self.sample_step(
                    xs[start:],
                    (conditions[start : curr + horizon] if conditions is not None else None),
                    from_nl[start:],
                    to_nl[start:],
                )

            curr += horizon

        return xs[:n_target]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_for_scoring(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixel-space [0, 1] for DROID scoring.

        Uses the dedicated scoring VAE if available (for pre-encoded latent
        datasets), otherwise falls back to the trainer's own VAE + unnormalize.
        """
        latents = latents.to(self.device, non_blocking=True)
        if self.scoring_vae is not None:
            self._prepare_scoring_vae_for_decode()
            T, B = latents.shape[:2]
            x = rearrange(latents, "t b c h w -> (t b) (h w) c")
            x = x.float()
            # Must run outside bf16 autocast; VAE expects fp32 activations.
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.scoring_vae.decode(x / self.scoring_scaling_factor)
            x = rearrange(x, "(t b) c h w -> t b c h w", t=T, b=B)
            x = x * 0.5 + 0.5
        else:
            x = self.vae_decode(latents)
            x = self._unnormalize_x(x)
        return torch.clamp(x, 0, 1)

    def _compute_rewards(
        self,
        candidates: list[torch.Tensor],
        gt_latents: torch.Tensor,
        n_gen: int,
    ) -> torch.Tensor:
        """Override in subclasses for non-DROID rewards. ``gt_latents``: (n_gen, B, ...)."""
        return self._score_videos(candidates)

    @torch.no_grad()
    def _score_videos(
        self, candidates: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode latent candidates and score each with DROID.

        Args:
            candidates: list of G tensors, each (T, B, C, H, W) in latent space.

        Returns:
            rewards: (G, B) float tensor (higher is better).
        """
        group_size = len(candidates)
        batch_size = candidates[0].shape[1]
        rewards = torch.zeros((group_size, batch_size), dtype=torch.float32, device=self.device)

        if self.scorer is None:
            raise RuntimeError(
                "DROID scorer is not initialized (cfg.grpo.reward != 'droid'). "
                "Override _compute_rewards or use a trainer that implements spatial-distance rewards."
            )
        # Decode + score only on rank 0 to avoid multi-rank VAE decode instability.
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            all_rewards: list[list[float]] = []
            for lat in candidates:
                decoded = self._decode_for_scoring(lat)

                batch_rewards: list[float] = []
                for b in range(decoded.shape[1]):
                    vid = decoded[:, b].permute(0, 2, 3, 1).cpu()
                    r = self.scorer.reward_from_video(vid)
                    batch_rewards.append(r)
                all_rewards.append(batch_rewards)
            rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

        if dist.is_initialized():
            dist.broadcast(rewards, src=0)

        return rewards

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # DDP/torchrun: pin each rank to its GPU (avoids all ranks defaulting to cuda:0).
        if torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", getattr(self, "local_rank", 0)))
            torch.cuda.set_device(lr)

        if self.profile_timing:
            self._step_perf_t0 = self._sync_perf_counter()

        xs, conditions, masks = self._preprocess_batch(batch)
        xs_latent = self.vae_encode(xs)

        n_data = xs_latent.shape[0]
        n_gen = min(n_data, self.grpo_max_frames)
        batch_size = xs_latent.shape[1]

        prompt = xs_latent[: self.context_frames]
        cond_full = conditions[:n_gen] if conditions is not None else None

        # ---- Phase 1: generate G candidate videos (no grad, eval) ----
        gen_t0 = self._sync_perf_counter() if self.profile_timing else 0.0
        self.diffusion_model.eval()
        candidates: list[torch.Tensor] = []
        for _ in range(self.group_size):
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
            gen = self._generate_video(prompt, cond_full, n_gen, seed)
            candidates.append(gen.detach())
        self.diffusion_model.train()
        if self.profile_timing:
            gen_dt = self._reduce_max_time(self._sync_perf_counter() - gen_t0)
            self._prof_gen_s = gen_dt
            self.log(
                "training/time_generation_s",
                gen_dt,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )

        # ---- Phase 2: score candidates ----
        reward_t0 = self._sync_perf_counter() if self.profile_timing else 0.0
        rewards = self._compute_rewards(candidates, xs_latent[:n_gen], n_gen)  # (G, B)
        if self.profile_timing:
            reward_dt = self._reduce_max_time(self._sync_perf_counter() - reward_t0)
            self._prof_reward_s = reward_dt
            self.log(
                "training/time_reward_s",
                reward_dt,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )

        # ---- Phase 3: group-relative advantages ----
        r_mean = rewards.mean(dim=0, keepdim=True)
        r_std = rewards.std(dim=0, keepdim=True).clamp(min=1e-4)
        adv = (rewards - r_mean) / r_std  # (G, B)

        # ---- Phase 4: advantage-weighted diffusion loss ----
        # Re-enable autograd: some reward code paths may leave grad disabled.
        torch.set_grad_enabled(True)

        # Keep GRPO loss horizon bounded (default: same as generation window).
        n_loss = min(self.n_frames, n_gen, self.grpo_loss_frames)
        cond_w = cond_full[:n_loss] if cond_full is not None else None

        # One batched forward over group_size (B*G effective batch) so DDP +
        # reentrant gradient checkpointing only runs a single backward through
        # diffusion_model per optimizer step. A loop of G separate forwards
        # triggers "Expected to mark a variable ready only once" under DDP.
        G = self.group_size
        x_list = [candidates[g][:n_loss] for g in range(G)]
        x_stack = torch.stack(x_list, dim=2)  # (n_loss, B, G, C, H, W)

        noise_list: list[torch.Tensor] = []
        t_list: list[torch.Tensor] = []
        for g in range(G):
            noise = torch.randn_like(x_list[g])
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
            t = torch.randint(
                0, self.timesteps, (n_loss, batch_size), device=self.device
            )
            noise_list.append(noise)
            t_list.append(t)
        noise_stack = torch.stack(noise_list, dim=2)
        t_stack = torch.stack(t_list, dim=2)  # (n_loss, B, G)

        x_flat = rearrange(x_stack, "t b g ... -> t (b g) ...")
        noise_flat = rearrange(noise_stack, "t b g ... -> t (b g) ...")
        t_flat = rearrange(t_stack, "t b g -> t (b g)")

        cond_flat = None
        if cond_w is not None:
            cond_flat = rearrange(
                cond_w.unsqueeze(2).expand(-1, -1, G, -1),
                "t b g d -> t (b g) d",
            )

        x_noisy_flat = self.q_sample(x_flat, t_flat, noise_flat)

        logprob_t0 = self._sync_perf_counter() if self.profile_timing else 0.0
        policy_prev_ckpt = getattr(self.diffusion_model, "gradient_checkpointing", False)
        if not self.use_checkpoint_in_grpo_loss and policy_prev_ckpt:
            self.diffusion_model.gradient_checkpointing = False
        pred_flat = self.diffusion_model(
            x=rearrange(x_noisy_flat, "t bg ... -> bg t ..."),
            t=rearrange(t_flat, "t bg -> bg t"),
            external_cond=(
                rearrange(cond_flat, "t bg d -> bg t d")
                if cond_flat is not None
                else None
            ),
        )
        if not self.use_checkpoint_in_grpo_loss and policy_prev_ckpt:
            self.diffusion_model.gradient_checkpointing = policy_prev_ckpt
        pred_flat = rearrange(pred_flat, "bg t ... -> t bg ...")

        nan_count = torch.isnan(pred_flat).sum()
        if dist.is_initialized():
            dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)
        if nan_count > 0:
            self.global_nan_number += 1
            self.log(
                "training/nan", self.global_nan_number, sync_dist=True, prog_bar=True
            )
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            if self.predict_v:
                target_flat = self.predict_v_from_x(x_flat, t_flat, noise_flat)
            else:
                target_flat = x_flat

            with torch.no_grad():
                ref_prev_ckpt = getattr(self.ref_model, "gradient_checkpointing", False)
                if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                    self.ref_model.gradient_checkpointing = False
                ref_pred_flat = self.ref_model(
                    x=rearrange(x_noisy_flat, "t bg ... -> bg t ..."),
                    t=rearrange(t_flat, "t bg -> bg t"),
                    external_cond=(
                        rearrange(cond_flat, "t bg d -> bg t d")
                        if cond_flat is not None
                        else None
                    ),
                )
                if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                    self.ref_model.gradient_checkpointing = ref_prev_ckpt
                ref_pred_flat = rearrange(ref_pred_flat, "bg t ... -> t bg ...")

            pred_s = rearrange(
                pred_flat, "t (b g) ... -> t b g ...", b=batch_size, g=G
            )
            target_s = rearrange(
                target_flat, "t (b g) ... -> t b g ...", b=batch_size, g=G
            )
            ref_s = rearrange(
                ref_pred_flat, "t (b g) ... -> t b g ...", b=batch_size, g=G
            )

            w_flat = self.compute_loss_weights(t_flat)
            w_s = rearrange(w_flat, "t (b g) -> t b g", b=batch_size, g=G)
            w_exp = w_s.view(
                n_loss, batch_size, G, *([1] * (pred_s.ndim - 3))
            )

            mse = F.mse_loss(pred_s, target_s.detach(), reduction="none") * w_exp
            kl = F.mse_loss(pred_s, ref_s.detach(), reduction="none") * w_exp
            reduce_dims = [0] + list(range(3, mse.ndim))
            mse_bg = mse.mean(dim=reduce_dims)  # (B, G)
            kl_bg = kl.mean(dim=reduce_dims)  # (B, G)

            adv_bg = adv.transpose(0, 1)  # (B, G); matches mse_bg
            loss = (adv_bg * mse_bg + self.kl_coeff * kl_bg).mean()

        if self.profile_timing:
            logprob_dt = self._reduce_max_time(self._sync_perf_counter() - logprob_t0)
            self._prof_logprob_s = logprob_dt
            self.log(
                "training/time_logprob_forward_s",
                logprob_dt,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )

        # ---- logging ----
        self.log("training/grpo_loss", loss, sync_dist=True, prog_bar=True)
        self.log("training/mean_reward", rewards.mean(), sync_dist=True, prog_bar=True)
        self.log("training/reward_std", rewards.std(), sync_dist=True)
        self.log("training/max_reward", rewards.max(), sync_dist=True)
        self.log("training/min_reward", rewards.min(), sync_dist=True)
        self.log("training/mean_advantage", adv.abs().mean(), sync_dist=True)

        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        super().on_before_backward(loss)
        if self.profile_timing:
            self._backward_start_time = self._sync_perf_counter()

    def on_after_backward(self) -> None:
        super().on_after_backward()
        backward_s = 0.0
        if self.profile_timing and self._backward_start_time is not None:
            backward_s = self._reduce_max_time(
                self._sync_perf_counter() - self._backward_start_time
            )
            self.log(
                "training/time_backward_s",
                backward_s,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self._backward_start_time = None

        # Wall time for preprocess + encode + generation + scoring + loss forward + backward
        # (does not include optimizer step; see Lightning batch order).
        if self.profile_timing and self._step_perf_t0 is not None:
            total_dt = self._reduce_max_time(
                self._sync_perf_counter() - self._step_perf_t0
            )
            self.log(
                "training/time_step_total_s",
                total_dt,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            accounted = (
                self._prof_gen_s
                + self._prof_reward_s
                + self._prof_logprob_s
                + backward_s
            )
            other = max(0.0, total_dt - accounted)
            self.log(
                "training/time_other_s",
                other,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )
            self._step_perf_t0 = None

            # self.log() does not print to SLURM unless the metric is on the progress bar;
            # emit one line so .log files always contain the breakdown.
            if self.global_rank == 0:
                print(
                    "[GRPO profile] "
                    f"step_total={total_dt:.2f}s | "
                    f"gen={self._prof_gen_s:.2f}s reward={self._prof_reward_s:.2f}s "
                    f"logprob_fwd={self._prof_logprob_s:.2f}s backward={backward_s:.2f}s "
                    f"other={other:.2f}s",
                    flush=True,
                )

    def _wandb_experiment(self):
        """Resolve W&B experiment API from a single logger or Lightning's logger collection."""
        log = self.logger
        if log is None:
            return None
        if hasattr(log, "experiment"):
            return log.experiment
        if hasattr(log, "loggers"):
            for lg in log.loggers:
                exp = getattr(lg, "experiment", None)
                if exp is not None and type(lg).__name__ == "WandbLogger":
                    return exp
        return None

    def _hydra_output_dir(self) -> Path:
        try:
            import hydra

            return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        except Exception:
            if self.trainer is not None and getattr(self.trainer, "log_dir", None):
                return Path(self.trainer.log_dir)
            return Path(".")

    def _resolve_val_dataloader(self):
        """First validation dataloader passed to ``Trainer.fit``, if any."""
        if self.trainer is None:
            return None
        vdl = self.trainer.val_dataloaders
        if vdl is None:
            return None
        if isinstance(vdl, (list, tuple)):
            return vdl[0] if len(vdl) > 0 else None
        return vdl

    def _grpo_rollout_autocast(self):
        """
        ``on_train_start`` (and some val paths) run outside Lightning's training AMP
        wrapper; DiT is bf16 under bf16-mixed and would otherwise mix dtypes in
        ``t_embedder`` (RuntimeError: mat1 bf16 vs mat2 float).
        """
        if self.trainer is None:
            return nullcontext()
        prec = str(getattr(self.trainer, "precision", None) or "")
        if "bf16" in prec.lower() and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def _grpo_export_val_pixel_videos(
        self,
        xs_pred: torch.Tensor,
        xs: torch.Tensor,
        *,
        log_step: int,
        namespace: str = "validation",
    ) -> Optional[dict]:
        """
        Write pred/gt MP4s under ``val_videos/step_{log_step:07d}``, optional W&B video,
        and return perceptual metrics (or None if inputs empty).
        ``xs_pred``, ``xs``: (T, B, C, H, W) on CPU, float in ``[0, 1]``.
        """
        grpo = self.cfg.grpo
        fps = int(grpo.get("val_video_fps", 20))

        out_root = self._hydra_output_dir() / "val_videos" / f"step_{log_step:07d}"
        out_root.mkdir(parents=True, exist_ok=True)

        _, B, _, _, _ = xs_pred.shape
        for b in range(B):
            vid = (xs_pred[:, b].permute(0, 2, 3, 1) * 255.0).byte().cpu().numpy()
            write_video(str(out_root / f"pred_b{b}.mp4"), vid, fps=fps)
            vid_gt = (xs[:, b].permute(0, 2, 3, 1) * 255.0).byte().cpu().numpy()
            write_video(str(out_root / f"gt_b{b}.mp4"), vid_gt, fps=fps)
        print(
            f"[GRPO validation] saved {B} pred/gt MP4 pair(s) under {out_root}",
            flush=True,
        )

        wb = self._wandb_experiment()
        if wb is not None:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else log_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=wb,
            )

        pred_m = xs_pred[self.context_frames :] * 2.0 - 1.0
        gt_m = xs[self.context_frames :] * 2.0 - 1.0
        return get_validation_metrics_for_videos(
            pred_m,
            gt_m,
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """
        Roll out the policy on the val batch, decode latents with ``scoring_vae``,
        and stash pixel tensors for :meth:`on_validation_epoch_end` (W&B + MP4).
        Non-global rank-0 processes no-op so we only pay decode / IO once.
        """
        grpo = self.cfg.grpo
        if not bool(grpo.get("val_save_video", True)):
            return torch.tensor(0.0, device=self.device)

        if self.trainer is None or not self.trainer.is_global_zero:
            return torch.tensor(0.0, device=self.device)

        if self.scoring_vae is None:
            if self.global_rank == 0:
                print(
                    "[GRPO validation] val_save_video is set but scoring_vae.ckpt is missing; "
                    "skipping val videos (latents cannot be decoded).",
                    flush=True,
                )
            return torch.tensor(0.0, device=self.device)

        xs, conditions, masks = self._preprocess_batch(batch)
        xs_latent = self.vae_encode(xs)

        n_data = xs_latent.shape[0]
        max_frames = int(grpo.get("val_video_max_frames", 48))
        n_gen = min(n_data, self.grpo_max_frames, max_frames)
        cond_full = conditions[:n_gen] if conditions is not None else None
        seed = int(grpo.get("val_video_seed", 0))

        self.diffusion_model.eval()
        with self._grpo_rollout_autocast():
            gen = self._generate_video(
                xs_latent[: self.context_frames], cond_full, n_gen, seed
            )
        self.diffusion_model.train()

        pred_pix = self._decode_for_scoring(gen).clamp(0, 1)
        gt_pix = self._decode_for_scoring(xs_latent[:n_gen]).clamp(0, 1)

        self.validation_step_outputs.append((pred_pix.cpu(), gt_pix.cpu()))
        return torch.tensor(0.0, device=self.device)

    def on_validation_epoch_end(self, namespace="validation") -> None:
        """
        Rank 0 writes MP4s / W&B video; all ranks ``log_dict`` the same scalars
        (broadcast from rank 0) so DDP does not block on uneven validation hooks.
        """
        metric_dict = None

        if self.trainer is not None and self.trainer.is_global_zero and self.validation_step_outputs:
            xs_pred = []
            xs = []
            for pred, gt in self.validation_step_outputs:
                xs_pred.append(pred)
                xs.append(gt)
            xs_pred = torch.cat(xs_pred, dim=1)
            xs = torch.cat(xs, dim=1)

            metric_dict = self._grpo_export_val_pixel_videos(
                xs_pred,
                xs,
                log_step=int(self.global_step),
                namespace=namespace,
            )

        self.validation_step_outputs.clear()

        if dist.is_initialized() and self.trainer is not None and self.trainer.world_size > 1:
            payload = [metric_dict]
            dist.broadcast_object_list(payload, src=0)
            metric_dict = payload[0]

        if metric_dict is not None:
            self.log_dict(
                {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

    @torch.no_grad()
    def on_train_start(self) -> None:
        """Save val rollout MP4s at global step 0 (before first training step)."""
        super().on_train_start()
        grpo = self.cfg.grpo
        if not bool(grpo.get("val_save_video", True)):
            return
        if not bool(grpo.get("val_save_at_step_zero", True)):
            return
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        if self.scoring_vae is None:
            print(
                "[GRPO validation] val_save_at_step_zero skipped: scoring_vae.ckpt missing.",
                flush=True,
            )
            return

        val_loader = self._resolve_val_dataloader()
        if val_loader is None:
            print(
                "[GRPO validation] val_save_at_step_zero skipped: no val_dataloader.",
                flush=True,
            )
            return

        try:
            batch = next(iter(val_loader))
        except StopIteration:
            print(
                "[GRPO validation] val_save_at_step_zero skipped: empty val loader.",
                flush=True,
            )
            return

        batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)

        xs, conditions, masks = self._preprocess_batch(batch)
        xs_latent = self.vae_encode(xs)

        n_data = xs_latent.shape[0]
        max_frames = int(grpo.get("val_video_max_frames", 48))
        n_gen = min(n_data, self.grpo_max_frames, max_frames)
        cond_full = conditions[:n_gen] if conditions is not None else None
        seed = int(grpo.get("val_video_seed", 0))

        self.diffusion_model.eval()
        with self._grpo_rollout_autocast():
            gen = self._generate_video(
                xs_latent[: self.context_frames], cond_full, n_gen, seed
            )
        self.diffusion_model.train()

        pred_pix = self._decode_for_scoring(gen).clamp(0, 1).cpu()
        gt_pix = self._decode_for_scoring(xs_latent[:n_gen]).clamp(0, 1).cpu()

        metric_dict = self._grpo_export_val_pixel_videos(
            pred_pix,
            gt_pix,
            log_step=0,
            namespace="validation",
        )
        if metric_dict is not None:
            # Lightning forbids on_step=True inside on_train_start (only epoch-style logging).
            self.log_dict(
                {f"validation/step0_{k}": v for k, v in metric_dict.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
        print("[GRPO validation] val_save_at_step_zero done (step_0000000).", flush=True)
