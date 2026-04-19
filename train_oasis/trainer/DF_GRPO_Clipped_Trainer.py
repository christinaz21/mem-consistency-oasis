"""
GRPO with clipped policy ratios for diffusion-forcing video.

Closer to the standard GRPO / PPO recipe than :class:`DiffusionForcingGRPO`:
group-relative advantages, importance ratios ``rho = pi_theta / pi_old`` with
PPO-style clipping, and KL regularization toward a frozen reference policy.

The policy density is a *surrogate* in noise space: for a forward noising step
``x_t = sqrt(alpha_t) x_0 + sqrt(1-alpha_t) eps`` with fixed ``eps``, we use
``log pi_theta propto -||eps - eps_theta(x_t,t)||^2 / (2 sigma^2)`` (constants
cancel in ``rho``). This is not the exact likelihood of DDIM-style ``sample_step``
rollouts; for that, use a stochastic reverse process and transition log-probs
(e.g. SDE/DDPM-style sampling).

References:
    - GRPO: https://arxiv.org/abs/2402.03300
"""
from __future__ import annotations

import copy
import os

import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.distributed as dist
from omegaconf import DictConfig

from train_oasis.trainer.DF_GRPO_Trainer import DiffusionForcingGRPO


class DiffusionForcingGRPOClipped(DiffusionForcingGRPO):
    """
    Same rollout + DROID scoring as :class:`DiffusionForcingGRPO`, but the policy
    update uses clipped GRPO-style surrogates: ``min(rho*A, clip(rho)*A)`` plus
    KL to ``ref_model``.

    ``policy_old`` holds the **pre-optimizer-step** weights: we copy
    ``diffusion_model.state_dict()`` into ``policy_old`` in
    :meth:`on_before_optimizer_step` (immediately before ``optimizer.step()``).
    After the step, ``diffusion_model`` is newer than ``policy_old``, so the
    ratio ``rho`` is not identically 1 (unlike syncing *after* the step).
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_cfg: DictConfig,
        model_ckpt: str = None,
    ):
        super().__init__(cfg, model_cfg, model_ckpt)
        grpo = cfg.grpo
        self.clip_epsilon: float = float(grpo.get("clip_epsilon", 0.2))
        self.logprob_var: float = float(grpo.get("logprob_var", 1.0))
        self.log_ratio_clamp: float = float(grpo.get("log_ratio_clamp", 20.0))

        self.policy_old = copy.deepcopy(self.diffusion_model)
        self.policy_old.requires_grad_(False)
        self.policy_old.eval()

    def _sync_policy_old_from_policy(self) -> None:
        self.policy_old.load_state_dict(self.diffusion_model.state_dict())
        self.policy_old.eval()

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int = 0) -> None:
        # Snapshot θ used as π_old on the *next* training batch (θ before this step).
        # Prefer this hook over overriding ``optimizer_step``: it always runs in the
        # trainer’s pre-step window (same timing as a leading sync in a custom
        # ``optimizer_step``), including with AMP / clipping / strategy quirks.
        self._sync_policy_old_from_policy()

    def _eps_pred_from_model_out(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> torch.Tensor:
        if self.predict_v:
            x0 = self.predict_start_from_v(x_noisy, t, model_pred)
        else:
            x0 = model_pred
        return self.predict_noise_from_start(x_noisy, t, x0)

    def _sum_sq_noise_error(
        self, noise_tgt: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        d = noise_tgt - eps_pred
        spatial_dims = tuple(range(2, d.ndim))
        return (d * d).sum(dim=spatial_dims)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
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

        reward_t0 = self._sync_perf_counter() if self.profile_timing else 0.0
        rewards = self._compute_rewards(candidates, xs_latent[:n_gen], n_gen)
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

        r_mean = rewards.mean(dim=0, keepdim=True)
        r_std = rewards.std(dim=0, keepdim=True).clamp(min=1e-4)
        adv = (rewards - r_mean) / r_std

        torch.set_grad_enabled(True)

        n_loss = min(self.n_frames, n_gen, self.grpo_loss_frames)
        cond_w = cond_full[:n_loss] if cond_full is not None else None
        G = self.group_size

        x_list = [candidates[g][:n_loss] for g in range(G)]
        x_stack = torch.stack(x_list, dim=2)

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
        t_stack = torch.stack(t_list, dim=2)

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
            var = self.logprob_var
            eps_theta = self._eps_pred_from_model_out(
                x_noisy_flat, t_flat, pred_flat
            )

            with torch.no_grad():
                old_prev = getattr(self.policy_old, "gradient_checkpointing", False)
                if not self.use_checkpoint_in_grpo_loss and old_prev:
                    self.policy_old.gradient_checkpointing = False
                pred_old = self.policy_old(
                    x=rearrange(x_noisy_flat, "t bg ... -> bg t ..."),
                    t=rearrange(t_flat, "t bg -> bg t"),
                    external_cond=(
                        rearrange(cond_flat, "t bg d -> bg t d")
                        if cond_flat is not None
                        else None
                    ),
                )
                if not self.use_checkpoint_in_grpo_loss and old_prev:
                    self.policy_old.gradient_checkpointing = old_prev
                pred_old = rearrange(pred_old, "bg t ... -> t bg ...")
                eps_old = self._eps_pred_from_model_out(
                    x_noisy_flat, t_flat, pred_old
                )

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
                eps_ref = self._eps_pred_from_model_out(
                    x_noisy_flat, t_flat, ref_pred_flat
                )

            # FP32 diagnostic: whether π and π_old differ in output space (bf16 forwards can hide this in rho).
            self.log(
                "training/grpo_mean_abs_pred_diff_fp32",
                (pred_flat.detach().float() - pred_old.float()).abs().mean(),
                sync_dist=True,
            )

            loglik_theta = -0.5 * self._sum_sq_noise_error(
                noise_flat, eps_theta
            ) / var
            loglik_old = -0.5 * self._sum_sq_noise_error(noise_flat, eps_old) / var
            log_ratio = torch.clamp(
                loglik_theta - loglik_old,
                -self.log_ratio_clamp,
                self.log_ratio_clamp,
            )
            rho = torch.exp(log_ratio)

            self.log(
                "training/grpo_mean_abs_log_ratio",
                log_ratio.detach().abs().mean(),
                sync_dist=True,
            )

            adv_bg = adv.transpose(0, 1)  # (B, G)
            adv_flat = rearrange(
                adv_bg.unsqueeze(0).expand(n_loss, -1, -1),
                "t b g -> t (b g)",
            )

            unclipped = rho * adv_flat
            rho_c = torch.clamp(
                rho, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
            clipped = rho_c * adv_flat
            surr_tbg = torch.minimum(unclipped, clipped)

            w_flat = self.compute_loss_weights(t_flat)
            w_s = rearrange(w_flat, "t (b g) -> t b g", b=batch_size, g=G)
            surr_s = rearrange(
                surr_tbg, "t (b g) -> t b g", b=batch_size, g=G
            )

            # Maximize clipped surrogate; minimize negative weighted mean.
            loss_policy = -(surr_s * w_s).mean()

            # KL(N(mu_theta, sigma^2 I) || N(mu_ref, sigma^2 I)) propto ||mu_theta - mu_ref||^2
            kl_tbg = (
                self._sum_sq_noise_error(eps_theta, eps_ref.detach()) / (2.0 * var)
            )
            kl_s = rearrange(kl_tbg, "t (b g) -> t b g", b=batch_size, g=G)
            kl_bg = (kl_s * w_s).mean(dim=0)
            loss_kl = self.kl_coeff * kl_bg.mean()

            loss = loss_policy + loss_kl

            clip_frac = ((rho - rho_c).abs() > 1e-6).float().mean()
            self.log("training/grpo_clip_frac", clip_frac, sync_dist=True)
            self.log(
                "training/grpo_mean_rho",
                rho.mean(),
                sync_dist=True,
            )

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

        self.log("training/grpo_loss", loss, sync_dist=True, prog_bar=True)
        self.log("training/mean_reward", rewards.mean(), sync_dist=True, prog_bar=True)
        self.log("training/reward_std", rewards.std(), sync_dist=True)
        self.log("training/max_reward", rewards.max(), sync_dist=True)
        self.log("training/min_reward", rewards.min(), sync_dist=True)
        self.log("training/mean_advantage", adv.abs().mean(), sync_dist=True)

        return {"loss": loss}
