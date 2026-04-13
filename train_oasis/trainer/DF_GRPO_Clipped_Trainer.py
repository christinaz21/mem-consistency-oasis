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
import math
import os
from typing import Optional

import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.distributed as dist
from omegaconf import DictConfig
import numpy as np

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
        # Optionally switch rho from the original noise-space surrogate to a
        # DDIM transition log-prob (Dance/DDPO-style). Requires eta > 0.
        self.ddim_logprob_eta: float = float(grpo.get("ddim_logprob_eta", 0.0))
        self.ddim_logprob_step: int = int(grpo.get("ddim_logprob_step", 1))
        self.ddim_logprob_sigma_min: float = float(grpo.get("ddim_logprob_sigma_min", 1e-3))
        if self.ddim_logprob_step <= 0:
            raise ValueError("cfg.grpo.ddim_logprob_step must be >= 1")
        if self.ddim_logprob_sigma_min <= 0:
            raise ValueError("cfg.grpo.ddim_logprob_sigma_min must be > 0")

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



    @torch.no_grad()
    def _generate_video_with_logprobs(
        self,
        prompt_latents: torch.Tensor,
        conditions: Optional[torch.Tensor],
        n_target: int,
        seed: int,
    ):
        """
        Same rollout structure as _generate_video, but stores per-step transitions and old-policy log-probs.

        Returns:
            {
                "final":      (T, B, C, H, W),
                "x_t":        (S, W, B, C, H, W),
                "x_next":     (S, W, B, C, H, W),
                "from_nl":    (S, W, B),
                "to_nl":      (S, W, B),
                "logp_old":   (S, W, B),
                "active":     (S, W, B),
                "cond":       (S, W, B, D) or None,
            }
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        xs = prompt_latents.clone()
        curr = xs.shape[0]
        batch_size = xs.shape[1]

        if n_target <= curr:
            return {
                "final": xs[:n_target],
                "x_t": None,
                "x_next": None,
                "from_nl": None,
                "to_nl": None,
                "logp_old": None,
                "active": None,
                "cond": None,
            }

        step_x_t = []
        step_x_next = []
        step_from = []
        step_to = []
        step_logp = []
        step_active = []
        step_cond = []

        while curr < n_target:
            horizon = 1
            substeps = self.gen_substeps_per_frame

            sched_vals = np.linspace(
                self.sampling_timesteps, 0, num=substeps + 1, dtype=np.int64
            )
            sched = sched_vals[:, None]

            noise_chunk = torch.randn(
                (horizon, *xs.shape[1:]), device=self.device, dtype=xs.dtype
            )
            noise_chunk = torch.clamp(noise_chunk, -self.clip_noise, self.clip_noise)
            xs = torch.cat([xs, noise_chunk], dim=0)

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

                x_in = xs[start:].clone()
                cond_in = (
                    conditions[start : curr + horizon].clone()
                    if conditions is not None
                    else None
                )

                x_next, logp_old, active_mask, _ = self.sample_step_with_logprob(
                    x=x_in,
                    external_cond=cond_in,
                    curr_noise_level=from_nl[start:],
                    next_noise_level=to_nl[start:],
                    eta=self.ddim_logprob_eta,
                    sigma_min=self.ddim_logprob_sigma_min,
                    prev_sample=None,
                    model=self.policy_old,
                )

                step_x_t.append(x_in.detach())
                step_x_next.append(x_next.detach())
                step_from.append(from_nl[start:].detach())
                step_to.append(to_nl[start:].detach())
                step_logp.append(logp_old.detach())
                step_active.append(active_mask.detach())
                if cond_in is not None:
                    step_cond.append(cond_in.detach())

                xs[start:] = x_next

            curr += horizon

        rollout = {
            "final": xs[:n_target].detach(),
            "x_t": torch.cat(step_x_t, dim=0),        # (S, W, B, ...)
            "x_next": torch.cat(step_x_next, dim=0),  # (S, W, B, ...)
            "from_nl": torch.cat(step_from, dim=0),   # (S, W, B)
            "to_nl": torch.cat(step_to, dim=0),       # (S, W, B)
            "logp_old": torch.cat(step_logp, dim=0),  # (S, W, B)
            "active": torch.cat(step_active, dim=0),  # (S, W, B)
            "cond": torch.cat(step_cond, dim=0) if conditions is not None else None,
        }
        return rollout



    def _ddim_transition_logprob(
        self,
        *,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float,
        prev_sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log p(x_{t_prev} | x_t; eps_theta) under a DDIM Gaussian transition.

        This matches the helper used in the spatial-distance GRPO trainer.
        Returns log-prob averaged across all non-(t,batch) dimensions.
        """
        if eta <= 0:
            raise ValueError("eta must be > 0 to compute DDIM log-prob")

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev]

        bc = (1,) * (x_t.ndim - 2)
        alpha_t_b = alpha_t.view(*alpha_t.shape, *bc)
        alpha_prev_b = alpha_prev.view(*alpha_prev.shape, *bc)

        x0 = (x_t - (1.0 - alpha_t_b).sqrt() * eps) / alpha_t_b.sqrt()

        variance = ((1.0 - alpha_prev) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_prev)
        variance = torch.clamp(variance, min=1e-20)
        sigma = (eta * variance.sqrt()).view(*variance.shape, *bc)
        sigma = torch.clamp(sigma, min=self.ddim_logprob_sigma_min)

        dir_coeff = torch.clamp(1.0 - alpha_prev_b - sigma * sigma, min=0.0).sqrt()
        mean = alpha_prev_b.sqrt() * x0 + dir_coeff * eps

        log_prob = (
            -((prev_sample.detach() - mean) ** 2) / (2.0 * (sigma * sigma))
            - torch.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
        reduce_dims = tuple(range(2, log_prob.ndim))
        return log_prob.mean(dim=reduce_dims)


    def _chunked_transition_logprob(
        self,
        *,
        x_t_flat: torch.Tensor,
        x_next_flat: torch.Tensor,
        from_nl_flat: torch.Tensor,
        to_nl_flat: torch.Tensor,
        cond_flat: torch.Tensor | None,
        model,
        chunk_n: int,
    ):
        logp_chunks = []
        clipped_chunks = []

        # Match DF_GRPO.training_step: non-reentrant checkpoint + bf16 AMP under DDP
        # can mismatch forward vs recomputation when the policy forward is split into
        # many chunks. Disable DiT checkpointing for this loss path unless opted in.
        restore_ckpt = False
        prev_ckpt = getattr(model, "gradient_checkpointing", False)
        if (
            model is self.diffusion_model
            and not self.use_checkpoint_in_grpo_loss
            and prev_ckpt
        ):
            model.gradient_checkpointing = False
            restore_ckpt = True
        try:
            for start in range(0, x_t_flat.shape[0], chunk_n):
                end = min(start + chunk_n, x_t_flat.shape[0])

                _, logp_chunk, _, clipped_chunk = self.sample_step_with_logprob(
                    x=x_t_flat[start:end],
                    external_cond=cond_flat[start:end]
                    if cond_flat is not None
                    else None,
                    curr_noise_level=from_nl_flat[start:end],
                    next_noise_level=to_nl_flat[start:end],
                    eta=self.ddim_logprob_eta,
                    sigma_min=self.ddim_logprob_sigma_min,
                    prev_sample=x_next_flat[start:end],
                    model=model,
                )
                logp_chunks.append(logp_chunk)
                clipped_chunks.append(clipped_chunk)
        finally:
            if restore_ckpt:
                model.gradient_checkpointing = True

        return torch.cat(logp_chunks, dim=0), torch.cat(clipped_chunks, dim=0)





    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", getattr(self, "local_rank", 0)))
            torch.cuda.set_device(lr)

        if self.ddim_logprob_eta <= 0:
            raise ValueError(
                "DDIM-logprob GRPO requires cfg.grpo.ddim_logprob_eta > 0 "
                "(eta=0 is deterministic, log-prob undefined)."
            )

        if self.profile_timing:
            self._step_perf_t0 = self._sync_perf_counter()

        xs, conditions, masks = self._preprocess_batch(batch)
        xs_latent = self.vae_encode(xs)

        n_data = xs_latent.shape[0]
        n_gen = min(n_data, self.grpo_max_frames)
        batch_size = xs_latent.shape[1]

        prompt = xs_latent[: self.context_frames]
        cond_full = conditions[:n_gen] if conditions is not None else None

        # ------------------------------------------------------------
        # policy_old should be the behavior policy for this batch
        # ------------------------------------------------------------
        self._sync_policy_old_from_policy()

        # ------------------------------------------------------------
        # Phase 1: rollout generation with stored transition log-probs
        # ------------------------------------------------------------
        gen_t0 = self._sync_perf_counter() if self.profile_timing else 0.0
        self.diffusion_model.eval()

        rollouts: list[dict] = []
        candidates: list[torch.Tensor] = []

        for _ in range(self.group_size):
            seed = torch.randint(0, 2**31 - 1, (1,), device=self.device).item()
            ro = self._generate_video_with_logprobs(prompt, cond_full, n_gen, seed)
            # print(
            #     ro["x_t"].shape,
            #     ro["x_next"].shape,
            #     ro["from_nl"].shape,
            #     ro["logp_old"].shape,
            # )
                        
            rollouts.append(ro)
            candidates.append(ro["final"])

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

        # ------------------------------------------------------------
        # Phase 2: reward computation on final generated videos
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Phase 3: group-relative advantages
        # ------------------------------------------------------------
        r_mean = rewards.mean(dim=0, keepdim=True)
        r_std = rewards.std(dim=0, keepdim=True).clamp(min=1e-4)
        adv = (rewards - r_mean) / r_std  # (G, B)

        torch.set_grad_enabled(True)

        # ------------------------------------------------------------
        # Phase 4: exact GRPO loss on stored rollout transitions
        # ------------------------------------------------------------
        G = self.group_size
        B = batch_size

        if rollouts[0]["x_t"] is None:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            N = rollouts[0]["x_t"].shape[0]

            # rollout tensors:
            # x_t:      (S, W, B, C, H, W)
            # x_next:   (S, W, B, C, H, W)
            # from_nl:  (S, W, B)
            # to_nl:    (S, W, B)
            # logp_old: (S, W, B)
            # active:   (S, W, B)

            x_t_stack = torch.stack([ro["x_t"] for ro in rollouts], dim=2)           # (N, B, G, C, H, W)
            x_next_stack = torch.stack([ro["x_next"] for ro in rollouts], dim=2)     # (N, B, G, C, H, W)
            from_nl_stack = torch.stack([ro["from_nl"] for ro in rollouts], dim=2)   # (N, B, G)
            to_nl_stack = torch.stack([ro["to_nl"] for ro in rollouts], dim=2)       # (N, B, G)
            logp_old_stack = torch.stack([ro["logp_old"] for ro in rollouts], dim=2) # (N, B, G)
            active_stack = torch.stack([ro["active"] for ro in rollouts], dim=2)     # (N, B, G)

            cond_stack = None
            if rollouts[0]["cond"] is not None:
                cond_stack = torch.stack([ro["cond"] for ro in rollouts], dim=2)        # (S, W, B, G, D)

            # flatten (B, G) into batch axis; keep N as model time dimension
            x_t_flat = rearrange(x_t_stack, "n b g ... -> n (b g) ...")
            x_next_flat = rearrange(x_next_stack, "n b g ... -> n (b g) ...")
            from_nl_flat = rearrange(from_nl_stack, "n b g -> n (b g)")
            to_nl_flat = rearrange(to_nl_stack, "n b g -> n (b g)")
            logp_old_flat = rearrange(logp_old_stack, "n b g -> n (b g)")
            active_flat = rearrange(active_stack, "n b g -> n (b g)")

            # print(x_t_stack.shape)   # (N, B, G, 16, 18, 32)
            # print(x_t_flat.shape)    # (N, B*G, 16, 18, 32)

            cond_flat = None
            if cond_stack is not None:
                cond_flat = rearrange(cond_stack, "n b g d -> n (b g) d")

            max_train_transitions = int(getattr(self, "grpo_max_train_transitions", 256))

            if x_t_flat.shape[0] > max_train_transitions:
                idx = torch.randperm(x_t_flat.shape[0], device=x_t_flat.device)[:max_train_transitions]
                idx = idx.sort().values

                x_t_flat = x_t_flat[idx]
                x_next_flat = x_next_flat[idx]
                from_nl_flat = from_nl_flat[idx]
                to_nl_flat = to_nl_flat[idx]
                logp_old_flat = logp_old_flat[idx]
                active_flat = active_flat[idx]
                if cond_flat is not None:
                    cond_flat = cond_flat[idx]

            N = x_t_flat.shape[0]

            logprob_t0 = self._sync_perf_counter() if self.profile_timing else 0.0

            # ------------------------------------------------------------
            # Current policy log-prob on the exact stored transition
            # ------------------------------------------------------------
            chunk_n = int(getattr(self, "grpo_transition_chunk_size", 32))

            logp_theta_flat, clipped_curr_flat = self._chunked_transition_logprob(
                x_t_flat=x_t_flat,
                x_next_flat=x_next_flat,
                from_nl_flat=from_nl_flat,
                to_nl_flat=to_nl_flat,
                cond_flat=cond_flat,
                model=self.diffusion_model,
                chunk_n=chunk_n,
            )
            logp_theta = rearrange(logp_theta_flat, "n (b g) -> n b g", b=B, g=G)
            logp_old = rearrange(logp_old_flat, "n (b g) -> n b g", b=B, g=G)
            active = rearrange(active_flat, "n (b g) -> n b g", b=B, g=G).float()
            clipped_curr = rearrange(clipped_curr_flat, "n (b g) -> n b g", b=B, g=G)


            self.log("training/logp_old_mean", logp_old.mean(), sync_dist=True)
            self.log("training/logp_theta_mean", logp_theta.mean(), sync_dist=True)
            self.log("training/logp_diff_mean", (logp_theta - logp_old).mean(), sync_dist=True)
            self.log(
                "training/logp_diff_std_direct",
                (logp_theta - logp_old).std(unbiased=False),
                sync_dist=True,
            )


            nan_count = torch.isnan(logp_theta).sum()
            if dist.is_initialized():
                dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)

            if nan_count > 0:
                self.global_nan_number += 1
                self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=True)
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # ------------------------------------------------------------
                # PPO / GRPO ratio on exact rollout transitions
                # ------------------------------------------------------------
                log_ratio_raw = logp_theta - logp_old
                log_ratio = torch.clamp(
                    log_ratio_raw,
                    -self.log_ratio_clamp,
                    self.log_ratio_clamp,
                )
                rho = torch.exp(log_ratio)

                hit_hi = (log_ratio >= (self.log_ratio_clamp - 1e-6)).float().mean()
                hit_lo = (log_ratio <= (-self.log_ratio_clamp + 1e-6)).float().mean()
                self.log("training/grpo_log_ratio_clip_hi_frac", hit_hi, sync_dist=True)
                self.log("training/grpo_log_ratio_clip_lo_frac", hit_lo, sync_dist=True)
                self.log(
                    "training/grpo_log_ratio_std",
                    log_ratio_raw.detach().std(unbiased=False),
                    sync_dist=True,
                )
                self.log(
                    "training/grpo_mean_abs_log_ratio",
                    log_ratio.detach().abs().mean(),
                    sync_dist=True,
                )

                adv_bg = adv.transpose(0, 1)   # (B, G)
                adv_nbg = adv_bg.unsqueeze(0).expand(N, B, G)

                unclipped = rho * adv_nbg
                rho_c = torch.clamp(rho, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                clipped = rho_c * adv_nbg
                surr = torch.minimum(unclipped, clipped)

                w = self.compute_loss_weights(clipped_curr.view(N, B * G))
                w = w.view(N, B, G)

                denom = active.sum().clamp(min=1.0)
                loss_policy = -((surr * w * active).sum() / denom)

                # ------------------------------------------------------------
                # Reference KL on the same stored rollout states
                # ------------------------------------------------------------
                # with torch.no_grad():
                #     ref_prev_ckpt = getattr(self.ref_model, "gradient_checkpointing", False)
                #     if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                #         self.ref_model.gradient_checkpointing = False

                #     ref_pred = self._chunked_model_forward(
                #         x_t_flat=x_t_flat,
                #         t_flat=clipped_curr_flat,
                #         cond_flat=cond_flat,
                #         model=self.ref_model,
                #         chunk_n=chunk_n,
                #     )

                #     if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                #         self.ref_model.gradient_checkpointing = ref_prev_ckpt

                # policy_prev_ckpt = getattr(self.diffusion_model, "gradient_checkpointing", False)
                # if not self.use_checkpoint_in_grpo_loss and policy_prev_ckpt:
                #     self.diffusion_model.gradient_checkpointing = False

                # pred = self._chunked_model_forward(
                #     x_t_flat=x_t_flat,
                #     t_flat=clipped_curr_flat,
                #     cond_flat=cond_flat,
                #     model=self.diffusion_model,
                #     chunk_n=chunk_n,
                # )


                # if not self.use_checkpoint_in_grpo_loss and policy_prev_ckpt:
                #     self.diffusion_model.gradient_checkpointing = policy_prev_ckpt

                # eps_theta = self._eps_pred_from_model_out(x_t_flat, clipped_curr_flat, pred)
                # eps_ref = self._eps_pred_from_model_out(x_t_flat, clipped_curr_flat, ref_pred)

                # kl_t = self._sum_sq_noise_error(eps_theta, eps_ref.detach()) / (2.0 * self.logprob_var)
                # kl = rearrange(kl_t, "n (b g) -> n b g", b=B, g=G)

                # loss_kl = self.kl_coeff * ((kl * w * active).sum() / denom)
                # loss = loss_policy + loss_kl

                loss_kl = torch.zeros((), device=self.device, dtype=loss_policy.dtype)
                loss = loss_policy + loss_kl   

                clip_frac = ((rho - rho_c).abs() > 1e-6).float().mean()
                self.log("training/grpo_clip_frac", clip_frac, sync_dist=True)
                self.log("training/grpo_mean_rho", rho.mean(), sync_dist=True)
                self.log(
                    "training/reward_best_in_group",
                    rewards.max(dim=0).values.mean(),
                    sync_dist=True,
                )
                self.log(
                    "training/reward_worst_in_group",
                    rewards.min(dim=0).values.mean(),
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

        # ------------------------------------------------------------
        # Final logging
        # ------------------------------------------------------------
        self.log("training/grpo_loss", loss, sync_dist=True, prog_bar=True)
        self.log("training/mean_reward", rewards.mean(), sync_dist=True, prog_bar=True)
        self.log("training/reward_std", rewards.std(), sync_dist=True)
        self.log("training/max_reward", rewards.max(), sync_dist=True)
        self.log("training/min_reward", rewards.min(), sync_dist=True)
        self.log("training/mean_abs_advantage", adv.abs().mean(), sync_dist=True)
        self.log("training/mean_advantage_signed", adv.mean(), sync_dist=True)

        return {"loss": loss}