"""
Clipped GRPO whose reward is **negative spatial distance** (world-eval VGGT + Chamfer),
matching ``fast_inference_spatial_distance.py`` / ``utils.spatial_distance``.

Requires ``cfg.grpo.reward: spatial_distance`` and ``cfg.grpo.spatial_distance`` (see
``config/algorithm/df_grpo_clipped_sd.yaml``). Ground-truth latents come from the batch
(``xs_latent[:n_gen]``); generated candidates are decoded and compared per (group, batch).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchvision.io import write_video

from train_oasis.trainer.DF_GRPO_Clipped_Trainer import DiffusionForcingGRPOClipped


def _train_oasis_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _world_eval_root() -> Path:
    return _train_oasis_repo_root().parent / "world-eval-latest"


class DiffusionForcingGRPOClippedSpatialDistance(DiffusionForcingGRPOClipped):
    """
    Same clipped policy loss as :class:`DiffusionForcingGRPOClipped`, but
    ``_compute_rewards`` uses spatial distance vs. ground-truth video from the batch
    instead of DROID on generated video alone.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_cfg: DictConfig,
        model_ckpt: str = None,
    ):
        if cfg.grpo.get("reward", "droid") != "spatial_distance":
            raise ValueError(
                "DiffusionForcingGRPOClippedSpatialDistance requires cfg.grpo.reward="
                "'spatial_distance' (use algorithm df_grpo_clipped_sd)."
            )
        super().__init__(cfg, model_cfg, model_ckpt)

        self._sd_cfg = cfg.grpo.spatial_distance
        wo = OmegaConf.select(cfg.grpo.spatial_distance, "world_eval_root", default=None)
        self._world_eval_root = (
            Path(str(wo)).expanduser().resolve()
            if wo is not None
            else _world_eval_root()
        )

        sd = self._sd_cfg
        self.sd_metric_key: str = str(OmegaConf.select(sd, "metric", default="mean"))
        if self.sd_metric_key not in ("mean", "max"):
            raise ValueError("grpo.spatial_distance.metric must be 'mean' or 'max'")
        self.sd_chunk_size: int = int(OmegaConf.select(sd, "chunk_size", default=4096))
        self.sd_video_fps: int = int(OmegaConf.select(sd, "video_fps", default=20))

        self._sd_reconstructor = None
        self._sd_imports_checked = False

        # DDIM transition log-prob ratio (DDPO/Dance-style).
        # eta must be > 0 to define a proper density; eta=0 is deterministic.
        self.ddim_logprob_eta: float = float(cfg.grpo.get("ddim_logprob_eta", 0.0))
        self.ddim_logprob_step: int = int(cfg.grpo.get("ddim_logprob_step", 1))
        if self.ddim_logprob_step <= 0:
            raise ValueError("cfg.grpo.ddim_logprob_step must be >= 1")

    def _ensure_spatial_distance_imports(self) -> None:
        if self._sd_imports_checked:
            return
        root = self._world_eval_root
        if not root.is_dir():
            raise ImportError(
                f"world-eval-latest not found at {root}. Place it next to the train-oasis "
                "repo (e.g. videogen/world-eval-latest) or set grpo.spatial_distance overrides."
            )
        sd_path = root / "utils" / "spatial_distance.py"
        if not sd_path.is_file():
            raise ImportError(f"Missing spatial distance module: {sd_path}")

        # world-eval must be on sys.path for VGGT imports inside spatial_distance.
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Do not use ``import utils.spatial_distance``: ``train_oasis/utils.py`` is often
        # loaded as top-level ``utils`` when cwd is train_oasis/train_oasis, which breaks
        # ``utils.spatial_distance`` (utils is a module, not a package).
        mod_name = "_world_eval_spatial_distance_ext"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, sd_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {sd_path}")
            sd_mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = sd_mod
            spec.loader.exec_module(sd_mod)
        except ImportError as e:
            raise ImportError(
                "Spatial distance reward failed to import. Ensure world-eval-latest is "
                "present and VGGT/OpenCV deps match utils.spatial_distance."
            ) from e

        self._ReconstructionConfig = sd_mod.ReconstructionConfig
        self._VGGTReconstructor = sd_mod.VGGTReconstructor
        self._sd_compute_pair = sd_mod.compute_spatial_distance
        self._sd_imports_checked = True

    def _ensure_sd_reconstructor(self) -> None:
        self._ensure_spatial_distance_imports()
        if self._sd_reconstructor is not None:
            return

        sd = self._sd_cfg
        dev = OmegaConf.select(sd, "device", default=None)
        if dev is None:
            dev = str(self.device)

        repo_dir = OmegaConf.select(sd, "vggt_repo_dir", default=None)
        model_path = OmegaConf.select(sd, "model_path", default=None)
        if repo_dir is None:
            repo_dir = self._world_eval_root / "vggt"
        else:
            repo_dir = Path(str(repo_dir)).expanduser().resolve()
        if model_path is None:
            model_path = self._world_eval_root / "vggt" / "model.pt"
        else:
            model_path = Path(str(model_path)).expanduser().resolve()

        RC = self._ReconstructionConfig
        self._sd_reconstructor = self._VGGTReconstructor(
            repo_dir=repo_dir,
            model_path=model_path,
            device=str(dev),
            config=RC(
                frame_stride=int(OmegaConf.select(sd, "frame_stride", default=1)),
                max_frames=OmegaConf.select(sd, "max_frames", default=None),
                vggt_batch_size=OmegaConf.select(sd, "vggt_batch_size", default=None),
                confidence_threshold=float(
                    OmegaConf.select(sd, "confidence_threshold", default=0.2)
                ),
                max_points_per_frame=int(
                    OmegaConf.select(sd, "max_points_per_frame", default=200000)
                ),
                max_merged_points=int(
                    OmegaConf.select(sd, "max_merged_points", default=200000)
                ),
                seed=int(OmegaConf.select(sd, "seed", default=0)),
            ),
        )

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

        Shapes:
          x_t, eps, prev_sample: (T, BG, ...)
          t, t_prev: (T, BG) integer timesteps in [0, self.timesteps)
        Returns:
          log_prob: (T, BG)
        """
        if eta <= 0:
            raise ValueError("eta must be > 0 to compute DDIM log-prob")
        # Gather alpha products.
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev]
        # Broadcast to sample shape.
        bc = (1,) * (x_t.ndim - 2)
        alpha_t_b = alpha_t.view(*alpha_t.shape, *bc)
        alpha_prev_b = alpha_prev.view(*alpha_prev.shape, *bc)

        # x0 from epsilon (works for both v-pred and x0-pred once eps is computed upstream).
        x0 = (x_t - (1.0 - alpha_t_b).sqrt() * eps) / alpha_t_b.sqrt()

        # DDIM variance (diffusers/DDIM paper).
        variance = ((1.0 - alpha_prev) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_prev)
        variance = torch.clamp(variance, min=1e-20)
        sigma = (eta * variance.sqrt()).view(*variance.shape, *bc)

        # mean = sqrt(alpha_prev) * x0 + sqrt(1 - alpha_prev - sigma^2) * eps
        dir_coeff = torch.clamp(1.0 - alpha_prev_b - sigma * sigma, min=0.0).sqrt()
        mean = alpha_prev_b.sqrt() * x0 + dir_coeff * eps

        # log N(prev_sample; mean, sigma^2 I)
        log_prob = (
            -((prev_sample.detach() - mean) ** 2) / (2.0 * (sigma * sigma))
            - torch.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
        reduce_dims = tuple(range(2, log_prob.ndim))
        return log_prob.mean(dim=reduce_dims)

    @torch.no_grad()
    def _compute_rewards(
        self,
        candidates: list[torch.Tensor],
        gt_latents: torch.Tensor,
        n_gen: int,
    ) -> torch.Tensor:
        """Return (G, B) rewards; higher is better (negative spatial distance)."""
        assert gt_latents.shape[0] == n_gen, (gt_latents.shape[0], n_gen)
        self._ensure_sd_reconstructor()

        group_size = len(candidates)
        batch_size = candidates[0].shape[1]
        rewards = torch.zeros(
            (group_size, batch_size), dtype=torch.float32, device=self.device
        )

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            gt_decoded = self._decode_for_scoring(gt_latents)

            gt_pcs: list = []
            for b in range(batch_size):
                tmp_gt = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_gt.close()
                tmp_gt_path = tmp_gt.name
                try:
                    vid = gt_decoded[:, b].permute(0, 2, 3, 1).clamp(0, 1)
                    write_video(
                        tmp_gt_path,
                        (vid * 255).byte().cpu().numpy(),
                        fps=self.sd_video_fps,
                    )
                    gt_pcs.append(
                        self._sd_reconstructor.reconstruct_video(tmp_gt_path)
                    )
                finally:
                    if os.path.isfile(tmp_gt_path):
                        os.unlink(tmp_gt_path)

            all_rewards: list[list[float]] = []
            for g in range(group_size):
                decoded = self._decode_for_scoring(candidates[g])
                batch_rewards: list[float] = []
                for b in range(batch_size):
                    tmp_gen = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    tmp_gen.close()
                    tmp_gen_path = tmp_gen.name
                    try:
                        vid = decoded[:, b].permute(0, 2, 3, 1).clamp(0, 1)
                        write_video(
                            tmp_gen_path,
                            (vid * 255).byte().cpu().numpy(),
                            fps=self.sd_video_fps,
                        )
                        gen_pc = self._sd_reconstructor.reconstruct_video(tmp_gen_path)
                        result = self._sd_compute_pair(
                            gen_pc,
                            gt_pcs[b],
                            chunk_size=self.sd_chunk_size,
                        )
                        raw = float(result.spatial_distance[self.sd_metric_key])
                        batch_rewards.append(-raw)
                    finally:
                        if os.path.isfile(tmp_gen_path):
                            os.unlink(tmp_gen_path)
                all_rewards.append(batch_rewards)

            rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

        if dist.is_initialized():
            dist.broadcast(rewards, src=0)

        return rewards

    def training_step(self, batch, batch_idx):
        """
        Same as DiffusionForcingGRPOClipped.training_step, but importance ratios
        use DDIM transition log-prob (Dance/DDPO-style) instead of the noise-space
        surrogate likelihood.
        """
        if torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", getattr(self, "local_rank", 0)))
            torch.cuda.set_device(lr)

        if self.ddim_logprob_eta <= 0:
            raise ValueError(
                "Spatial-distance GRPO with DDIM log-prob requires cfg.grpo.ddim_logprob_eta > 0 "
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
            t = torch.randint(0, self.timesteps, (n_loss, batch_size), device=self.device)
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
            external_cond=rearrange(cond_flat, "t bg d -> bg t d") if cond_flat is not None else None,
        )
        if not self.use_checkpoint_in_grpo_loss and policy_prev_ckpt:
            self.diffusion_model.gradient_checkpointing = policy_prev_ckpt
        pred_flat = rearrange(pred_flat, "bg t ... -> t bg ...")

        nan_count = torch.isnan(pred_flat).sum()
        if dist.is_initialized():
            dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)
        if nan_count > 0:
            self.global_nan_number += 1
            self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=True)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            var = self.logprob_var
            eps_theta = self._eps_pred_from_model_out(x_noisy_flat, t_flat, pred_flat)

            with torch.no_grad():
                old_prev = getattr(self.policy_old, "gradient_checkpointing", False)
                if not self.use_checkpoint_in_grpo_loss and old_prev:
                    self.policy_old.gradient_checkpointing = False
                pred_old = self.policy_old(
                    x=rearrange(x_noisy_flat, "t bg ... -> bg t ..."),
                    t=rearrange(t_flat, "t bg -> bg t"),
                    external_cond=rearrange(cond_flat, "t bg d -> bg t d") if cond_flat is not None else None,
                )
                if not self.use_checkpoint_in_grpo_loss and old_prev:
                    self.policy_old.gradient_checkpointing = old_prev
                pred_old = rearrange(pred_old, "bg t ... -> t bg ...")
                eps_old = self._eps_pred_from_model_out(x_noisy_flat, t_flat, pred_old)

                ref_prev_ckpt = getattr(self.ref_model, "gradient_checkpointing", False)
                if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                    self.ref_model.gradient_checkpointing = False
                ref_pred_flat = self.ref_model(
                    x=rearrange(x_noisy_flat, "t bg ... -> bg t ..."),
                    t=rearrange(t_flat, "t bg -> bg t"),
                    external_cond=rearrange(cond_flat, "t bg d -> bg t d") if cond_flat is not None else None,
                )
                if not self.use_checkpoint_in_grpo_loss and ref_prev_ckpt:
                    self.ref_model.gradient_checkpointing = ref_prev_ckpt
                ref_pred_flat = rearrange(ref_pred_flat, "bg t ... -> t bg ...")
                eps_ref = self._eps_pred_from_model_out(x_noisy_flat, t_flat, ref_pred_flat)

            self.log(
                "training/grpo_mean_abs_pred_diff_fp32",
                (pred_flat.detach().float() - pred_old.float()).abs().mean(),
                sync_dist=True,
            )

            # DDIM transition log-prob ratio.
            t_prev = torch.clamp(t_flat - self.ddim_logprob_step, min=0)
            # Sample x_{t_prev} using theta mean + shared noise so log probs are comparable.
            # (Matches the dance/ddpo idea: log p(prev_sample | mean(theta), sigma).)
            alpha_t = self.alphas_cumprod[t_flat]
            alpha_prev = self.alphas_cumprod[t_prev]
            variance = ((1.0 - alpha_prev) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_prev)
            variance = torch.clamp(variance, min=1e-20)
            bc = (1,) * (x_noisy_flat.ndim - 2)
            sigma = (self.ddim_logprob_eta * variance.sqrt()).view(*variance.shape, *bc)
            z = torch.randn_like(x_noisy_flat)

            # Build prev_sample from theta mean.
            alpha_t_b = alpha_t.view(*alpha_t.shape, *bc)
            alpha_prev_b = alpha_prev.view(*alpha_prev.shape, *bc)
            x0_theta = (x_noisy_flat - (1.0 - alpha_t_b).sqrt() * eps_theta) / alpha_t_b.sqrt()
            dir_coeff = torch.clamp(1.0 - alpha_prev_b - sigma * sigma, min=0.0).sqrt()
            mean_theta = alpha_prev_b.sqrt() * x0_theta + dir_coeff * eps_theta
            prev_sample = mean_theta + sigma * z

            logp_theta = self._ddim_transition_logprob(
                x_t=x_noisy_flat,
                eps=eps_theta,
                t=t_flat,
                t_prev=t_prev,
                eta=self.ddim_logprob_eta,
                prev_sample=prev_sample,
            )
            logp_old = self._ddim_transition_logprob(
                x_t=x_noisy_flat,
                eps=eps_old,
                t=t_flat,
                t_prev=t_prev,
                eta=self.ddim_logprob_eta,
                prev_sample=prev_sample,
            )
            log_ratio = torch.clamp(logp_theta - logp_old, -self.log_ratio_clamp, self.log_ratio_clamp)
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
            rho_c = torch.clamp(rho, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            clipped = rho_c * adv_flat
            surr_tbg = torch.minimum(unclipped, clipped)

            w_flat = self.compute_loss_weights(t_flat)
            w_s = rearrange(w_flat, "t (b g) -> t b g", b=batch_size, g=G)
            surr_s = rearrange(surr_tbg, "t (b g) -> t b g", b=batch_size, g=G)
            loss_policy = -(surr_s * w_s).mean()

            # Keep KL regularization in eps-space (same as base clipped trainer).
            kl_tbg = self._sum_sq_noise_error(eps_theta, eps_ref.detach()) / (2.0 * var)
            kl_s = rearrange(kl_tbg, "t (b g) -> t b g", b=batch_size, g=G)
            kl_bg = (kl_s * w_s).mean(dim=0)
            loss_kl = self.kl_coeff * kl_bg.mean()

            loss = loss_policy + loss_kl

            clip_frac = ((rho - rho_c).abs() > 1e-6).float().mean()
            self.log("training/grpo_clip_frac", clip_frac, sync_dist=True)
            self.log("training/grpo_mean_rho", rho.mean(), sync_dist=True)

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
