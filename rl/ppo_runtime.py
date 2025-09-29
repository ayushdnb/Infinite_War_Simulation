from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
# NEW: Import the learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from .. import config
from ..engine.agent_registry import AgentsRegistry


@dataclass
class _Buf:
    obs: List[torch.Tensor]
    act: List[torch.Tensor]
    logp: List[torch.Tensor]
    val: List[torch.Tensor]
    rew: List[torch.Tensor]
    done: List[torch.Tensor]


class PerAgentPPORuntime:
    """
    Minimal per-agent PPO runtime:
      • Tiny replay window per agent (T steps)
      • GAE + clipped PPO updates on each agent's own model
      • One optimizer and LR scheduler per agent (Adam + CosineAnnealingLR)
    """

    def __init__(self, registry: AgentsRegistry, device: torch.device, obs_dim: int, act_dim: int):
        self.registry = registry
        self.device = device
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Hyperparams (overridable in config)
        self.T        = int(getattr(config, "PPO_WINDOW_TICKS", 64))
        self.epochs   = int(getattr(config, "PPO_EPOCHS", 2))
        self.lr       = float(getattr(config, "PPO_LR", 3e-4))
        self.clip     = float(getattr(config, "PPO_CLIP", 0.2))
        self.ent_coef = float(getattr(config, "PPO_ENTROPY_COEF", 0.01))
        self.vf_coef  = float(getattr(config, "PPO_VALUE_COEF", 0.5))
        self.gamma    = float(getattr(config, "PPO_GAMMA", 0.99))
        self.lam      = float(getattr(config, "PPO_LAMBDA", 0.95))

        # --- NEW: Scheduler hyperparameters ---
        self.T_max = int(getattr(config, "PPO_LR_T_MAX", 500_000)) # Steps for one decay cycle
        self.eta_min = float(getattr(config, "PPO_LR_ETA_MIN", 1e-6)) # Minimum learning rate

        self._buf: Dict[int, _Buf] = {}
        self._opt: Dict[int, optim.Optimizer] = {}
        # --- NEW: Dictionary to hold schedulers for each agent ---
        self._sched: Dict[int, CosineAnnealingLR] = {}
        self._step = 0

    def _get_buf(self, aid: int) -> _Buf:
        if aid not in self._buf:
            self._buf[aid] = _Buf([], [], [], [], [], [])
        return self._buf[aid]

    def _get_opt(self, aid: int, model: nn.Module) -> optim.Optimizer:
        if aid not in self._opt:
            self._opt[aid] = optim.Adam(model.parameters(), lr=self.lr)
            # --- NEW: Create a scheduler whenever a new optimizer is created ---
            self._sched[aid] = CosineAnnealingLR(self._opt[aid], T_max=self.T_max, eta_min=self.eta_min)
        return self._opt[aid]

    @torch.no_grad()
    def record_step(
        self,
        agent_ids: torch.Tensor,
        team_ids: torch.Tensor,
        obs: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        team_reward_red: float,
        team_reward_blue: float,
        done: torch.Tensor,
    ) -> None:
        """
        Append a single decision step for all agents in this tick.
        Reward shaping: each agent gets its TEAM reward for this tick.
        """
        logp_all = F.log_softmax(logits, dim=-1)
        logp_a = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)

        r_red  = torch.full((agent_ids.numel(),), float(team_reward_red),  device=self.device, dtype=obs.dtype)
        r_blue = torch.full((agent_ids.numel(),), float(team_reward_blue), device=self.device, dtype=obs.dtype)
        rew = torch.where(team_ids == 2.0, r_red, r_blue)

        for i in range(agent_ids.numel()):
            aid = int(agent_ids[i].item())
            b = self._get_buf(aid)
            b.obs.append(obs[i].detach().clone())
            b.act.append(actions[i].detach().clone())
            b.logp.append(logp_a[i].detach().clone())
            vi = values if values.dim() == 0 else values[i]
            b.val.append(vi.detach().reshape(1).clone())
            b.rew.append(rew[i].detach().clone())
            b.done.append(done[i].detach().clone())

        self._step += 1
        if self._step % self.T == 0:
            self._train_window_and_clear()

    def _stack(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards, values, dones: shape (T,)
        returns: (advantages, returns)
        """
        T = rewards.numel()
        adv = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t].item())
            if t == T - 1:
                next_val_t = next_value # Bootstrap with 0 at the end of the window
            else:
                next_val_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val_t * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae
        
        ret = adv + values

        if adv.numel() > 1:
            std = adv.std(unbiased=False)
            if float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
        return adv, ret

    def _policy_value(self, model: nn.Module, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = model(obs)
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        logp = F.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return logits, values, entropy

    def _ensure_trainable(self, model: nn.Module) -> None:
        for p in model.parameters():
            p.requires_grad_(True)

    def _train_window_and_clear(self) -> None:
        for aid, b in list(self._buf.items()):
            if len(b.obs) < 1:
                continue
            model = self.registry.brains[aid]
            if model is None:
                self._buf.pop(aid, None)
                continue

            dtype = next(model.parameters()).dtype
            self._ensure_trainable(model)
            model.train()
            opt = self._get_opt(aid, model)

            obs      = torch.stack(b.obs, dim=0).to(self.device, dtype=dtype)
            act      = torch.stack(b.act, dim=0).to(self.device).long()
            logp_old = torch.stack(b.logp, dim=0).to(self.device, dtype=dtype)
            val_old  = self._stack(b.val).to(self.device, dtype=dtype).view(-1)
            rew      = torch.stack(b.rew, dim=0).to(self.device, dtype=dtype)
            done     = torch.stack(b.done, dim=0).to(self.device).bool()

            adv, ret = self._gae(rew, val_old, done)

            with torch.enable_grad():
                for _ in range(self.epochs):
                    logits, values, entropy = self._policy_value(model, obs)
                    logp = F.log_softmax(logits, dim=-1).gather(1, act.view(-1,1)).squeeze(1)
                    ratio = torch.exp(logp - logp_old)

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                    loss_pi = -torch.min(surr1, surr2).mean()

                    loss_v = F.mse_loss(values.view(-1), ret)
                    loss_ent = -entropy.mean()
                    loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            
            # --- NEW: Step the scheduler for this agent after its update window ---
            if aid in self._sched:
                self._sched[aid].step()

            self._buf[aid] = _Buf([], [], [], [], [], [])