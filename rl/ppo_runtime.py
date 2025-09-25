from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn, optim

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
      • One optimizer per agent (Adam)
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

        self._buf: Dict[int, _Buf] = {}
        self._opt: Dict[int, optim.Optimizer] = {}
        self._step = 0

    def _get_buf(self, aid: int) -> _Buf:
        if aid not in self._buf:
            self._buf[aid] = _Buf([], [], [], [], [], [])
        return self._buf[aid]

    def _get_opt(self, aid: int, model: nn.Module) -> optim.Optimizer:
        if aid not in self._opt:
            self._opt[aid] = optim.Adam(model.parameters(), lr=self.lr)
        return self._opt[aid]

    @torch.no_grad()
    def record_step(
        self,
        agent_ids: torch.Tensor,    # (K,)
        team_ids: torch.Tensor,     # (K,) float: 2.0 red / 3.0 blue
        obs: torch.Tensor,          # (K,D)
        logits: torch.Tensor,       # (K,A) (post-mask, used to sample; may be fp16 under AMP)
        values: torch.Tensor,       # (K,) or scalar if K==1 (may be fp16 under AMP)
        actions: torch.Tensor,      # (K,)
        team_reward_red: float,
        team_reward_blue: float,
        done: torch.Tensor,         # (K,) bool
    ) -> None:
        """
        Append a single decision step for all agents in this tick.
        Reward shaping: each agent gets its TEAM reward for this tick.
        """
        # logp of chosen actions (same dtype as logits; no grad)
        logp_all = F.log_softmax(logits, dim=-1)                      # (K,A)
        logp_a = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)   # (K,)

        # map team -> reward (match obs dtype for consistency; will be cast later)
        r_red  = torch.full((agent_ids.numel(),), float(team_reward_red),  device=self.device, dtype=obs.dtype)
        r_blue = torch.full((agent_ids.numel(),), float(team_reward_blue), device=self.device, dtype=obs.dtype)
        rew = torch.where(team_ids == 2.0, r_red, r_blue)  # (K,)

        # store per-agent (force scalars -> shape (1,), keep dtypes as-is; we'll unify later)
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
        # tensors are 1-D for val; use cat to avoid extra dims
        return torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
    import torch
    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        rewards, values, dones: shape (T,)
        returns: (advantages, returns)
        """
        T = rewards.numel()
        adv = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0  # bootstrap 0 at window boundary

        if T <= 1:
            mask = 1.0 - float(dones[0].item())
            delta = rewards[0] + self.gamma * next_value * mask - values[0]
            adv[0] = delta
            ret = adv + values
            return adv, ret

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t].item())
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae
            next_value = values[t].item()

        ret = adv + values

        # normalize advantages per window if var>0
        if adv.numel() > 1:
            std = adv.std(unbiased=False)
            if float(std.item()) > 0.0:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
        return adv, ret

    def _policy_value(self, model: nn.Module, obs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        logits, values = model(obs)                         # logits: (T,A), values: (T,) or (T,1)
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        logp = F.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return logits, values, entropy

    def _ensure_trainable(self, model: nn.Module) -> None:
        # Works for eager modules and ScriptModules (per-parameter toggle)
        for p in model.parameters():
            p.requires_grad_(True)

    def _train_window_and_clear(self) -> None:
        # Train each agent independently on its own buffer
        for aid, b in list(self._buf.items()):
            if len(b.obs) == 0:
                continue
            model = self.registry.brains[aid]
            if model is None:
                self._buf[aid] = _Buf([], [], [], [], [], [])
                continue

            # unify dtype to model param dtype (usually fp32) to avoid fp16/fp32 mixing
            dtype = next(model.parameters()).dtype

            self._ensure_trainable(model)
            model.train()
            opt = self._get_opt(aid, model)

            obs   = torch.stack(b.obs, dim=0).to(self.device, dtype=dtype)      # (T,D)
            act   = torch.stack(b.act, dim=0).to(self.device).long()            # (T,)
            logp_old = torch.stack(b.logp, dim=0).to(self.device, dtype=dtype)  # (T,)
            val_old  = self._stack(b.val).to(self.device, dtype=dtype).view(-1) # (T,)
            rew      = torch.stack(b.rew, dim=0).to(self.device, dtype=dtype)   # (T,)
            done     = torch.stack(b.done, dim=0).to(self.device).bool()        # (T,)

            adv, ret = self._gae(rew, val_old, done)                            # (T,), (T,)

            with torch.enable_grad():
                for _ in range(self.epochs):
                    logits, values, entropy = self._policy_value(model, obs)    # logits/values in 'dtype'
                    logp = F.log_softmax(logits, dim=-1).gather(1, act.view(-1,1)).squeeze(1)  # (T,)
                    ratio = torch.exp(logp - logp_old)

                    # policy loss (clipped surrogate)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                    loss_pi = -torch.min(surr1, surr2).mean()

                    # value loss (match dtypes)
                    loss_v = F.mse_loss(values.view(-1), ret)

                    # entropy (encourage exploration)
                    loss_ent = -entropy.mean()

                    loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            # clear buffer
            self._buf[aid] = _Buf([], [], [], [], [], [])

        # keep step counter rolling (no reset)
