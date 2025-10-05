from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
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
    Minimal per-agent PPO runtime.
    """

    def __init__(self, registry: AgentsRegistry, device: torch.device, obs_dim: int, act_dim: int):
        self.registry = registry
        self.device = device
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.T        = int(getattr(config, "PPO_WINDOW_TICKS", 64))
        self.epochs   = int(getattr(config, "PPO_EPOCHS", 2))
        self.lr       = float(getattr(config, "PPO_LR", 3e-4))
        self.clip     = float(getattr(config, "PPO_CLIP", 0.2))
        self.ent_coef = float(getattr(config, "PPO_ENTROPY_COEF", 0.01))
        self.vf_coef  = float(getattr(config, "PPO_VALUE_COEF", 0.5))
        self.gamma    = float(getattr(config, "PPO_GAMMA", 0.99))
        self.lam      = float(getattr(config, "PPO_LAMBDA", 0.95))
        self.T_max = int(getattr(config, "PPO_LR_T_MAX", 500_000))
        self.eta_min = float(getattr(config, "PPO_LR_ETA_MIN", 1e-6))

        self._buf: Dict[int, _Buf] = {}
        self._opt: Dict[int, optim.Optimizer] = {}
        self._sched: Dict[int, CosineAnnealingLR] = {}
        self._step = 0

    def _get_buf(self, aid: int) -> _Buf:
        if aid not in self._buf:
            self._buf[aid] = _Buf([], [], [], [], [], [])
        return self._buf[aid]

    def _get_opt(self, aid: int, model: nn.Module) -> optim.Optimizer:
        if aid not in self._opt:
            self._opt[aid] = optim.Adam(model.parameters(), lr=self.lr)
            self._sched[aid] = CosineAnnealingLR(self._opt[aid], T_max=self.T_max, eta_min=self.eta_min)
        return self._opt[aid]

    @torch.no_grad()
    def record_step(
        self,
        agent_ids: torch.Tensor,
        obs: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Append a single decision step for all agents in this tick.
        """
        logp_a = F.log_softmax(logits, dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)

        for i in range(agent_ids.numel()):
            aid = int(agent_ids[i].item())
            b = self._get_buf(aid)
            b.obs.append(obs[i])
            b.act.append(actions[i])
            b.logp.append(logp_a[i])
            b.val.append(values[i].reshape(1))
            b.rew.append(rewards[i])
            b.done.append(done[i])

        self._step += 1
        if self._step % self.T == 0:
            self._train_window_and_clear()

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = rewards.numel()
        adv = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t].item())
            next_val_t = values[t + 1] if t < T - 1 else 0.0
            
            delta = rewards[t] + self.gamma * next_val_t * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae
        
        ret = adv + values
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, ret

    def _policy_value(self, model: nn.Module, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = model(obs)
        values = values.squeeze(-1)
        logp = F.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return logits, values, entropy

    def _train_window_and_clear(self) -> None:
        for aid, b in list(self._buf.items()):
            if not b.obs: continue
            model = self.registry.brains[aid]
            if model is None:
                self._buf.pop(aid, None)
                continue

            model.train()
            opt = self._get_opt(aid, model)
            
            obs = torch.stack(b.obs)
            act = torch.stack(b.act).long()
            logp_old = torch.stack(b.logp)
            val_old = torch.cat(b.val)
            rew = torch.stack(b.rew)
            done = torch.stack(b.done).bool()

            adv, ret = self._gae(rew, val_old, done)

            with torch.enable_grad():
                for _ in range(self.epochs):
                    logits, values, entropy = self._policy_value(model, obs)
                    logp = F.log_softmax(logits, dim=-1).gather(1, act.view(-1,1)).squeeze(1)
                    ratio = torch.exp(logp - logp_old)
                    surr1, surr2 = ratio * adv, torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                    
                    loss_pi = -torch.min(surr1, surr2).mean()
                    loss_v = F.mse_loss(values, ret)
                    loss_ent = -entropy.mean()
                    loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            
            if aid in self._sched:
                self._sched[aid].step()

            b.obs.clear(); b.act.clear(); b.logp.clear(); b.val.clear(); b.rew.clear(); b.done.clear()# 