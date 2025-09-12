# final_war_sim/rl/ppo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config

# ----------------------------- helpers -----------------------------
def _cfg(name: str, default):
    return getattr(config, name, default)

PPO_WINDOW_TICKS     = _cfg("PPO_WINDOW_TICKS", 20)     # optimize every 20 ticks
PPO_GAMMA            = _cfg("PPO_GAMMA", 0.99)
PPO_LAMBDA           = _cfg("PPO_LAMBDA", 0.95)         # GAE
PPO_LR               = _cfg("PPO_LR", 3e-4)
PPO_CLIP             = _cfg("PPO_CLIP", 0.2)
PPO_EPOCHS           = _cfg("PPO_EPOCHS", 3)
PPO_ENTROPY_BONUS    = _cfg("PPO_ENTROPY_BONUS", 0.01)
PPO_VALUE_COEF       = _cfg("PPO_VALUE_COEF", 0.5)
PPO_MAX_GRAD_NORM    = _cfg("PPO_MAX_GRAD_NORM", 1.0)
PPO_MAX_AGENTS_PER_WINDOW = _cfg("PPO_MAX_AGENTS_PER_WINDOW", 1400)  # cap per-window updates
AMP_ENABLED          = _cfg("amp_enabled", lambda: False)()  # keep your existing amp flag

TeamName = str  # "red" or "blue"

@dataclass
class StepStore:
    # Flat stores over the window (we’ll filter by agent_id when training)
    agent_id: List[int]
    obs: List[torch.Tensor]
    act: List[torch.Tensor]
    logp: List[torch.Tensor]
    val: List[torch.Tensor]
    rew: List[Optional[float]]  # per-step team reward; can be None -> filled later

    def __init__(self) -> None:
        self.agent_id = []; self.obs = []; self.act = []; self.logp = []; self.val = []; self.rew = []

    def append(self, agent_ids: torch.Tensor, obs: torch.Tensor,
               actions: torch.Tensor, logp: torch.Tensor, values: torch.Tensor,
               reward: Optional[float] = None) -> None:
        # All tensors are assumed already on the right device/dtype
        B = actions.shape[0]
        assert obs.shape[0] == B and logp.shape[0] == B and values.shape[0] == B and agent_ids.shape[0] == B
        self.agent_id += list(map(int, agent_ids.tolist()))
        self.obs.append(obs.detach())
        self.act.append(actions.detach())
        self.logp.append(logp.detach())
        self.val.append(values.detach())
        self.rew += [reward] * B

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[float]], torch.Tensor]:
        if len(self.agent_id) == 0:
            return (torch.empty(0),)*5 + (torch.empty(0),)
        agent_id = torch.tensor(self.agent_id, dtype=torch.long, device=self.obs[0].device)
        obs  = torch.cat(self.obs, dim=0)
        act  = torch.cat(self.act, dim=0)
        logp = torch.cat(self.logp, dim=0)
        val  = torch.cat(self.val, dim=0)
        return agent_id, obs, act, logp, self.rew, val

    def clear(self) -> None:
        self.__init__()

@dataclass
class WindowBuf:
    red: StepStore
    blue: StepStore
    def __init__(self) -> None:
        self.red = StepStore(); self.blue = StepStore()

# -------------------------------------------------------------------
# Per-Agent PPO with GAE (CTDE optional via per-step team rewards)
# -------------------------------------------------------------------
class PerAgentPPO:
    """
    Trains many tiny per-agent brains with PPO every N ticks without exploding compute.

    Usage pattern:
      ppo = PerAgentPPO(registry)
      ppo.begin_window(stats.snapshot())
      -- each tick --
      ppo.record_step(team, agent_ids, obs, logits, values, actions, reward=None)
      if (tick+1) % PPO_WINDOW_TICKS == 0:
          logs = ppo.end_window_and_train(stats)

    Notes:
      • reward=None -> at window end we will equally spread the team delta across steps (back-compat).
      • If you can supply per-step team reward 'reward=float' each tick, GAE becomes much better.
      • Only up to PPO_MAX_AGENTS_PER_WINDOW distinct agents are updated each window (sampled).
    """
    def __init__(self, registry) -> None:
        self.registry = registry
        self.device = registry.device
        self.buf = WindowBuf()
        self._snap = None  # stats snapshot at window start
        self._optim: Dict[int, torch.optim.Optimizer] = {}  # per-agent optimizer cache

    # ------------- window control -------------
    def begin_window(self, stats_snapshot) -> None:
        self._snap = stats_snapshot
        self.buf.red.clear(); self.buf.blue.clear()

    def end_window_and_train(self, stats) -> Dict[str, float]:
        if self._snap is None:
            return {}
        logs: Dict[str, float] = {}

        # Prepare both teams
        for team_name, store in (("red", self.buf.red), ("blue", self.buf.blue)):
            aid, obs, act, logp_old, rew_list, vpred = store.finalize()
            if obs.numel() == 0:
                logs[f"{team_name}_N"] = 0
                continue

            # Fill missing rewards: equal-split of team delta (keeps old behavior if you didn't pass per-step rewards)
            if any(r is None for r in rew_list):
                delta = stats.delta_since(self._snap)[team_name]  # float
                fill_val = float(delta) / max(1, len(rew_list))
                rew = torch.tensor([fill_val if r is None else r for r in rew_list],
                                   device=self.device, dtype=obs.dtype)
            else:
                rew = torch.tensor(rew_list, device=self.device, dtype=obs.dtype)

            # GAE advantages (sequence is flat-in-time across many agents; still effective as a baseline)
            with torch.no_grad():
                # We need V_{t+1}. Approx by shifting and padding last with V_T (common trick for truncated windows).
                v_next = torch.cat([vpred[1:], vpred[-1:].clone()], dim=0)
                delta = rew + PPO_GAMMA * v_next - vpred
                adv = torch.zeros_like(delta)
                gae = 0.0
                for t in reversed(range(delta.shape[0])):
                    gae = float(delta[t].item()) + PPO_GAMMA * PPO_LAMBDA * gae
                    adv[t] = gae
                ret = adv + vpred

            # Standardize advantages
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # Pick a subset of agents to actually update this window
            unique_agents = torch.unique(aid).tolist()
            random.shuffle(unique_agents)
            target_agents = unique_agents[:min(len(unique_agents), PPO_MAX_AGENTS_PER_WINDOW)]

            # Train each selected agent on its own data (keeps per-agent tiny brains)
            for agent_id in target_agents:
                mask = (aid == agent_id)
                if int(mask.sum().item()) == 0:
                    continue
                o_i  = obs[mask]
                a_i  = act[mask]
                lp_i = logp_old[mask]
                adv_i= adv[mask]
                ret_i= ret[mask]

                model = self.registry.brains[agent_id]
                if model is None:
                    continue

                opt = self._optim.get(agent_id, None)
                if opt is None:
                    opt = torch.optim.AdamW(model.parameters(), lr=PPO_LR)
                    self._optim[agent_id] = opt

                for _ in range(PPO_EPOCHS):
                    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                        logits, value = model(o_i)
                        dist = torch.distributions.Categorical(logits=logits)
                        logp = dist.log_prob(a_i)
                        ratio = torch.exp(logp - lp_i)

                        clip = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
                        loss_pi = -(torch.min(ratio * adv_i, clip * adv_i)).mean() \
                                  - PPO_ENTROPY_BONUS * dist.entropy().mean()
                        loss_v  = F.mse_loss(value.squeeze(-1), ret_i)
                        loss    = loss_pi + PPO_VALUE_COEF * loss_v

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
                    opt.step()

                # lightweight logs
                logs[f"a{agent_id}_N"]  = int(o_i.size(0))
                logs[f"a{agent_id}_Lpi"] = float(loss_pi.item())
                logs[f"a{agent_id}_Lv"]  = float(loss_v.item())

            # team-level logs
            logs[f"{team_name}_N"] = int(obs.size(0))

        self._snap = None
        # clear for next window
        self.buf.red.clear(); self.buf.blue.clear()
        return logs

    # ------------- data collection -------------
    @torch.no_grad()
    def record_step(self, team: TeamName, agent_ids: torch.Tensor,
                    obs: torch.Tensor,
                    logits: Optional[torch.Tensor],
                    values: torch.Tensor,
                    actions: torch.Tensor,
                    logp: Optional[torch.Tensor] = None,
                    reward: Optional[float] = None) -> None:
        """
        Append a batch of transitions for a single tick.

        Args:
          team: "red" or "blue"
          agent_ids: (B,) long tensor with registry indices
          obs: (B,F) observations used by their own brains
          logits: (B,A) action logits from each agent's brain on obs
          values: (B,) value predictions from each agent's brain on obs
          actions: (B,) chosen actions
          logp: optional (B,) log-prob; if None, computed from logits+actions
          reward: optional float per-step team reward (better). If None, we will
                  fill using team delta at window end (back-compat).
        """
        if logp is None:
            if logits is None:
                raise ValueError("Either logits or logp must be provided to record_step().")
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)

        store = self.buf.red if team == "red" else self.buf.blue
        store.append(agent_ids=agent_ids, obs=obs, actions=actions, logp=logp, values=values, reward=reward)
