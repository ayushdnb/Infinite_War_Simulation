from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# --- Transformer Building Blocks ---

class CrossAttentionBlock(nn.Module):
    """
    A single block of Cross-Attention followed by a Feed-Forward network.
    Includes residual connections and layer normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(query, key_value, key_value, need_weights=False)
        x = self.norm1(query + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class SelfAttentionBlock(nn.Module):
    """
    A single block of Self-Attention followed by a Feed-Forward network.
    Includes residual connections and layer normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


# --- The Main Transformer Brain ---

class TransformerBrain(nn.Module):
    """
    A transformer-based brain that processes observations by treating raycasts
    as a sequence of tokens and enriching them with the agent's state via attention.
    """
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int = 32, mlp_hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)

        self.num_rays = 64
        self.ray_feat_dim = 8
        self.rich_feat_dim = self.obs_dim - (self.num_rays * self.ray_feat_dim)
        
        if self.rich_feat_dim <= 0:
            raise ValueError(f"obs_dim ({obs_dim}) is not large enough for {self.num_rays} rays.")

        self.ray_embed_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_embed_proj = nn.Linear(self.ray_feat_dim, self.embed_dim)

        self.rich_embed_norm = nn.LayerNorm(self.rich_feat_dim)
        self.rich_embed_proj = nn.Linear(self.rich_feat_dim, self.embed_dim)

        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_rays, self.embed_dim))

        self.cross_attention = CrossAttentionBlock(self.embed_dim)
        self.self_attention = SelfAttentionBlock(self.embed_dim)

        mlp_input_dim = self.embed_dim * 2
        self.fc_in = nn.Linear(mlp_input_dim, mlp_hidden)
        self.fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        self.critic = nn.Linear(mlp_hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # --- NEW: Specific, JIT-compatible helper methods ---
    # This replaces the generic '_norm_then_linear' which caused scripting errors.
    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """Embeds ray features using a JIT-safe sequence."""
        x_norm = self.ray_embed_norm(rays_raw.float())
        return self.ray_embed_proj(x_norm.to(dtype=self.ray_embed_proj.weight.dtype))

    def _embed_rich(self, rich_raw: torch.Tensor) -> torch.Tensor:
        """Embeds rich features using a JIT-safe sequence."""
        x_norm = self.rich_embed_norm(rich_raw.float())
        return self.rich_embed_proj(x_norm.to(dtype=self.rich_embed_proj.weight.dtype))
    # --- END NEW ---

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]

        rays_raw = obs[:, :self.num_rays * self.ray_feat_dim].view(B, self.num_rays, self.ray_feat_dim)
        rich_raw = obs[:, self.num_rays * self.ray_feat_dim:]

        # Use the new, specific helper methods
        ray_tokens = self._embed_rays(rays_raw)
        rich_token = self._embed_rich(rich_raw).unsqueeze(1)

        ray_tokens = ray_tokens + self.positional_encoding
        contextual_ray_tokens = self.cross_attention(query=ray_tokens, key_value=rich_token)
        processed_ray_tokens = self.self_attention(contextual_ray_tokens)

        pooled_ray_summary = processed_ray_tokens.mean(dim=1)
        mlp_input = torch.cat([pooled_ray_summary, rich_token.squeeze(1)], dim=-1)

        h = F.gelu(self.fc_in(mlp_input))
        h = F.gelu(self.fc1(h))

        logits = self.actor(h)
        value = self.critic(h)

        return logits, value

    def param_count(self) -> int:
        """Utility to count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def scripted_transformer_brain(obs_dim: int, act_dim: int) -> torch.jit.ScriptModule:
    """TorchScript brain for non-PPO runs."""
    model = TransformerBrain(obs_dim, act_dim)
    return torch.jit.script(model)