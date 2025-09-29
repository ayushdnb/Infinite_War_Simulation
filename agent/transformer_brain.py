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
        # Simple feed-forward part
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): The sequence to be updated (e.g., ray tokens). Shape: (B, T_q, D)
            key_value (torch.Tensor): The context sequence (e.g., rich feature token). Shape: (B, T_kv, D)
        Returns:
            torch.Tensor: The updated query sequence. Shape: (B, T_q, D)
        """
        # --- Cross-Attention ---
        # Query: ray tokens, Key/Value: rich feature token
        attn_output, _ = self.attn(query, key_value, key_value, need_weights=False)
        # Residual connection and normalization
        x = self.norm1(query + attn_output)

        # --- Feed-Forward ---
        ffn_output = self.ffn(x)
        # Residual connection and normalization
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
        # Simple feed-forward part
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input sequence. Shape: (B, T, D)
        Returns:
            torch.Tensor: The output sequence. Shape: (B, T, D)
        """
        # --- Self-Attention ---
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        # Residual connection and normalization
        x = self.norm1(x + attn_output)

        # --- Feed-Forward ---
        ffn_output = self.ffn(x)
        # Residual connection and normalization
        x = self.norm2(x + ffn_output)
        return x


# --- The Main Transformer Brain ---

class TransformerBrain(nn.Module):
    """
    A transformer-based brain that processes observations by treating raycasts
    as a sequence of tokens and enriching them with the agent's state via attention.
    Architecture:
    1.  Embeds 64 rays (8 features each) and rich features (21) into 32-dim tokens.
    2.  Adds learnable positional encodings to the ray tokens.
    3.  Uses Cross-Attention to inject rich feature context into each ray token.
    4.  Uses Self-Attention to allow ray tokens to communicate and reason spatially.
    5.  Pools the processed tokens into a fixed-size vector.
    6.  Feeds the vector into a standard MLP head to produce policy and value outputs.
    """
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int = 32, mlp_hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)

        # --- Input Processing ---
        self.num_rays = 64
        self.ray_feat_dim = 8
        self.rich_feat_dim = self.obs_dim - (self.num_rays * self.ray_feat_dim)
        
        if self.rich_feat_dim <= 0:
            raise ValueError(f"obs_dim ({obs_dim}) is not large enough for {self.num_rays} rays of {self.ray_feat_dim} features.")

        # Embedding layers
        self.ray_embed_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_embed_proj = nn.Linear(self.ray_feat_dim, self.embed_dim)

        self.rich_embed_norm = nn.LayerNorm(self.rich_feat_dim)
        self.rich_embed_proj = nn.Linear(self.rich_feat_dim, self.embed_dim)

        # Learnable positional encoding for the 64 rays
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_rays, self.embed_dim))

        # --- Transformer Blocks ---
        self.cross_attention = CrossAttentionBlock(self.embed_dim)
        self.self_attention = SelfAttentionBlock(self.embed_dim)

        # --- MLP Head ---
        # Input to MLP is 2 pooled tokens (mean-pooled rays + rich context)
        mlp_input_dim = self.embed_dim * 2
        self.fc_in = nn.Linear(mlp_input_dim, mlp_hidden)
        self.fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        self.critic = nn.Linear(mlp_hidden, 1)

        self.init_weights()

    def _norm_then_linear(self, x: torch.Tensor, norm: nn.LayerNorm, linear: nn.Linear) -> torch.Tensor:
        """
        Run LayerNorm in fp32 (stable), then cast to linear's dtype before the matmul.
        Works correctly under AMP.
        """
        # Always run LN in fp32 for numeric stability
        with torch.cuda.amp.autocast(enabled=False):
            x32 = norm(x.float())
        # Cast to the Linear weight dtype (Half when AMP kicks in)
        return linear(x32.to(dtype=linear.weight.dtype, device=x32.device))

    def init_weights(self):
        """Apply Xavier initialization to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the observation tensor and returns action logits and state value.
        Args:
            obs (torch.Tensor): Shape (B, obs_dim). Assumes first 512 features are rays.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, value)
        """
        B = obs.shape[0]

        # 1. Separate and embed inputs
        rays_raw = obs[:, :self.num_rays * self.ray_feat_dim].view(B, self.num_rays, self.ray_feat_dim)
        rich_raw = obs[:, self.num_rays * self.ray_feat_dim:]

        # Normalize and project using the AMP-safe helper
        ray_tokens = self._norm_then_linear(rays_raw, self.ray_embed_norm, self.ray_embed_proj)
        rich_token = self._norm_then_linear(rich_raw, self.rich_embed_norm, self.rich_embed_proj).unsqueeze(1)

        # 2. Add positional encoding
        ray_tokens = ray_tokens + self.positional_encoding

        # 3. Inject context with Cross-Attention
        # Rays "query" the rich features for context
        contextual_ray_tokens = self.cross_attention(query=ray_tokens, key_value=rich_token)

        # 4. Reason spatially with Self-Attention
        processed_ray_tokens = self.self_attention(contextual_ray_tokens)

        # 5. Pool tokens for the MLP head
        # Mean pooling provides a summary of the environment
        pooled_ray_summary = processed_ray_tokens.mean(dim=1) # (B, D)
        
        # Concatenate with the rich feature token to retain direct self-state info
        mlp_input = torch.cat([pooled_ray_summary, rich_token.squeeze(1)], dim=-1) # (B, D * 2)

        # 6. MLP Head
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