from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class _SelfAttnBlock(nn.Module):
    """Pre-LN style transformer block: Self-Attention + Feed-Forward Network, with residual connections.
    
    Architecture:
        x -> MultiHeadAttention -> LayerNorm(x + attn) -> FFN -> LayerNorm(x + ffn) -> output
    """
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        """Initialize self-attention block.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            ffn_mult: Multiplier for FFN hidden dimension (default: 4)
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention + FFN with pre-LN residuals.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        
        # FFN with residual
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class _CrossAttnBlock(nn.Module):
    """Cross-Attention block: Query attends to Key-Value pairs, with FFN and residuals.
    
    Architecture:
        q, kv -> CrossAttn -> LayerNorm(q + attn) -> FFN -> LayerNorm(q + ffn) -> output
    """
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        """Initialize cross-attention block.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            ffn_mult: Multiplier for FFN hidden dimension (default: 4)
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention where q attends to kv, followed by FFN.
        
        Args:
            q: Query tensor of shape (batch_size, q_seq_len, d_model)
            kv: Key/Value tensor of shape (batch_size, kv_seq_len, d_model)
            
        Returns:
            Output tensor of same shape as q
        """
        # Cross-attention with residual
        a, _ = self.attn(q, kv, kv, need_weights=False)
        q = self.norm1(q + a)
        
        # FFN with residual
        f = self.ffn(q)
        q = self.norm2(q + f)
        return q


class TronBrain(nn.Module):
    """
    TRON v1 Transformer-based brain for per-agent control.
    
    Architecture has 4 stages:
      1. Ray encoder: Self-attention over ray tokens to extract spatial features
      2. Semantic encoder: Self-attention over semantic tokens (own, world, zone, team, combat, instinct)
      3. Fusion: Semantic+decision tokens cross-attend to processed ray tokens
      4. Readout: Decision tokens only -> logits and value
    
    Forward contract (MUST stay): forward(obs) -> (logits, value)
    
    Observation layout (from config):
      [rays_flat (num_rays * ray_feat_dim) | rich_base(23) | instinct(4)]
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        """Initialize TronBrain with observation and action dimensions.
        
        Args:
            obs_dim: Total observation dimension (must match config.OBS_DIM)
            act_dim: Action dimension (number of discrete actions)
            
        Raises:
            ValueError: If obs_dim doesn't match expected layout or config invalid
            RuntimeError: If semantic indices missing from config
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # --- Observation layout invariants (must match Phase 1-3 pipeline) ---
        # These constants are loaded from config and define the observation structure
        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT", 32))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM", 8))
        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 23))
        self.instinct_dim = int(getattr(config, "INSTINCT_DIM", 4))

        # Compute expected dimensions for validation
        self.rays_flat_dim = self.num_rays * self.ray_feat_dim
        self.expected_obs_dim = self.rays_flat_dim + self.rich_base_dim + self.instinct_dim

        # Validate observation dimension against config layout
        if self.obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"TronBrain obs_dim mismatch: got obs_dim={self.obs_dim}, "
                f"expected {self.expected_obs_dim} = ({self.num_rays}*{self.ray_feat_dim})"
                f"+{self.rich_base_dim}+{self.instinct_dim}."
            )

        # Semantic partition indices must exist (Phase 3 output)
        # These indices define how to slice rich_base into semantic tokens
        idx_map = getattr(config, "SEMANTIC_RICH_BASE_INDICES", None)
        if idx_map is None:
            raise RuntimeError(
                "SEMANTIC_RICH_BASE_INDICES missing in config. "
                "Apply Phase 3 (semantic partitioning) before using TronBrain."
            )
        self._idx_map: Dict[str, List[int]] = dict(idx_map)

        # --- TRON hyperparams (config-driven, safe defaults) ---
        d_model = int(getattr(config, "TRON_D_MODEL", 64))
        n_heads = int(getattr(config, "TRON_HEADS", 4))
        ray_layers = int(getattr(config, "TRON_RAY_LAYERS", 4))
        sem_layers = int(getattr(config, "TRON_SEM_LAYERS", 2))
        fusion_layers = int(getattr(config, "TRON_FUSION_LAYERS", 2))
        mlp_hidden = int(getattr(config, "TRON_MLP_HIDDEN", 128))

        # Validate transformer configuration
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("Invalid TRON config: TRON_D_MODEL and TRON_HEADS must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(f"Invalid TRON config: TRON_D_MODEL ({d_model}) must be divisible by TRON_HEADS ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads

        # --- Ray token embedding (8 features -> d_model) + learned direction embedding ---
        # Each ray has 8 features, projected to d_model
        self.ray_in_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_in_proj = nn.Linear(self.ray_feat_dim, d_model)
        # Learned positional embedding for ray directions (1, num_rays, d_model)
        self.ray_dir_embed = nn.Parameter(torch.randn(1, self.num_rays, d_model) * 0.02)

        # --- Semantic token embeddings (variable input dims -> d_model) ---
        # Expected semantic groups from Phase 3:
        #   own_context, world_context, zone_context, team_context, combat_context
        # Each group has different dimensionality, projected to common d_model
        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]
        self._sem_keys = sem_keys

        # Create LayerNorm and Linear layers for each semantic group
        self.sem_in_norm = nn.ModuleDict()
        self.sem_in_proj = nn.ModuleDict()
        for k in sem_keys:
            idxs = self._idx_map.get(k, None)
            if idxs is None or len(idxs) == 0:
                raise RuntimeError(f"Missing/empty semantic index list for '{k}' in SEMANTIC_RICH_BASE_INDICES.")
            din = int(len(idxs))
            self.sem_in_norm[k] = nn.LayerNorm(din)
            self.sem_in_proj[k] = nn.Linear(din, d_model)

        # Instinct embedding (4 features -> d_model)
        self.instinct_in_norm = nn.LayerNorm(self.instinct_dim)
        self.instinct_in_proj = nn.Linear(self.instinct_dim, d_model)

        # Memory token (learnable, not from observation)
        # Acts as a persistent memory across time steps
        self.memory_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Decision tokens (learnable): 3 tokens for different decision-making roles
        # D0: tactical decisions, D1: objective decisions, D2: safety decisions
        self.decision_tokens = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # --- Stage blocks ---
        # Stage 1: Ray encoder - processes ray tokens independently
        self.ray_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(ray_layers)])
        
        # Stage 2: Semantic encoder - processes combined token sequence
        self.sem_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(sem_layers)])
        
        # Stage 3: Fusion - semantic tokens attend to ray tokens
        self.fusion = nn.ModuleList([_CrossAttnBlock(d_model, n_heads) for _ in range(fusion_layers)])

        # --- Readout from decision tokens only ---
        # MLP that takes concatenated decision tokens (3 * d_model) -> logits and value
        self.read_fc0 = nn.Linear(3 * d_model, mlp_hidden)
        self.read_fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        self.critic = nn.Linear(mlp_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for linear layers and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- embedding helpers (stable + AMP-friendly) ----
    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """Embed raw ray features to d_model dimension.
        
        Args:
            rays_raw: Raw ray features of shape (B, num_rays, ray_feat_dim)
            
        Returns:
            Embedded rays of shape (B, num_rays, d_model)
        """
        # Use float32 for LayerNorm stability, then cast to model dtype for projection
        x = self.ray_in_norm(rays_raw.float())
        x = self.ray_in_proj(x.to(dtype=self.ray_in_proj.weight.dtype))
        return x

    def _embed_sem(self, x_raw: torch.Tensor, key: str) -> torch.Tensor:
        """Embed raw semantic features for a specific group to d_model dimension.
        
        Args:
            x_raw: Raw semantic features of shape (B, group_dim)
            key: Semantic group name (e.g., "own_context")
            
        Returns:
            Embedded semantic features of shape (B, d_model)
        """
        n = self.sem_in_norm[key]
        p = self.sem_in_proj[key]
        x = n(x_raw.float())
        x = p(x.to(dtype=p.weight.dtype))
        return x

    def _embed_instinct(self, inst_raw: torch.Tensor) -> torch.Tensor:
        """Embed raw instinct features to d_model dimension.
        
        Args:
            inst_raw: Raw instinct features of shape (B, instinct_dim)
            
        Returns:
            Embedded instinct features of shape (B, d_model)
        """
        x = self.instinct_in_norm(inst_raw.float())
        x = self.instinct_in_proj(x.to(dtype=self.instinct_in_proj.weight.dtype))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through TronBrain.
        
        Args:
            obs: Flat observation tensor of shape (batch_size, OBS_DIM)
                 Layout: [rays_flat | rich_base | instinct]
                 
        Returns:
            Tuple of (logits, value):
                logits: Action logits of shape (batch_size, act_dim)
                value: Value prediction of shape (batch_size, 1)
                
        Raises:
            RuntimeError: If observation shape doesn't match expected dimensions
                         or if output shapes are incorrect
        """
        # ----- hard invariants -----
        # Verify observation is 2D (batch, features)
        if obs.dim() != 2:
            raise RuntimeError(f"TronBrain.forward expects obs 2D [B,F], got shape={tuple(obs.shape)}")
        B, Fdim = int(obs.size(0)), int(obs.size(1))
        # Verify feature dimension matches expected layout
        if Fdim != self.expected_obs_dim:
            raise RuntimeError(
                f"TronBrain.forward obs dim mismatch: got F={Fdim}, expected {self.expected_obs_dim}."
            )

        # ----- split observation into components -----
        # Split according to layout: [rays_flat | rich_base | instinct]
        rays_flat = obs[:, : self.rays_flat_dim]
        rich_base = obs[:, self.rays_flat_dim : self.rays_flat_dim + self.rich_base_dim]
        instinct = obs[:, self.rays_flat_dim + self.rich_base_dim : self.expected_obs_dim]

        # ----- Stage 1: Ray Processing -----
        # Reshape flat rays to (B, num_rays, ray_feat_dim)
        rays_raw = rays_flat.view(B, self.num_rays, self.ray_feat_dim)
        # Embed rays to d_model and add directional embeddings
        ray_tok = self._embed_rays(rays_raw)
        ray_tok = ray_tok + self.ray_dir_embed.to(dtype=ray_tok.dtype, device=ray_tok.device)

        # Apply self-attention blocks to process ray tokens
        for blk in self.ray_encoder:
            ray_tok = blk(ray_tok)

        # ----- Stage 2: Semantic Token Construction -----
        # Build semantic tokens from rich_base partitions
        sem_tokens: List[torch.Tensor] = []
        for k in self._sem_keys:
            # Extract indices for this semantic group
            idxs = self._idx_map[k]
            # Gather features using indices (maintains order from config)
            xk = rich_base.index_select(dim=1, index=torch.tensor(idxs, device=rich_base.device, dtype=torch.long))
            # Embed and add sequence dimension
            sem_tokens.append(self._embed_sem(xk, k).unsqueeze(1))  # (B, 1, D)

        # Embed instinct token from tail
        inst_tok = self._embed_instinct(instinct).unsqueeze(1)  # (B, 1, D)

        # Prepare decision and memory tokens
        dec = self.decision_tokens.expand(B, -1, -1).to(device=obs.device)  # (B, 3, D)
        mem = self.memory_token.expand(B, -1, -1).to(device=obs.device)    # (B, 1, D)
        
        # Concatenate all tokens: decisions (3), semantics (5), instinct (1), memory (1)
        sem = torch.cat(sem_tokens, dim=1)  # (B, 5, D)
        tok = torch.cat([dec, sem, inst_tok, mem], dim=1)  # (B, 3+5+1+1=10, D)

        # Apply self-attention to token sequence
        for blk in self.sem_encoder:
            tok = blk(tok)

        # ----- Stage 3: Fusion (Cross-Attention) -----
        # Semantic+decision tokens attend to processed ray tokens
        for blk in self.fusion:
            tok = blk(tok, ray_tok)

        # ----- Stage 4: Readout -----
        # Extract only decision tokens (first 3) for readout
        dec_out = tok[:, :3, :].reshape(B, 3 * self.d_model)
        
        # MLP head
        h = F.gelu(self.read_fc0(dec_out))
        h = F.gelu(self.read_fc1(h))

        # Final projections
        logits = self.actor(h)              # (B, act_dim)
        value = self.critic(h)              # (B, 1)

        # ----- Final shape asserts (critical "DO NOT BREAK" invariants) -----
        if logits.dim() != 2 or logits.size(0) != B or logits.size(1) != self.act_dim:
            raise RuntimeError(f"Bad logits shape from TronBrain: got {tuple(logits.shape)}, expected ({B},{self.act_dim})")
        if value.dim() != 2 or value.size(0) != B or value.size(1) != 1:
            raise RuntimeError(f"Bad value shape from TronBrain: got {tuple(value.shape)}, expected ({B},1)")

        return logits, value