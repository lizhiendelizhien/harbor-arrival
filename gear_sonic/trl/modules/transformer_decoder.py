"""Causal Transformer decoder used by the Stage 3.1 action policy."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Apply rotary position embeddings to attention queries and keys."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        inv_freq = torch.repeat_interleave(inv_freq, 2)
        rotate_indices = torch.arange(head_dim, dtype=torch.long).reshape(-1, 2).flip(-1).flatten()
        rotate_sign = torch.ones(head_dim, dtype=torch.float32)
        rotate_sign[::2] = -1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("rotate_indices", rotate_indices, persistent=False)
        self.register_buffer("rotate_sign", rotate_sign.reshape(1, 1, 1, head_dim), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, dim=-1, index=self.rotate_indices) * self.rotate_sign.to(
            dtype=x.dtype
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = query.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=query.device, dtype=self.inv_freq.dtype)
            angles = position_ids[None, None, :, None] * self.inv_freq[None, None, None, :]
            cos = angles.cos().to(dtype=query.dtype)
            sin = angles.sin().to(dtype=query.dtype)
        else:
            position_ids = position_ids.to(device=query.device, dtype=self.inv_freq.dtype)
            angles = position_ids[:, None, :, None] * self.inv_freq[None, None, None, :]
            cos = angles.cos().to(dtype=query.dtype)
            sin = angles.sin().to(dtype=query.dtype)
        return (
            query * cos + self._rotate_half(query) * sin,
            key * cos + self._rotate_half(key) * sin,
        )


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and a bounded causal window."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        attention_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key, position_ids=position_ids)
        return query, key, value

    def forward(
        self, x: torch.Tensor, episode_attnmask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        query, key, value = self._project_qkv(x)

        positions = torch.arange(seq_len, device=x.device)
        distance = positions[:, None] - positions[None, :]
        allowed = (distance >= 0) & (distance < self.window_size)
        allowed = allowed[None, None, :, :]
        if episode_attnmask is not None:
            if episode_attnmask.shape != (batch_size, seq_len, seq_len):
                raise ValueError(
                    "episode_attnmask must have shape "
                    f"{(batch_size, seq_len, seq_len)}, got {tuple(episode_attnmask.shape)}"
                )
            allowed = allowed & ~episode_attnmask[:, None, :, :].bool()

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.output(attended)

    def forward_with_past(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        past_valid: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        allowed_mask: torch.Tensor | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        query, key, value = self._project_qkv(x, position_ids=position_ids)
        if past_key is not None:
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        total_len = key.shape[-2]
        if allowed_mask is None:
            key_positions = torch.arange(total_len, device=x.device)
            query_positions = key_positions[-seq_len:]
            distance = query_positions[:, None] - key_positions[None, :]
            allowed = (distance >= 0) & (distance < self.window_size)
            allowed = allowed[None, None, :, :].expand(batch_size, 1, seq_len, total_len)
        else:
            if allowed_mask.shape != (batch_size, seq_len, total_len):
                raise ValueError(
                    "allowed_mask must have shape "
                    f"{(batch_size, seq_len, total_len)}, got {tuple(allowed_mask.shape)}"
                )
            allowed = allowed_mask[:, None, :, :].bool()

        if past_valid is not None:
            current_valid = torch.ones(
                batch_size, seq_len, dtype=x.dtype, device=x.device
            )
            valid = torch.cat([past_valid.to(dtype=x.dtype), current_valid], dim=1)
            if valid.shape[1] != total_len:
                valid = valid[:, -total_len:]
            allowed = allowed & (valid[:, None, None, :] > 0.5)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        output = self.output(attended)
        if not return_cache:
            return output
        new_key = key[..., -self.window_size :, :].detach()
        new_value = value[..., -self.window_size :, :].detach()
        return output, {"key": new_key, "value": new_value}

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        cache_valid: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        if seq_len != 1:
            raise ValueError(f"forward_step expects a single frame, got seq_len={seq_len}")
        query, key, value = self._project_qkv(x, position_ids=position_ids)

        if cache is not None:
            key = torch.cat([cache["key"], key], dim=-2)
            value = torch.cat([cache["value"], value], dim=-2)
        if key.shape[-2] > self.window_size:
            key = key[..., -self.window_size :, :]
            value = value[..., -self.window_size :, :]

        if cache_valid is None:
            allowed = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        else:
            current_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            allowed = torch.cat([cache_valid, current_valid], dim=1)
            if allowed.shape[1] > self.window_size:
                allowed = allowed[:, -self.window_size :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[:, None, None, :],
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        new_cache = {"key": key.detach(), "value": value.detach()}
        return self.output(attended), new_cache


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer block with SiLU feed-forward layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        window_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            attention_dropout=attention_dropout,
            rope_base=rope_base,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, episode_attnmask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), episode_attnmask))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        cache_valid: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_out, new_cache = self.attn.forward_step(
            self.attn_norm(x),
            cache=cache,
            cache_valid=cache_valid,
            position_ids=position_ids,
        )
        x = x + self.dropout(attn_out)
        return x + self.dropout(self.ffn(self.ffn_norm(x))), new_cache

    def forward_with_past(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        past_valid: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        allowed_mask: torch.Tensor | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_result = self.attn.forward_with_past(
            self.attn_norm(x),
            past_key=past_key,
            past_value=past_value,
            past_valid=past_valid,
            position_ids=position_ids,
            allowed_mask=allowed_mask,
            return_cache=return_cache,
        )
        if return_cache:
            attn_out, new_cache = attn_result
        else:
            attn_out = attn_result
            new_cache = None
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if return_cache:
            return x, new_cache
        return x


class TransformerActionDecoder(nn.Module):
    """Map a sequence of per-frame policy inputs to per-frame actions."""

    is_transformer_decoder = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        ffn_dim: int = 1024,
        window_size: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_base: float = 10000.0,
        rope_max_seq_len: int = 1048576,
        rope_reset_margin: int = 1024,
        primary_output_dim: int | None = None,
        output_head_dims: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.rope_reset_margin = int(rope_reset_margin)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    window_size=window_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        if primary_output_dim is not None and not (0 < primary_output_dim < output_dim):
            raise ValueError(
                "primary_output_dim must be between 0 and output_dim when a separate "
                f"auxiliary head is requested, got {primary_output_dim=} and {output_dim=}"
            )
        if output_head_dims is not None:
            output_head_dims = tuple(int(dim) for dim in output_head_dims)
            if len(output_head_dims) < 2 or any(dim <= 0 for dim in output_head_dims):
                raise ValueError(
                    "output_head_dims must contain at least two positive dimensions, "
                    f"got {output_head_dims}"
                )
            if sum(output_head_dims) != output_dim:
                raise ValueError(
                    f"output_head_dims sum to {sum(output_head_dims)}, expected {output_dim}"
                )
            if primary_output_dim is not None and output_head_dims[0] != primary_output_dim:
                raise ValueError(
                    "The first output_head_dims entry must equal primary_output_dim, "
                    f"got {output_head_dims[0]} and {primary_output_dim}"
                )
            primary_output_dim = output_head_dims[0]
        self.primary_output_dim = primary_output_dim
        primary_dim = output_dim if primary_output_dim is None else primary_output_dim
        self.action_head = nn.Linear(d_model, primary_dim)
        if output_head_dims is None:
            auxiliary_dims = (
                () if primary_output_dim is None else (output_dim - primary_output_dim,)
            )
        else:
            auxiliary_dims = output_head_dims[1:]
        # Keep the first auxiliary projection under its Stage4_1 parameter name
        # so a Stage4_1 checkpoint can initialize Stage4_2's reconstruction head.
        self.auxiliary_head = (
            nn.Linear(d_model, auxiliary_dims[0]) if auxiliary_dims else None
        )
        self.extra_output_heads = nn.ModuleList(
            nn.Linear(d_model, dim) for dim in auxiliary_dims[1:]
        )
        self.output_head_dims = (
            (output_dim,) if output_head_dims is None else output_head_dims
        )
        self._kv_cache: list[dict[str, torch.Tensor]] | None = None
        self._cache_valid: torch.Tensor | None = None
        self._cache_positions: torch.Tensor | None = None
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _project_outputs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the primary and optional auxiliary output heads."""
        normalized = self.final_norm(hidden)
        primary = self.action_head(normalized)
        if self.auxiliary_head is None:
            return primary
        outputs = [primary, self.auxiliary_head(normalized)]
        outputs.extend(head(normalized) for head in self.extra_output_heads)
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        episode_attnmask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_prefix_len = int(kwargs.pop("kv_prefix_len", 0) or 0)
        kv_prefix_state = kwargs.pop("kv_prefix_state", None)
        kv_prefix_dones = kwargs.pop("kv_prefix_dones", None)
        if kwargs:
            unused = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected TransformerActionDecoder kwargs: {unused}")
        if x.ndim != 3:
            raise ValueError(f"TransformerActionDecoder expects [B, S, F], got {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected frame dim {self.input_dim}, got {x.shape[-1]}")
        if kv_prefix_state is not None:
            return self.forward_with_rollout_prefix_kv(
                x,
                kv_prefix_state=kv_prefix_state,
                dones=kv_prefix_dones,
            )
        if kv_prefix_len > 0:
            return self.forward_with_prefix_kv(
                x,
                prefix_len=kv_prefix_len,
                episode_attnmask=episode_attnmask,
            )
        hidden = self.input_projection(x)
        for layer in self.layers:
            hidden = layer(hidden, episode_attnmask)
        return self._project_outputs(hidden)

    def _prefix_target_allowed_mask(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        episode_attnmask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        target_len = seq_len - prefix_len
        key_positions = torch.arange(seq_len, device=device)
        query_positions = torch.arange(prefix_len, seq_len, device=device)
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < self.window_size)
        allowed = allowed[None, :, :].expand(batch_size, target_len, seq_len)
        if episode_attnmask is not None:
            if episode_attnmask.shape != (batch_size, seq_len, seq_len):
                raise ValueError(
                    "episode_attnmask must have shape "
                    f"{(batch_size, seq_len, seq_len)}, got {tuple(episode_attnmask.shape)}"
                )
            allowed = allowed & ~episode_attnmask[:, prefix_len:, :].bool()
        return allowed

    def forward_with_prefix_kv(
        self,
        x: torch.Tensor,
        prefix_len: int,
        episode_attnmask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if prefix_len <= 0:
            return self.forward(x, episode_attnmask=episode_attnmask)
        if prefix_len >= seq_len:
            raise ValueError(
                f"kv_prefix_len must be smaller than sequence length, got {prefix_len=} "
                f"for seq_len={seq_len}"
            )

        prefix_hidden = self.input_projection(x[:, :prefix_len]).detach()
        target_hidden = self.input_projection(x[:, prefix_len:])
        target_len = seq_len - prefix_len
        device = x.device
        prefix_positions = torch.arange(prefix_len, device=device)[None, :].expand(
            batch_size, prefix_len
        )
        target_positions = torch.arange(prefix_len, seq_len, device=device)[None, :].expand(
            batch_size, target_len
        )
        target_allowed = self._prefix_target_allowed_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            prefix_len=prefix_len,
            episode_attnmask=episode_attnmask,
            device=device,
        )

        for layer in self.layers:
            with torch.no_grad():
                norm_prefix = layer.attn_norm(prefix_hidden)
                _, prefix_key, prefix_value = layer.attn._project_qkv(
                    norm_prefix,
                    position_ids=prefix_positions,
                )
                prefix_mask = None
                if episode_attnmask is not None:
                    prefix_mask = episode_attnmask[:, :prefix_len, :prefix_len]
                prefix_hidden = layer(prefix_hidden, prefix_mask).detach()

            target_hidden = layer.forward_with_past(
                target_hidden,
                past_key=prefix_key.detach(),
                past_value=prefix_value.detach(),
                past_valid=None,
                position_ids=target_positions,
                allowed_mask=target_allowed,
                return_cache=False,
            )

        return self._project_outputs(target_hidden)

    def _rollout_prefix_allowed_mask(
        self,
        batch_size: int,
        target_len: int,
        prefix_len: int,
        prefix_valid: torch.Tensor,
        dones: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = prefix_len + target_len
        key_positions = torch.arange(total_len, device=device)
        query_positions = torch.arange(prefix_len, total_len, device=device)
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < self.window_size)
        allowed = allowed[None, :, :].expand(batch_size, target_len, total_len).clone()

        prefix_valid = prefix_valid.to(device=device, dtype=torch.bool)
        if prefix_valid.shape != (batch_size, prefix_len):
            raise ValueError(
                f"prefix_valid must have shape {(batch_size, prefix_len)}, "
                f"got {tuple(prefix_valid.shape)}"
            )
        allowed[:, :, :prefix_len] &= prefix_valid[:, None, :]

        if dones is not None:
            dones = dones.to(device=device, dtype=torch.bool)
            if dones.shape != (batch_size, target_len):
                raise ValueError(
                    f"dones must have shape {(batch_size, target_len)}, got {tuple(dones.shape)}"
                )
            target_starts = torch.zeros_like(dones)
            if target_len > 1:
                target_starts[:, 1:] = dones[:, :-1]
            target_episode_ids = torch.cumsum(target_starts.to(torch.long), dim=1)

            prefix_same_episode = target_episode_ids == 0
            allowed[:, :, :prefix_len] &= prefix_same_episode[:, :, None]

            target_same_episode = (
                target_episode_ids[:, :, None] == target_episode_ids[:, None, :]
            )
            allowed[:, :, prefix_len:] &= target_same_episode

        return allowed

    def forward_with_rollout_prefix_kv(
        self,
        x: torch.Tensor,
        kv_prefix_state: dict[str, torch.Tensor],
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, target_len, _ = x.shape
        past_key_values = kv_prefix_state["past_key_values"].to(device=x.device)
        prefix_valid = kv_prefix_state["cache_valid"].to(device=x.device)
        abs_position = kv_prefix_state["abs_position"].to(device=x.device, dtype=x.dtype)
        if past_key_values.shape[0] != len(self.layers) or past_key_values.shape[1] != 2:
            raise ValueError(
                "past_key_values must have shape "
                f"[num_layers, 2, batch, heads, window, head_dim], got {tuple(past_key_values.shape)}"
            )
        if past_key_values.shape[2] != batch_size:
            raise ValueError(
                f"past_key_values batch mismatch: {past_key_values.shape[2]} vs {batch_size}"
            )

        prefix_len = past_key_values.shape[-2]
        target_offsets = torch.arange(target_len, device=x.device, dtype=x.dtype)
        target_positions = abs_position[:, None] + target_offsets[None, :]
        allowed_mask = self._rollout_prefix_allowed_mask(
            batch_size=batch_size,
            target_len=target_len,
            prefix_len=prefix_len,
            prefix_valid=prefix_valid,
            dones=dones,
            device=x.device,
        )

        hidden = self.input_projection(x)
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.forward_with_past(
                hidden,
                past_key=past_key_values[layer_idx, 0].detach(),
                past_value=past_key_values[layer_idx, 1].detach(),
                past_valid=prefix_valid,
                position_ids=target_positions,
                allowed_mask=allowed_mask,
                return_cache=False,
            )
        return self._project_outputs(hidden)

    def forward_onnx_step(
        self,
        x: torch.Tensor,
        past_key_values: torch.Tensor,
        cache_valid: torch.Tensor,
        abs_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"forward_onnx_step expects [B, F] or [B, 1, F], got {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected frame dim {self.input_dim}, got {x.shape[-1]}")

        batch_size = x.shape[0]
        position_ids = abs_position.to(device=x.device, dtype=x.dtype).reshape(batch_size, 1)
        hidden = self.input_projection(x)
        new_cache = []
        current_valid = torch.ones(batch_size, 1, dtype=cache_valid.dtype, device=x.device)
        valid_index = torch.full(
            (batch_size, 1),
            self.window_size - 1,
            dtype=torch.long,
            device=x.device,
        )
        present_valid = torch.roll(cache_valid, shifts=-1, dims=1).scatter(
            1,
            valid_index,
            current_valid,
        )
        allowed = present_valid[:, None, None, :] > 0.5
        for layer_idx, layer in enumerate(self.layers):
            layer_key = torch.roll(past_key_values[layer_idx, 0], shifts=-1, dims=-2)
            layer_value = torch.roll(past_key_values[layer_idx, 1], shifts=-1, dims=-2)
            query, key, value = layer.attn._project_qkv(
                layer.attn_norm(hidden),
                position_ids=position_ids,
            )
            key_index = torch.full_like(key, self.window_size - 1, dtype=torch.long)
            value_index = torch.full_like(value, self.window_size - 1, dtype=torch.long)
            layer_key = layer_key.scatter(-2, key_index, key)
            layer_value = layer_value.scatter(-2, value_index, value)
            attended = F.scaled_dot_product_attention(
                query,
                layer_key,
                layer_value,
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(
                batch_size,
                1,
                hidden.shape[-1],
            )
            hidden = hidden + layer.dropout(layer.attn.output(attended))
            hidden = hidden + layer.dropout(layer.ffn(layer.ffn_norm(hidden)))
            new_cache.append(torch.stack([layer_key, layer_value], dim=0))

        present_key_values = torch.stack(new_cache, dim=0)
        present_abs_position = abs_position.to(device=x.device, dtype=x.dtype) + 1.0
        outputs = self._project_outputs(hidden).squeeze(1)
        return outputs, present_key_values, present_valid, present_abs_position

    def reset_cache(self, reset_mask: torch.Tensor | None = None):
        if reset_mask is None or self._kv_cache is None:
            self._kv_cache = None
            self._cache_valid = None
            self._cache_positions = None
            return
        reset_mask = reset_mask.to(device=self._cache_valid.device, dtype=torch.bool)
        if reset_mask.ndim != 1:
            reset_mask = reset_mask.view(-1)
        if reset_mask.numel() != self._cache_valid.shape[0]:
            self._kv_cache = None
            self._cache_valid = None
            self._cache_positions = None
            return
        if not reset_mask.any():
            return
        self._cache_valid[reset_mask] = False
        self._cache_positions[reset_mask] = 0
        for layer_cache in self._kv_cache:
            layer_cache["key"][reset_mask] = 0
            layer_cache["value"][reset_mask] = 0

    def _maybe_reset_rope_window(self):
        if (
            self._kv_cache is None
            or self._cache_valid is None
            or self._cache_positions is None
            or self.rope_max_seq_len <= 0
        ):
            return
        reset_at = (
            self.rope_max_seq_len - self.rope_reset_margin
            if self.rope_max_seq_len > self.rope_reset_margin
            else self.rope_max_seq_len
        )
        overflow_mask = self._cache_positions >= reset_at
        if overflow_mask.any():
            self.reset_cache(overflow_mask)

    def get_cache_state(self) -> dict[str, torch.Tensor] | None:
        if self._kv_cache is None or self._cache_valid is None or self._cache_positions is None:
            return None
        history_len = max(self.window_size - 1, 0)
        if history_len == 0:
            history_slice = slice(0, 0)
        else:
            history_slice = slice(-history_len, None)
        past_key_values = torch.stack(
            [
                torch.stack(
                    [
                        layer_cache["key"][..., history_slice, :],
                        layer_cache["value"][..., history_slice, :],
                    ],
                    dim=0,
                )
                for layer_cache in self._kv_cache
            ],
            dim=0,
        )
        return {
            "past_key_values": past_key_values.detach().clone(),
            "cache_valid": self._cache_valid[:, history_slice].detach().clone(),
            "abs_position": self._cache_positions.detach().clone(),
        }

    def forward_step(
        self,
        x: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"forward_step expects [B, 1, F] or [B, F], got {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected frame dim {self.input_dim}, got {x.shape[-1]}")

        batch_size = x.shape[0]
        device = x.device
        if (
            self._cache_valid is None
            or self._cache_valid.shape[0] != batch_size
            or self._cache_valid.device != device
        ):
            self._kv_cache = None
            self._cache_valid = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            self._cache_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        if reset_mask is not None:
            self.reset_cache(reset_mask)
            if self._cache_valid is None:
                self._cache_valid = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
                self._cache_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        self._maybe_reset_rope_window()
        old_valid = self._cache_valid
        position_ids = self._cache_positions[:, None]
        hidden = self.input_projection(x)
        old_cache = self._kv_cache
        new_cache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if old_cache is None else old_cache[layer_idx]
            hidden, layer_cache = layer.forward_step(
                hidden,
                cache=layer_cache,
                cache_valid=old_valid,
                position_ids=position_ids,
            )
            new_cache.append(layer_cache)

        current_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        self._cache_valid = torch.cat([old_valid, current_valid], dim=1)
        if self._cache_valid.shape[1] > self.window_size:
            self._cache_valid = self._cache_valid[:, -self.window_size :]
        self._cache_positions = self._cache_positions + 1
        self._kv_cache = new_cache
        return self._project_outputs(hidden)
