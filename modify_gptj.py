import types
from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJAttention


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(
    tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    # tensor: [1, kv_seq_len, 16, 64]
    # sin, cos: [1, kv_seq_len, 32]
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)  # [1, kv_seq_len, 1, 64]
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)  # [1, kv_seq_len, 1, 64]
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


def pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Tuple[torch.Tensor]],
    Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
]:
    query = self.q_proj(hidden_states)  # [1, 1, 4096]
    key = self.k_proj(hidden_states)  # [1, 1, 4096]
    value = self.v_proj(hidden_states)

    query = self._split_heads(
        query, self.num_attention_heads, self.head_dim, True
    )  # [1, 1, 16, 256]
    key = self._split_heads(
        key, self.num_attention_heads, self.head_dim, True
    )  # [1, 1, 16, 256]
    value = self._split_heads(
        value, self.num_attention_heads, self.head_dim, False
    )  # [1, 16, 1, 256]

    kv_seq_len = 1
    if layer_past is not None:
        past_key = layer_past[0]  # [1, kv_seq_len - 1, 16, 256]
        past_value = layer_past[1]  # [1, 16, kv_seq_len - 1, 256]
        key = torch.cat((past_key, key), dim=1)  # [1, kv_seq_len, 16,  256]
        value = torch.cat((past_value, value), dim=2)  # [1, 16, kv_seq_len, 256]
        kv_seq_len = key.shape[1]

    if use_cache is True:
        present = (key.to(hidden_states.dtype), value)
    else:
        present = None

    embed_positions = self._get_embed_positions(
        position_ids
    )  # [1, 2048, 64], [bsz, ctx, RoPe_dim]

    position_ids = (
        torch.arange(kv_seq_len).unsqueeze(0).to(embed_positions.device)
    )  # [1, kv_seq_len]
    repeated_position_ids = position_ids.unsqueeze(-1).repeat(
        1, 1, embed_positions.shape[-1]
    )  # [1, kv_seq_len, 64]
    seq_sincos = torch.gather(
        embed_positions, 1, repeated_position_ids
    )  # [1, kv_seq_len, 64]
    seq_sin, seq_cos = torch.split(
        seq_sincos, seq_sincos.shape[-1] // 2, dim=-1
    )  # [1, kv_seq_len, 32]

    k_rot = key[:, :, :, : self.rotary_dim]  # [1, kv_seq_len, 16, 64]
    k_pass = key[:, :, :, self.rotary_dim :]  # [1, kv_seq_len, 16, 256 - 64]
    k_rot = apply_rotary_pos_emb(k_rot, seq_sin, seq_cos)  # [1, kv_seq_len, 16, 64]

    pos_sin = seq_sin[:, -1, :].unsqueeze(1)  # [1, 1, 32]
    pos_cos = seq_cos[:, -1, :].unsqueeze(1)  # [1, 1, 32]
    q_rot = query[:, :, :, : self.rotary_dim]  # [1, 1, 16, 64]
    q_pass = query[:, :, :, self.rotary_dim :]  # [1, 1, 16, 256 - 64]
    q_rot = apply_rotary_pos_emb(q_rot, pos_sin, pos_cos)  # [1, 1, 16, 64]

    key = torch.cat([k_rot, k_pass], dim=-1)  # [1, kv_seq_len, 16, 256]
    query = torch.cat([q_rot, q_pass], dim=-1)  # [1, 1, 16, 256]

    key = key.permute(0, 2, 1, 3)  # [1, 16, kv_seq_len, 256]
    query = query.permute(0, 2, 1, 3)  # [1, 16, 1, 256]

    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_dim
    )
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def modify_gptj(model: AutoModelForCausalLM):
    """ Modify the GPT-J model to apply positional embeddings relative to the cache, not the overall sequence length.

    Args:
        model (AutoModelForCausalLM): The model to modify.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            modify_gptj(module)

        if isinstance(module, GPTJAttention):
            model._modules[name].forward = types.MethodType(
                pos_shift_attention_forward, model._modules[name]
            )
