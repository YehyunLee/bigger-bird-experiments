import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

from shared.patched_model import classification_forward
from shared.sparse_attn_utils import token_mask_1d


class LightningHybridAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, block_size: int = 64):
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout),
            is_decoder=base_attn.is_decoder,
            bias=base_attn.k_proj.bias is not None,
        )
        self.q_proj.load_state_dict(base_attn.q_proj.state_dict())
        self.k_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_proj.load_state_dict(base_attn.v_proj.state_dict())
        self.out_proj.load_state_dict(base_attn.out_proj.state_dict())
        self.block_size = block_size

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _windowed_softmax_attention(self, Q, K, V, attention_mask, bsz, tgt_len, src_len):
        """Sliding-window attention via unfold: O(T * W * d), no T x T tensor."""
        half = self.block_size // 2
        w = min(2 * half + 1, src_len)
        device, dtype = Q.device, Q.dtype
        BH = Q.size(0)
        neg_inf = torch.finfo(dtype).min

        K_pad = F.pad(K, (0, 0, half, half))
        V_pad = F.pad(V, (0, 0, half, half))
        K_win = K_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)
        V_win = V_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)

        scores = torch.einsum("btd,btwd->btw", Q, K_win) / (self.head_dim ** 0.5)

        token_mask = token_mask_1d(attention_mask, bsz, src_len, device)
        if token_mask is not None:
            q_pos = torch.arange(tgt_len, device=device).view(1, -1, 1)
            col_off = torch.arange(w, device=device).view(1, 1, -1) - half
            key_pos = (q_pos + col_off).clamp(0, src_len - 1)
            am = token_mask.unsqueeze(1).unsqueeze(1).expand(bsz, self.num_heads, tgt_len, src_len)
            am = am.reshape(BH, tgt_len, src_len)
            key_allowed = torch.gather(am, 2, key_pos.expand(BH, -1, -1))
            scores = scores.masked_fill(~key_allowed, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.einsum("btw,btwd->btd", attn, V_win)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(query_states, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = key_states.reshape(BH, -1, self.head_dim)
        V = value_states.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        local_out = self._windowed_softmax_attention(
            Q, K, V, attention_mask, bsz, tgt_len, src_len
        )

        if tgt_len <= self.block_size * 4:
            out = local_out
        else:
            Q_l = F.elu(Q) + 1.0
            K_l = F.elu(K) + 1.0
            token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)
            if token_mask is not None:
                pm = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
                K_l = K_l * pm.unsqueeze(-1)
            KV = torch.bmm(K_l.transpose(1, 2), V)
            Z = K_l.sum(dim=1, keepdim=True)
            Num = torch.bmm(Q_l, KV)
            Den = torch.bmm(Q_l, Z.transpose(1, 2))
            global_out = Num / (Den + 1e-6)
            out = local_out + 0.5 * global_out

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(model: nn.Module, block_size: int = 64):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, LightningHybridAttention(child, block_size))
            else:
                _recurse(child)

    _recurse(model)


class PatchedModel(nn.Module):
    def __init__(self, base_model, block_size=64):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, block_size=block_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)

    @property
    def config(self):
        return self.model.config
