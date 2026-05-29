import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

# Idea D: Token Dropping (Skip Layers)
# After the EARLY layers extract local syntax, drop low-importance tokens for the
# remaining (more expensive) layers. Importance is computed from the hidden-state
# L2 norm (a cheap proxy for "this token is being attended to / carries signal").
# Subsequent layers process a SHORTER sequence -> attention cost drops quadratically.


class TokenDropEncoder(nn.Module):
    """Wraps a BartEncoder, intercepting the layer loop to drop tokens mid-stream.

    Layers 0..drop_after_layer-1: run on full sequence (dense attention).
    At depth = drop_after_layer: rank tokens by importance, keep top (1 - drop_ratio).
    Remaining layers run on the smaller sequence.
    """
    def __init__(self, encoder: nn.Module, drop_after_layer: int = 3, drop_ratio: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        bsz, seq_len = input_ids.shape

        # BART embedding (BartScaledWordEmbedding already includes scale)
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        # Position embeddings: signature is embed_positions(input) where input has shape info
        embed_pos = self.encoder.embed_positions(input_ids)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.encoder.layernorm_embedding(hidden_states)
        dropout_p = self.encoder.config.dropout
        hidden_states = F.dropout(hidden_states, p=dropout_p, training=self.training)

        # Token-level mask we maintain across layers
        if attention_mask is None:
            current_mask = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.long)
        else:
            current_mask = attention_mask  # [B, T]
        kept_indices = None

        for i, layer in enumerate(self.encoder.layers):
            # Build additive 4D mask from current 2D token mask
            cur_len = hidden_states.size(1)
            mask_f = current_mask.float() if current_mask.dtype != torch.float else current_mask
            ext_mask = (1.0 - mask_f) * torch.finfo(hidden_states.dtype).min
            ext_mask = ext_mask[:, None, None, :]  # [B, 1, 1, cur_len]
            out = layer(hidden_states, attention_mask=ext_mask, layer_head_mask=None, output_attentions=False)
            hidden_states = out[0] if isinstance(out, tuple) else out

            # Drop tokens AFTER this layer if it's the drop_after_layer
            if i + 1 == self.drop_after_layer and self.drop_ratio > 0:
                # Importance = L2 norm of hidden state; mask out padding first
                norms = hidden_states.norm(dim=-1)  # [B, T]
                pad_mask = (current_mask == 0)
                norms = norms.masked_fill(pad_mask, -1e9)
                cur_len = hidden_states.size(1)
                keep_n = max(1, int(cur_len * (1.0 - self.drop_ratio)))
                _, top_idx = torch.topk(norms, k=keep_n, dim=-1)  # [B, keep_n]
                top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order
                gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                hidden_states = torch.gather(hidden_states, 1, gather_idx)
                current_mask = torch.gather(current_mask, 1, top_idx)
                kept_indices = top_idx

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None, attentions=None), current_mask, kept_indices


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class PatchedModel(nn.Module):
    def __init__(self, base_model, drop_after_layer: int = 3, drop_ratio: float = 0.3):
        super().__init__()
        self.model = base_model
        # Wrap the encoder
        encoder = base_model.model.encoder
        self.token_drop_encoder = TokenDropEncoder(encoder, drop_after_layer=drop_after_layer, drop_ratio=drop_ratio)
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden)
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        encoder_out, final_mask, _ = self.token_drop_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last = encoder_out.last_hidden_state
        if final_mask is None:
            final_mask = torch.ones(last.size()[:2], device=last.device, dtype=torch.long)
        pooled = self.attn_pool(last, final_mask)
        logits = self.model.classification_head(pooled)
        loss = None
        if labels is not None:
            if labels.dtype != torch.long: labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.model.config
