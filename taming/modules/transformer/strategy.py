import imp
import math
import numpy as np
import torch

class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner_attention = self.get_inner_att(config)
        self.query_projection = nn.Linear(config.n_embd, config.n_embd)
        self.key_projection = nn.Linear(config.n_embd, config.n_embd)
        self.value_projection = nn.Linear(config.n_embd, config.n_embd)
        self.out_projection = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_head

    def forward(self, queries, keys, values, attn_mask=None, *args, **kargs):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(queries, keys, values, attn_mask).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class LinearAttention(AttentionLayer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_inner_att(self, config):
        return InnerLinearAttention(config.n_embd)


class CausalLinearAttention(AttentionLayer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_inner_att(self, config):
        return InnerCausalLinearAttention(config.n_embd)


class InnerCausalLinearAttention(nn.Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, *args, **kargs):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Compute the normalizers
        Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)

        # Compute the unnormalized result
        V = causal_linear(Q, K, values)

        return V * Z[:, :, :, None]
        

class InnerLinearAttention(nn.Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, *args, **kargs):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()
        
class LinearAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.W_q= nn.Linear(config.n_embd, config.n_embd)
        self.W_k = nn.Linear(config.n_embd, config.n_embd)
        self.W_v = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.eps = 1e-6

    def forward(self, query, key, value, attn_mask=None,  *args, **kargs):
        N, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_head
        query = self.W_q(query).view(N, L, H, -1)
        key = self.W_k(key).view(N, S, H, -1)
        value = self.W_v(value).view(N, S, H, -1)
        Q = F.elu(query) + 1
        K = F.elu(key) + 1
        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, value)
        # Compute the normalizer
        Z = Z = torch.ones(1, dtype=K.dtype, device=K.device) \
            / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)
        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        V = self.resid_drop(self.proj(V.contiguous()))
        return 

class CausalLinearAttention(LinearAttention):
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, query, key, value, attn_mask=None, *args, **kargs):
        N, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_head
        query = self.W_q(query).view(N, L, H, -1)
        key = self.W_k(key).view(N, S, H, -1)
        value = self.W_v(value).view(N, S, H, -1)
        Q = F.elu(query) + 1
        K = F.elu(key) + 1
        # Compute the normalizers
        Z = 1.0 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        V = causal_linear(Q, K, value) * Z[:, :, :, None] 
        V = self.resid_drop(self.proj(V.contiguous()))
        return V

def _gamma_func(r, mode="cosine"):
    if mode == "linear":
        return r
    elif mode == "cosine":
        return 1 - np.cos(r * np.pi / 2)
    elif mode == "square":
        return r**2
    elif mode == "cubic":
        return r**3
    else:
        raise NotImplementedError

def get_uique_mask(target_ids):
    batch_size, seq_len = target_ids.shape
    source_mask_list = []
    for _ in range(batch_size):
        rt = np.random.uniform()
        #rt = 1 - math.sqrt(1-rt)
        source_mask = torch.bernoulli(rt * torch.ones(seq_len, dtype=torch.float))
        # omit mask which is all ones, nothing to predict
        while int(source_mask.sum()) == seq_len:
            source_mask = torch.bernoulli(rt * torch.ones(seq_len, dtype=torch.float))
        source_mask_list.append(source_mask)
    batch_source_mask = torch.stack(source_mask_list).long().to(target_ids.device)
    return batch_source_mask

def get_batch_mask(target_ids):
    rt = math.floor(_gamma_func(np.random.uniform()) * target_ids.shape[1])
    sample = torch.rand(target_ids.shape, device=target_ids.device).topk(rt, dim=1).indices
    mask = torch.zeros(target_ids.shape, dtype=torch.bool, device=target_ids.device)
    mask = mask.scatter_(dim=1, index=sample, value=True).long()
    return mask

def _select_tokens(probs, pred_ids, num_to_decode):
        _, index = torch.topk(probs, k=num_to_decode, dim=1)
        true_matrix = torch.ones_like(probs).bool()
        false_matrix = torch.zeros_like(probs).bool()
        mask = false_matrix.scatter_(1, index, true_matrix).long()
        return mask.to(pred_ids.device)

def mask_predict(probs, pred_ids, num_nar_steps=None, step=None,
                last_latent=None, last_mask=None, **kargs):
        assert step != 0
        rt = _gamma_func(step / num_nar_steps)
        num_to_decode = math.ceil(rt * pred_ids.shape[1])
        num_to_decode = num_to_decode - last_mask[0].sum()
        
        if num_to_decode < 0:
            new_latent, new_mask = last_latent, last_mask
        else:
            probs *= (1-last_mask)
            mask = _select_tokens(probs, pred_ids, num_to_decode)
            try:
                new_latent = pred_ids*mask + last_latent*(1-mask)
            except:
                breakpoint()
            new_mask = last_mask + mask
        return new_latent, new_mask

def block_predict(probs, pred_ids, num_nar_steps=None, step=None,
                last_latent=None, last_mask=None, **kargs):
    pass
