import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class FusionEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_sizes: list, emb_sizes: list, device):
        super().__init__()
        self.embs = [nn.Linear(1, e) if s == -1 else nn.Embedding(s, e) for s, e in zip(vocab_sizes, emb_sizes)]
        self.embs = [x.to(device) for x in self.embs]
        self.proj = nn.Linear(sum(emb_sizes), d_model)
        self.d_model = d_model

    def forward(self, features_list):
        x = [self.embs[i](x) for i, x in enumerate(features_list)]
        output = self.proj(torch.concat(x, axis=-1))
        return output * math.sqrt(self.d_model)
    
class ModifiedFusionEmbeddings(nn.Module):
    def __init__(self, d_model, ntokens_location, ntokens_user, vocab_sizes: list, emb_sizes: list, device):
        super(ModifiedFusionEmbeddings, self).__init__()

        # Embedding for user and location tokens
        self.user_embedding = nn.Embedding(ntokens_user, emb_sizes[0]).to(device)
        self.location_embedding = nn.Embedding(ntokens_location , emb_sizes[0]).to(device)

        # Embeddings for the rest of the features
        self.additional_embs = [nn.Linear(1, e) if s == -1 else nn.Embedding(s, e) for s, e in zip(vocab_sizes[1:], emb_sizes[1:])]
        self.additional_embs = [x.to(device) for x in self.additional_embs]
        
        # Projection layer
        self.proj = nn.Linear(sum(emb_sizes) , d_model).to(device)
        self.d_model = d_model

    def forward(self, features_list):
        # Extract user and location data
        user_location_data = features_list[0]
        user_ids = user_location_data[:, 0]
        location_ids = user_location_data[:, 1:]

        # Embedding user and location 
        user_embed = self.user_embedding(user_ids)
        location_embed = self.location_embedding(location_ids)

        # Concatenate user and location embeddings
        user_location_combined = torch.cat([user_embed.unsqueeze(1), location_embed], dim=1)
        # Embedding additional features
        additional_features = [self.additional_embs[i](x) for i, x in enumerate(features_list[1:])]
            
        # Concatenating all embeddings
        combined = torch.cat([user_location_combined] + additional_features, dim=-1)

        # Projecting to common dimension
        output = self.proj(combined)
        return output * math.sqrt(self.d_model)


    
class ModifiedEmbeddings(nn.Module):
    def __init__(self, d_model, ntokens_location, ntokens_user):
        super(ModifiedEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(ntokens_user, d_model)
        self.location_embedding = nn.Embedding( ntokens_location, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Assuming x[:, 0] is userId and the rest are locations
        user_embed = self.user_embedding(x[:, 0])
        location_embed = self.location_embedding(x[:, 1:])

        # Concatenating user embedding with location embeddings
        combined = torch.cat([user_embed.unsqueeze(1), location_embed], dim=1)
        return combined * math.sqrt(self.d_model)
    
# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model

    # def forward(self, x):
    #     return self.lut(x) * math.sqrt(self.d_model)
    
# old positional encoding class
# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], 
#                          requires_grad=False)
#         return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.wpe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.long, device= x.device)
        pos_emb = self.wpe(pos) # position embeddings of shape (seq_len, d_model)
        x = x + pos_emb
        return self.dropout(x)


class BERT_model(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed, d_model, ntokens):
        # print("BERT_model init")
        super(BERT_model, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.proj = nn.Linear(d_model, ntokens)
        
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        return self.proj(self.encode(src, src_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)