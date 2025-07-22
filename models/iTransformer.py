import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.classifier = nn.Linear(configs.d_model * (configs.enc_in + 1), configs.num_class)
        self.dropout = nn.Dropout(configs.dropout)
        #self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Position Encoding
        pos_vector = torch.arange(1, L + 1)
        
        x_mark_enc = pos_vector.float().repeat(B, 1).unsqueeze(-1).to(x_enc.device)

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        x = enc_out.view(enc_out.size(0), -1)  # Flatten: (batch_size, input_dim * d_model)
        x = self.dropout(x)
        return self.classifier(x), attns  # (batch_size, num_classes)

        # # B N E -> B N S -> B S N 
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # return dec_out, attns


    def forward(self, x_cwt, x_enc, mask=None):
        # Input [B, D, L]

        x_enc = x_enc.permute(0, 2, 1)

        dec_out, attns = self.forecast(x_enc)
        
        if self.output_attention:
            return dec_out #[:, -self.pred_len:, :], attns
        else:
            return dec_out#[:, -self.pred_len:, :]  # [B, L, D]