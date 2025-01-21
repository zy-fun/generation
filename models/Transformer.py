import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.Attention import FullAttention, AttentionLayer
from layers.Embed import EdgeEmbedding
import numpy as np

class Model(nn.Module):
    def __init__(self, configs, pretrained_emb=None):
        super(Model, self).__init__()
        self.configs = configs
        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_layers = configs.d_layers
        dropout = configs.dropout
        activation = configs.activation

        n_vocab = configs.n_vocab
        enc_emb = configs.enc_emb
        enc_dim = configs.enc_dim
        dec_dim = configs.dec_dim
        c_out = configs.c_out

        # Embedding
        self.enc_embedding = EdgeEmbedding(n_vocab, enc_emb, enc_dim, d_model, dropout=dropout, pretrained=pretrained_emb) 
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.dec_embedding = nn.Linear(dec_dim, d_model)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, scale=None, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, scale=None, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    # def from_pretrained_embedding(self, pretrained):
    #     self.enc_embedding.from_pretrained(pretrained, freeze=True)

    def forward(self, enc_seq, enc_feature, dec_in):
        # Embedding
        enc_out = self.enc_embedding(enc_seq, enc_feature)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(dec_in)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        if self.configs.c_out == 1:
            dec_out = dec_out.squeeze(-1)
        return dec_out