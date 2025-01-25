import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import DecoderLayer, Encoder, EncoderLayer
from layers.VAETransformer_Dec import VAETDecoder, VAETDecoderLayer
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
                        FullAttention(mask_flag=False, attention_dropout=dropout,
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
        self.decoder = VAETDecoder(
            vae_encoder=VAETDecoderLayer(
                            # self attention
                            AttentionLayer(
                                FullAttention(mask_flag=True, scale=None, attention_dropout=dropout,
                                                output_attention=False),
                                d_model, n_heads),
                            # cross attention
                            AttentionLayer(
                                FullAttention(mask_flag=False, scale=None, attention_dropout=dropout,
                                                output_attention=False),
                                d_model, n_heads),
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout,
                            activation=activation,
                        ),
            vae_decoder=DecoderLayer(
                            # self attention
                            AttentionLayer(
                                FullAttention(mask_flag=True, scale=None, attention_dropout=dropout,
                                                output_attention=False),
                                d_model, n_heads),
                            # cross attention
                            AttentionLayer(
                                FullAttention(mask_flag=False, scale=None, attention_dropout=dropout,
                                                output_attention=False),
                                d_model, n_heads),
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout,
                            activation=activation,
                        ),
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def encode(self, enc_seq, enc_feature):
        enc_out = self.enc_embedding(enc_seq, enc_feature)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def decode(self, enc_out, dec_in):
        dec_out = self.dec_embedding(dec_in)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        if self.configs.c_out == 1:
            dec_out = dec_out.squeeze(-1)
        return dec_out

    def forward(self, enc_seq, enc_feature, dec_in):
        enc_out = self.encode(enc_seq, enc_feature)

        dec_out = self.decode(enc_out, dec_in)
        return dec_out

    def predict(self, enc_seq, enc_feature, dec_in):
        enc_out = self.encode(enc_seq, enc_feature)

        dec_out = self.dec_embedding(dec_in)
        dec_out = self.decoder.predict(dec_out, enc_out, x_mask=None, cross_mask=None)
        if self.configs.c_out == 1:
            dec_out = dec_out.squeeze(-1)
        return dec_out
