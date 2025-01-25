import torch
import torch.nn as nn
import torch.nn.functional as F

class VAETDecoder(nn.Module):
    def __init__(self, vae_encoder, vae_decoder, norm_layer=None, projection=None):
        super(VAETDecoder, self).__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.norm = norm_layer
        self.projection = projection
        pass

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        mu, logvar = self.vae_encoder(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        return mu, logvar

    def decode(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = self.vae_decoder(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        mu, logvar = self.encode(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        return x

    def predict(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        mu, logvar = self.encode(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        z = mu
        x = self.decode(z, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        return x

class VAETDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(VAETDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.mu_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.mu_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.logvar_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.logvar_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mu_norm = nn.LayerNorm(d_model)
        self.logvar_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)

        mu = self.dropout(self.activation(self.mu_conv1(y.transpose(-1, 1))))
        mu = self.dropout(self.mu_conv2(mu).transpose(-1, 1))

        logvar = self.dropout(self.activation(self.logvar_conv1(y.transpose(-1, 1))))
        logvar = self.dropout(self.logvar_conv2(logvar).transpose(-1, 1))

        return self.mu_norm(mu + y), self.logvar_norm(logvar)