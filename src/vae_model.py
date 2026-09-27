import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm_new(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super(Norm_new, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            x = self.norm(x)
            x = x.squeeze(-1)
        elif x.dim() == 4:
            x = self.norm(x)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Norm_new(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            Norm_new(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class PlainEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=8, feats=[32, 64, 128, 256]):
        super(PlainEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, feats[0], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(feats[0], feats[1], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(feats[1], feats[2], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[2]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(feats[2], feats[3], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[3]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten_dim = feats[3] * 8 * 20

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            Norm_new(512),
            nn.GELU(),
            nn.Linear(512, 128),
            Norm_new(128),
            nn.GELU(),
            nn.Linear(128, 64),
            Norm_new(64),
            nn.GELU(),
            nn.Linear(64, 2 * latent_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        z = self.fc_layers(x)
        mean, log_var = torch.split(z, self.latent_dim, dim=-1)
        return mean, log_var


class PlainDecoder(nn.Module):
    def __init__(self, output_channels=3, latent_dim=8, feats=[256, 128, 64, 32]):
        super(PlainDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.feats = feats

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            Norm_new(64),
            nn.GELU(),
            nn.Linear(64, 128),
            Norm_new(128),
            nn.GELU(),
            nn.Linear(128, 512),
            Norm_new(512),
            nn.GELU(),
            nn.Linear(512, feats[0] * 8 * 20),
            nn.GELU(),
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(feats[0], feats[1], kernel_size=2, stride=2),
            Norm_new(feats[1]),
            nn.GELU(),

            nn.ConvTranspose2d(feats[1], feats[2], kernel_size=2, stride=2),
            Norm_new(feats[2]),
            nn.GELU(),

            nn.ConvTranspose2d(feats[2], feats[3], kernel_size=2, stride=2),
            Norm_new(feats[3]),
            nn.GELU(),

            nn.ConvTranspose2d(feats[3], output_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc_layers(z)
        z = z.view(z.size(0), self.feats[0], 8, 20)
        return self.deconv_layers(z)


class ResidualEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=8, feats=[32, 64, 128, 256]):
        super(ResidualEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, feats[0], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(feats[0]),

            nn.Conv2d(feats[0], feats[1], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(feats[1]),

            nn.Conv2d(feats[1], feats[2], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[2]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(feats[2]),

            nn.Conv2d(feats[2], feats[3], kernel_size=2, stride=1, padding='same'),
            Norm_new(feats[3]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(feats[3]),
        )

        self.flatten_dim = feats[3] * 8 * 20

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            Norm_new(512),
            nn.GELU(),
            nn.Linear(512, 128),
            Norm_new(128),
            nn.GELU(),
            nn.Linear(128, 64),
            Norm_new(64),
            nn.GELU(),
            nn.Linear(64, 2 * latent_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        z = self.fc_layers(x)
        mean, log_var = torch.split(z, self.latent_dim, dim=-1)
        return mean, log_var


class ResidualDecoder(nn.Module):
    def __init__(self, output_channels=3, latent_dim=8, feats=[256, 128, 64, 32]):
        super(ResidualDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.feats = feats

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            Norm_new(64),
            nn.GELU(),
            nn.Linear(64, 128),
            Norm_new(128),
            nn.GELU(),
            nn.Linear(128, 512),
            Norm_new(512),
            nn.GELU(),
            nn.Linear(512, feats[0] * 8 * 20),
            nn.GELU(),
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(feats[0], feats[1], kernel_size=2, stride=2),
            Norm_new(feats[1]),
            nn.GELU(),
            ResidualBlock(feats[1]),

            nn.ConvTranspose2d(feats[1], feats[2], kernel_size=2, stride=2),
            Norm_new(feats[2]),
            nn.GELU(),
            ResidualBlock(feats[2]),

            nn.ConvTranspose2d(feats[2], feats[3], kernel_size=2, stride=2),
            Norm_new(feats[3]),
            nn.GELU(),
            ResidualBlock(feats[3]),

            nn.ConvTranspose2d(feats[3], output_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc_layers(z)
        z = z.view(z.size(0), self.feats[0], 8, 20)
        return self.deconv_layers(z)


class PlainVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=8):
        super(PlainVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = PlainEncoder(input_channels=input_channels, latent_dim=latent_dim)
        self.decoder = PlainDecoder(output_channels=input_channels, latent_dim=latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    def loss_function(self, x, x_hat, mean, log_var, beta=1.0):
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def encode(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            mean, _ = self.encoder(x)
        if was_training:
            self.train()
        return mean


class ResidualVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=8):
        super(ResidualVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ResidualEncoder(input_channels=input_channels, latent_dim=latent_dim)
        self.decoder = ResidualDecoder(output_channels=input_channels, latent_dim=latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    def loss_function(self, x, x_hat, mean, log_var, beta=1.0):
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def encode(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            mean, _ = self.encoder(x)
        if was_training:
            self.train()
        return mean
