import torch
import torch.nn as nn
from os.path import exists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, weights_file="weights/encoder.state"):
        super(Encoder, self).__init__()
        self.weights_file = weights_file
        self.latent_size = 1024

        # the encoder uses a diagonal sigma to reduce parameters
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=self.latent_size//128,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(self.latent_size*2, self.latent_size*2),

        )

        self.load_from_file()

    def forward(self, x):
        encoded = self.model(x)

        latent_size = self.get_latent_size()
        mu, sqrt_sigma = encoded[:, :latent_size], encoded[:, latent_size:]
        sigma = sqrt_sigma*sqrt_sigma
        noise = torch.randn(mu.shape).to(device)
        return mu + noise * sigma, mu, sigma

    def get_latent_size(self):
        return self.latent_size

    def load_from_file(self):
        if exists(self.weights_file):
            self.load_state_dict(torch.load(self.weights_file))
            print("Loaded encoder")

    def save_to_file(self):
        torch.save(self.state_dict(), self.weights_file)


class Decoder(nn.Module):
    def __init__(self, weights_file="weights/decoder.state"):
        super(Decoder, self).__init__()
        self.weights_file = weights_file
        self.latent_size = 1024

        # 0.05 seems to be a good value for sigma
        self.sigma = 0.05

        self.model = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.latent_size),
            nn.LeakyReLU(),
            torch.nn.Unflatten(1, (self.latent_size//256, 16, 16)),
            nn.ConvTranspose2d(
                in_channels=self.latent_size//(256),
                out_channels=32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1
            ),
        )
        self.load_from_file()

    def forward(self, z):
        y = self.model(z)
        # noise = torch.randn(y.shape).to(device)
        # y += noise * self.sigma
        return y

    def sample(self, n):
        z = torch.randn((n, self.get_latent_size())).to(device)
        return self.forward(z)

    def get_latent_size(self):
        return self.latent_size

    def load_from_file(self):
        if exists(self.weights_file):
            self.load_state_dict(torch.load(self.weights_file))
            print("Loaded decoder")

    def save_to_file(self):
        torch.save(self.state_dict(), self.weights_file)
