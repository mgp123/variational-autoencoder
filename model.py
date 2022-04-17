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
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            torch.nn.Flatten(),

        )

        self.fc_mu = torch.nn.Linear(4096, self.latent_size)           
        self.fc_sigma = torch.nn.Linear(4096, self.latent_size)           


        self.load_from_file()

    def forward(self, x):
        z = self.model(x)

        mu = self.fc_mu(z)
        sqrt_sigma =self.fc_sigma(z)

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

        self.model = nn.Sequential(
            torch.nn.Linear(self.latent_size, 4096),
            nn.LeakyReLU(),
            torch.nn.Unflatten(1, (64, 8, 8)),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1
            ),
        )
        self.load_from_file()

    def forward(self, z):
        y = self.model(z)
        return y

    def sample(self, n, std=1):
        z = torch.randn((n, self.get_latent_size())).to(device)
        return self.forward(z*std)

    def get_latent_size(self):
        return self.latent_size

    def load_from_file(self):
        if exists(self.weights_file):
            self.load_state_dict(torch.load(self.weights_file))
            print("Loaded decoder")

    def save_to_file(self):
        torch.save(self.state_dict(), self.weights_file)

