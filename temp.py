import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Decoder, Encoder, device


decoder = Decoder()
decoder = decoder.to(device)
decoder.eval()

encoder = Encoder()
encoder = encoder.to(device)

image = torch.randn((1,3,128,128))
image = image.to(device)

# print(decoder(encoder(image)[1]).shape)


with torch.no_grad():
    std = 0.7
    images = decoder.sample(16,std)
    images = images.to("cpu")
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("samples_std_"+ str(std)+".png", bbox_inches='tight')
