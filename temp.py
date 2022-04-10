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

encoder = Encoder()
encoder = encoder.to(device)

image = torch.randn((1,3,128,128))
image = image.to(device)

print(decoder(encoder(image)[1]).shape)