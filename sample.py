from model import Decoder, Encoder, device
import torchvision
import torch
import matplotlib.pyplot as plt
import sys


decoder = Decoder()
decoder = decoder.to(device)
decoder.eval()

n = int(sys.argv[1])
std = float(sys.argv[2])

with torch.no_grad():
    images = decoder.sample(n,std)
    images = images.to("cpu")
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("samples_std_"+ str(std)+".png", bbox_inches='tight')
