import torch
import torchvision
from tqdm import tqdm
import os

from data_loader import get_data_loaders, batch_size
from model import Encoder, Decoder
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("weights", exist_ok=True)

    data_loader_train, data_loader_test = get_data_loaders()

    encoder = Encoder()
    encoder = encoder.to(device)
    decoder = Decoder()
    decoder = decoder.to(device)

    # 3e-4 seems to be a good value for lr
    lr = 3e-4

    normalize_reconstruction_weight = 0.9999
    normalize_kl_weight = 1- normalize_reconstruction_weight


    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    mse = torch.nn.MSELoss()
    epochs = 250
    epochs_per_save = 5

    writer = SummaryWriter()
    training_batch = 0
    samples_per_write = 6400
    
    for epoch in tqdm(range(epochs)):

        for dataset_samples, _ in tqdm(data_loader_train, leave=False):

            dataset_samples = dataset_samples.to(device)
            latent_samples, mu, sigma = encoder(dataset_samples)
            generator_samples = decoder(latent_samples)

            square_error_term = mse(generator_samples, dataset_samples) * normalize_reconstruction_weight
            kl_div_term = torch.sum(mu**2) + torch.sum(sigma) - torch.sum(torch.log(sigma))
            kl_div_term *= 1/batch_size
            loss = normalize_reconstruction_weight*square_error_term + normalize_kl_weight*kl_div_term

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            training_batch += 1

            if training_batch*batch_size % samples_per_write == 0:

                writer.add_scalar("loss/samples", loss.item(), training_batch*batch_size)
                writer.add_scalar("square_error_term/samples", square_error_term.item(), training_batch*batch_size)
                writer.add_scalar("kl_div_term/samples", kl_div_term.item(), training_batch*batch_size)



        if epoch % epochs_per_save == epochs_per_save - 1:

            encoder.save_to_file()
            decoder.save_to_file()

            with torch.no_grad():
                img = decoder.sample(8)
                img_grid = torchvision.utils.make_grid(img, nrow=2)
                writer.add_image("generator sample epoch " + str(epoch + 1), img_grid)

                for test_samples, _ in data_loader_test:
                    test_samples = test_samples.to(device)
                    reconstructed_samples = decoder(encoder(test_samples)[1])
                    img = torch.cat((test_samples, reconstructed_samples), 0)
                    img_grid = torchvision.utils.make_grid(img, nrow=4)
                    writer.add_image("reconstructed sample epoch " + str(epoch + 1), img_grid)

    writer.flush()
    writer.close()