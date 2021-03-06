## Variational Autoencoder

A very basic variational autoencoder for 128x128 images.

The architecture looks like this:

<p align="center">
    <img src="images/architecture.svg">
</p>

### Training

    python3 train.py

(you also need a couple of libraries)

It should take around 6 hours of training depending on the GPU.

### Sampling 

    python3 sample.py <n> <std>

where ```n``` is the number of samples and ```std``` a number between 0 and 1 that is going to multiply the latent variable during sampling. 

A small ```std``` leads to samples closer to the mean and therefore more "conservative".

### Some samples

Here are some samples with different ```std```s trained on FFHQ.


- ```std```=1

<p align="center">
    <img src="images/samples_std_1.0.png">
</p>

- ```std```=0.85

<p align="center">
    <img src="images/samples_std_0.85.png">
</p>



- ```std```=0.4

<p align="center">
    <img src="images/samples_std_0.4.png">
</p>