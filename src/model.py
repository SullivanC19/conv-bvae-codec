import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), latent_variables=32, ch1=32, ch2=64, fc=256, kernel_size=4):
        super(Encoder, self).__init__()

        # kernel size must be even so that we halve the input size each layer
        assert kernel_size % 2 == 0

        self.latent_variables = latent_variables
        self.fc = fc
        
        stride = 2
        padding = kernel_size // 2 - 1
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0], ch1, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(ch1, ch1, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2, kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        last_conv_size = ch2 * (img_shape[1] // 16) * (img_shape[2] // 16)

        self.fc = nn.Sequential(
            nn.Linear(last_conv_size, fc), 
            nn.ReLU(),
            nn.Linear(fc, self.latent_variables * 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.fc(x)
        mean, log_var = torch.split(x, self.latent_variables, dim=-1)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), latent_variables=32, ch1=32, ch2=64, fc=256, kernel_size=4):
        super(Decoder, self).__init__()

        # kernel size must be even so that we halve the input size each layer
        assert kernel_size % 2 == 0

        stride = 2
        padding = kernel_size // 2 - 1

        self.latent_variables = latent_variables
        self.fc = fc
        self.first_conv_shape = (ch2, img_shape[1] // 16, img_shape[2] // 16)

        first_conv_size = ch2 * (img_shape[1] // 16) * (img_shape[2] // 16)
        self.fc = nn.Sequential(
            nn.Linear(latent_variables, fc),
            nn.ReLU(),
            nn.Linear(fc, first_conv_size),
            nn.ReLU()
        )
 
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ch2, ch2, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(ch2, ch1, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(ch1, ch1, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(ch1, img_shape[0], kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )

    def forward(self, z):
        x_hat = self.fc(z)
        x_hat = torch.reshape(x_hat, (x_hat.size(0), *self.first_conv_shape))
        return self.conv(x_hat)

class ConvBetaVae(nn.Module):
    def __init__(self, latent_variables, beta_value, img_size=64):
        super(ConvBetaVae, self).__init__()
        self.encoder = Encoder(latent_variables=latent_variables, img_shape=(3, img_size, img_size))
        self.decoder = Decoder(latent_variables=latent_variables, img_shape=(3, img_size, img_size))
        self.latent_variables = latent_variables
        self.beta_value = beta_value
        self.img_size = 3 * img_size ** 2
    
    def loss(self, x):        
        z_hat, mean, log_var = self.sample_z(x)
        std = torch.exp(log_var / 2)

        # KL Divergence loss
        element_wise = (-torch.log(std) * 2 + std.pow(2) + mean.pow(2) - 1) / 2
        kl = element_wise.sum(-1).mean()

        # decode using sampled z
        x_hat = self.decoder(z_hat)

        # Gaussian reconstruction loss
        distr = torch.distributions.Normal(x, 1 / math.sqrt(2 * math.pi))
        rec = distr.log_prob(x_hat).view(-1, self.img_size).sum(-1).mean()

        # compute negative estimated lower bound
        return -rec + kl * self.beta_value, -rec, kl
    
    def sample_prior(self, batches):
        return torch.normal(torch.zeros((batches, self.latent_variables)))

    def sample_z(self, x):
        # encode x and extract mean and std for posterior
        mean, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)

        # sample z using reparameterization trick
        eps = self.sample_prior(x.size(0))
        z = mean + eps * std

        return z, mean, log_var

    def forward(self, x):
        mean, _ = self.encoder(x)

        # decode using mean of z distribution
        return self.decoder(mean)