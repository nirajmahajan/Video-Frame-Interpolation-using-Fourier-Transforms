import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim = 16):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        self.enc = Encoder(self.encoded_space_dim, -1)
        self.dec = Decoder(self.encoded_space_dim, -1)

    def forward(self, x):
        mu,log_var = self.enc(x)
        return mu, log_var, self.dec(self.reparameterize(mu,log_var))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 48, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 99, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(99),
            nn.Conv2d(99, 192, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 32, 1, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_mu = nn.Sequential(
            nn.Linear(4 * 4 * 32, encoded_space_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(4 * 4 * 32, encoded_space_dim),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 4 * 4 * 32),
            nn.LeakyReLU(0.2),
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 192, 1, stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(192),
            nn.ConvTranspose2d(192, 192, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(192),
            nn.ConvTranspose2d(192, 96, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.ConvTranspose2d(48, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

# class Encoder(nn.Module):
    
#     def __init__(self, encoded_space_dim,fc2_input_dim):
#         super().__init__()
        
#         ### Convolutional section
#         self.encoder_cnn = nn.Sequential(
#             nn.Conv2d(1, 8, 3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, 3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),
#             nn.ReLU(True)
#         )
        
#         ### Flatten layer
#         self.flatten = nn.Flatten(start_dim=1)
# ### Linear section
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(4 * 4 * 32, 128),
#             nn.ReLU(True),
#             nn.Linear(128, encoded_space_dim)
#         )
        
#     def forward(self, x):
#         x = self.encoder_cnn(x)
#         x = self.flatten(x)
#         x = self.encoder_lin(x)
#         return x
# class Decoder(nn.Module):
    
#     def __init__(self, encoded_space_dim,fc2_input_dim):
#         super().__init__()
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 4 * 4 * 32),
#             nn.ReLU(True)
#         )

#         self.unflatten = nn.Unflatten(dim=1, 
#         unflattened_size=(32, 8, 8))

#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
#         x = self.decoder_lin(x)
#         x = self.unflatten(x)
#         x = self.decoder_conv(x)
#         return x


# enc = Encoder(16,-1)
# dec = Decoder(16,-1)
# vae = VariationalAutoEncoder(16)
# print(enc.encoder_cnn(torch.zeros(10,1,64,64)).shape)
# print(enc(torch.zeros(10,1,64,64))[1].shape)
# print(vae(torch.zeros(10,1,64,64))[0].shape)
# print(vae(torch.zeros(10,1,64,64))[1].shape)
# print(vae(torch.zeros(10,1,64,64))[2].shape)