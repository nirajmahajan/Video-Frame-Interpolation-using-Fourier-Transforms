import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets

import sys
sys.path.append('../../')

from utils.functions import fetch_loss_function

EPS = 1e-10

class Trainer(nn.Module):
    def __init__(self, model, trainset, testset, parameters, device):
        super(Trainer, self).__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.parameters = parameters
        self.device = device
        self.trainloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=self.parameters['train_batch_size'], 
                            shuffle = True
                        )
        self.traintestloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=self.parameters['train_batch_size'], 
                            shuffle = False
                        )
        self.testloader = torch.utils.data.DataLoader(
                            testset, 
                            batch_size=self.parameters['test_batch_size'],
                            shuffle = False
                        )

        if self.parameters['optimizer'] == 'Adam':
            self.optim = optim.Adam(self.model.parameters(), lr=self.parameters['lr'], betas=self.parameters['optimizer_params'])
        elif self.parameters['optimizer'] == 'SGD':
            mom, wt_dec = self.parameters['optimizer_params']
            self.optim = optim.SGD(self.model.parameters(), lr=self.parameters['lr'], momentum = mom, weight_decay = wt_dec)
            
        if self.parameters['scheduler'] == 'None':
            self.scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            self.scheduler = optim.lr_scheduler.StepLR(self.optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])

        self.criterion = fetch_loss_function(self.parameters['loss_recon'], self.device, self.parameters['loss_params']).to(self.device)
        self.cosine = nn.CosineSimilarity(dim = 2)
        self.criterion_reconFT = fetch_loss_function(self.parameters['loss_reconstructed_FT'], self.device, self.parameters['loss_params'])
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch):
        avglossrecon = 0
        avglossmu = 0
        avglossreconft = 0
        beta1 = self.parameters['beta1']
        beta2 = self.parameters['beta2']
        betaKL = self.parameters['betaKL']
        for i, (indices, data) in tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), position=0, leave = True):
            batch, num_frames, numr, numc = data.shape
            data_resized = data.resize(batch*num_frames, 1, numr, numc).to(self.device)
            for param in self.model.parameters():
                param.grad = None
            self.model.train()
            mu, logvar, outp = self.model(data_resized) # B, 1, X, Y
            KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_recon = self.criterion(outp, data_resized)
            loss_reconft = torch.tensor([0]).to(self.device)
            if self.criterion_reconFT is not None:
                if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                    loss_reconft = self.criterion_reconFT(outp, data_resized)
                else:
                    predfft = (torch.fft.fft2(outp)+EPS).log()
                    predfft = torch.stack((predfft.real, predfft.imag),-1)
                    targetfft = (torch.fft.fft2(data_resized) + EPS).log()
                    targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                    loss_reconft = self.criterion_reconFT(predfft, targetfft)
            mus = mu.reshape(batch,num_frames, self.parameters['encoded_space_dim']) # 10, 20, 8
            num_mus = mus.shape[1]
            mu_vecs = mus[:,0:num_mus-1,:] - mus[:,1:num_mus,:]
            loss_mu = (1-self.cosine(mu_vecs[:,0:num_mus-2,:], mu_vecs[:,1:num_mus-1,:])).mean()
            loss = ((loss_recon+(beta1*loss_mu))*((20*8000)))  + beta2*loss_reconft + betaKL*KLD_loss

            loss.backward()
            self.optim.step()

            avglossrecon += loss_recon.item()/(len(self.trainloader))
            avglossmu += loss_mu.item()/(len(self.trainloader))
            avglossreconft += loss_reconft.item()/(len(self.trainloader))

        if self.scheduler is not None:
            self.scheduler.step()

        print('Average Recon Loss for Epoch {} = {}' .format(epoch, avglossrecon), flush = True)
        print('Average Latent Loss for Epoch {} = {}' .format(epoch, avglossmu), flush = True)
        if self.criterion_reconFT is not None:
            print('Average Recon FT Loss for Epoch {} = {}' .format(epoch, avglossreconft), flush = True)
        return avglossrecon, avglossmu, avglossreconft

    def evaluate(self, epoch, train = False):
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
        avglossrecon = 0
        avglossmu = 0
        avglossreconft = 0
        with torch.no_grad():
            for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), position=0, leave = True):
                batch, num_frames, numr, numc = data.shape
                data_resized = data.resize(batch*num_frames, 1, numr, numc).to(self.device)
                self.model.eval()
                mu, logvar, outp = self.model(data_resized) # B, 1, X, Y
                loss_recon = self.criterion(outp, data_resized)
                loss_reconft = torch.tensor([0]).to(self.device)
                if self.criterion_reconFT is not None:
                    if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                        loss_reconft = self.criterion_reconFT(outp, data_resized)
                    else:
                        predfft = (torch.fft.fft2(outp) + EPS).log()
                        predfft = torch.stack((predfft.real, predfft.imag),-1)
                        targetfft = (torch.fft.fft2(data_resized) + EPS).log()
                        targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                        loss_reconft = self.criterion_reconFT(predfft, targetfft)
                mus = mu.reshape(batch,num_frames, self.parameters['encoded_space_dim']) # 10, 20, 8
                num_mus = mus.shape[1]
                mu_vecs = mus[:,0:num_mus-1,:] - mus[:,1:num_mus,:]
                loss_mu = (1-self.cosine(mu_vecs[:,0:num_mus-2,:], mu_vecs[:,1:num_mus-1,:])).mean()

                avglossrecon += loss_recon.item()/(len(dloader))
                avglossmu += loss_mu.item()/(len(dloader))
                avglossreconft += loss_reconft.item()/(len(dloader))

        print('{} Loss After {} Epochs:'.format(dstr, epoch), flush = True)
        print('Recon Loss = {}'.format(avglossrecon), flush = True)
        print('Latent Loss = {}'.format(avglossmu), flush = True)
        if self.criterion_reconFT is not None:
            print('Recon FT Loss = {}'.format(avglossreconft), flush = True)
        return avglossrecon, avglossmu, avglossreconft

    def visualise(self, epoch, train = False):
        num_plots = min(self.parameters['test_batch_size'], 10)
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
            path = './results/train'
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            path = './results/test'
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, data) in enumerate(dloader):
                batch, num_frames, numr, numc = data.shape
                data_resized = data.resize(batch*num_frames, 1, numr, numc).to(self.device)
                self.model.eval()
                mu, logvar, outp = self.model(data_resized) # B, 1, X, Y
                break
            for i in range(num_plots):
                targi = data_resized[i].squeeze().cpu().numpy()
                predi = outp[i].squeeze().cpu().numpy()
                # fig = plt.figure(figsize = (8,8))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # plt.imshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                # plt.subplot(2,2,3)
                # plt.imshow(predi, cmap = 'gray')
                # plt.title('Our Predicted Frame')
                # plt.subplot(2,2,4)
                # plt.imshow(targi, cmap = 'gray')
                # plt.title('Actual Frame')
                # plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                # plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                # plt.close('all')
                fig = plt.figure(figsize = (8,4))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # plt.imshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                plt.subplot(1,2,1)
                plt.imshow(predi, cmap = 'gray')
                plt.title('Our Reconstructed Image')
                plt.subplot(1,2,2)
                plt.imshow(targi, cmap = 'gray')
                plt.title('Actual Image')
                plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                plt.close('all')