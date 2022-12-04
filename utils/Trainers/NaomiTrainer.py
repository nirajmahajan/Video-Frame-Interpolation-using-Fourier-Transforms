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
        self.criterion_reconFT = fetch_loss_function(self.parameters['loss_reconstructed_FT'], self.device, self.parameters['loss_params'])

    def train(self, epoch):
        avglossrecon = 0
        avglossreconft = 0
        beta2 = self.parameters['beta2']
        for i, (indices, data) in tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), position=0, leave = True):
            batch, num_frames, nrows, ncols = data.shape
            for param in self.model.parameters():
                param.grad = None
            self.model.train()
            pred = self.model(data.to(self.device)) # B, 5, 64, 64
            pred = pred[:,1:4,:,:]
            target = data.to(self.device)[:,1:4,:,:]
            loss_recon = self.criterion(pred, target)
            
            loss_reconft = torch.tensor([0]).to(self.device)
            if self.criterion_reconFT is not None:
                if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                    loss_reconft = self.criterion_reconFT(pred, target)
                else:
                    predfft = (torch.fft.fft2(pred)+EPS).log()
                    predfft = torch.stack((predfft.real, predfft.imag),-1)
                    targetfft = (torch.fft.fft2(target) + EPS).log()
                    targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                    loss_reconft = self.criterion_reconFT(predfft, targetfft)
            loss = loss_recon + beta2*loss_reconft

            loss.backward()
            self.optim.step()

            avglossrecon += loss_recon.item()/(len(self.trainloader))
            avglossreconft += loss_reconft.item()/(len(self.trainloader))

        if self.scheduler is not None:
            self.scheduler.step()

        print('Average Recon Loss for Epoch {} = {}' .format(epoch, avglossrecon), flush = True)
        if self.criterion_reconFT is not None:
            print('Average Recon FT Loss for Epoch {} = {}' .format(epoch, avglossreconft), flush = True)
        return avglossrecon, avglossreconft

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
        avglossreconft = 0
        with torch.no_grad():
            for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), position=0, leave = True):
                batch, num_frames, nrows, ncols = data.shape
                self.model.eval()
                pred = self.model(data.to(self.device)) # B, 5, 64, 64
                pred = pred[:,1:4,:,:]
                target = data.to(self.device)[:,1:4,:,:]
                loss_recon = self.criterion(pred, target)
                
                loss_reconft = torch.tensor([0]).to(self.device)
                if self.criterion_reconFT is not None:
                    if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                        loss_reconft = self.criterion_reconFT(pred, target)
                    else:
                        predfft = (torch.fft.fft2(pred)+EPS).log()
                        predfft = torch.stack((predfft.real, predfft.imag),-1)
                        targetfft = (torch.fft.fft2(target) + EPS).log()
                        targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                        loss_reconft = self.criterion_reconFT(predfft, targetfft)

                avglossrecon += loss_recon.item()/(len(self.trainloader))
                avglossreconft += loss_reconft.item()/(len(self.trainloader))

        print('{} Loss After {} Epochs:'.format(dstr, epoch), flush = True)
        print('Recon Loss = {}'.format(avglossrecon), flush = True)
        if self.criterion_reconFT is not None:
            print('Recon FT Loss = {}'.format(avglossreconft), flush = True)
        return avglossrecon, avglossreconft

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
                batch, num_frames, nrows, ncols = data.shape
                self.model.eval()
                pred = self.model(data.to(self.device)) # B, 5, 128
                break
            for pnumi in range(num_plots):
                fig = plt.figure(figsize = (20,10))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # plt.imshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                iter = 1
                for i in range(5):
                    plt.subplot(2,5,iter)
                    iter += 1
                    plt.imshow(data[pnumi,i,:,:].squeeze().detach().cpu().numpy(), cmap = 'gray')
                    plt.title('Original Frame {}'.format(i+1))
                for i in range(5):
                    plt.subplot(2,5,iter)
                    iter += 1
                    plt.imshow(pred[pnumi,i,:,:].squeeze().detach().cpu().numpy(), cmap = 'gray')
                    plt.title('Predicted Frame {}'.format(i+1))
                plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, pnumi)))
                plt.close('all')