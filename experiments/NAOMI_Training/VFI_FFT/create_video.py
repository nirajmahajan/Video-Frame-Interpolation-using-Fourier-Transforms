from torchvision.io import write_video
from torchvision.io import read_video
import torch
import torchvision

def images2video(images, out_path, fps = 15, video_codec = 'libx264', repeat = 1):
    images = (images*255).type(torch.uint8)
    write_video(out_path, images.unsqueeze(3).repeat(repeat, 1, 1, 3), fps= fps, video_codec = video_codec)

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

EPS = 1e-10

sys.path.append('../../../')
os.environ['display'] = 'localhost:14.0'

from utils.myModels3.NAOMI import NAOMI
from utils.myModels2.NAOMI import Discriminator
from utils.myDatasets.MovingMNIST import MovingMNIST
from utils.Trainers.NaomiTrainer3 import Trainer

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()
seed_torch(args.seed)

parameters = {}
parameters['train_batch_size'] = 150
parameters['test_batch_size'] = 150
parameters['lr'] = 3e-4
parameters['num_epochs'] = 100
parameters['train_test_split'] = 0.8
parameters['dataset_path'] = '../../../datasets/moving_mnist'
parameters['normalisation'] = True
parameters['encoded_space_dim'] = 128
parameters['gru_hidden_dim'] = 256
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'StepLR'
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'step_size': parameters['num_epochs']//3,
    'gamma': 0.5,
    'verbose': True
}
parameters['loss_recon'] = 'L2'
parameters['loss_recon_decoder'] = 'L2'
parameters['loss_reconstructed_FT'] = 'None'
parameters['train_losses'] = []
parameters['test_losses'] = []
parameters['beta1'] = 0
parameters['beta2'] = 0.5
parameters['beta3'] = 0.5
parameters['betaKL'] = 1
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
} 
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['loss_recon']) in ['L1', 'L2', 'SSIM', 'MS_SSIM']
assert(parameters['loss_reconstructed_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None', 'Cosine-Watson']

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 1
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

checkpoint_path = './checkpoints/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir('./results/'):
    os.mkdir('./results/')
if not os.path.isdir('./results/train'):
    os.mkdir('./results/train')
if not os.path.isdir('./results/test'):
    os.mkdir('./results/test')

trainset = MovingMNIST(
                    parameters['dataset_path'], 
                    train = True, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'],
                )
testset = MovingMNIST(
                    parameters['dataset_path'], 
                    train = False, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'],
                )

model = NAOMI(gru_hidden_dim = parameters['gru_hidden_dim'], gru_input_dim = parameters['encoded_space_dim'], num_layers = 1, device = device)
model.to(device)
disc = Discriminator().to(device)
trainer = Trainer(model, disc, trainset, testset, parameters, device)

model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
if (not args.state == -1):
    model_state = args.state
print('Loading checkpoint at model state {}'.format(model_state), flush = True)
dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
pre_e = dic['e']
trainer.model.load_state_dict(dic['model'])
trainer.optim.load_state_dict(dic['optim'])
if parameters['scheduler'] != 'None':
    trainer.scheduler.load_state_dict(dic['scheduler'])
losses = dic['losses']
test_losses = dic['test_losses']
print('Resuming Training after {} epochs'.format(pre_e), flush = True)

data_orig = MovingMNIST.access_data(0)

if not os.path.isdir('out_vids'):
    os.mkdir('out_vids')
images2video(data_orig, 'out_vids/level_0_fps_5.mp4', repeat = 1, fps = 5)

ans = data_orig.clone()
for level in range(1,4):
    data = ans.clone()
    ans = None
    vi = 0
    step = 2
    while 1:
        end = vi+step
        if end >= data.shape[0]:
            break
        pred = torch.zeros(1,5,1,64,64)
        pred[:,0,:,:,:] = data[vi,:,:]
        pred[:,4,:,:,:] = data[end,:,:]
        vi += step
        with torch.no_grad():
            # trainer.model.eval()
            fft_data = (torch.fft.fftshift(torch.fft.fft2(pred.to(device)), dim = (-2,-1))+EPS)
            real = fft_data.real
            imag = fft_data.imag
            mag = (real ** 2 + imag ** 2 + EPS) ** (0.5)
            phase = torch.atan2(imag, real+EPS)
            pred = trainer.model(phase.reshape(phase.shape[0], 5, -1), mode = 0).reshape(pred.shape) # B, 5, 1, 64, 64
            pred = trainer.model(pred.detach().reshape(pred.shape[0], 5, -1), mode = 1).reshape(pred.shape) # B, 5, 1, 64, 64
            real = mag * torch.cos(pred)
            imag = mag * torch.sin(pred)
            pred = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(real, imag), dim = (-2,-1))).real.cpu()
            pred = pred - pred.min()
            pred = pred / pred.max()
            pred = pred.squeeze()
        if ans is None:
            ans = pred.squeeze()
        else:
            ans = torch.cat((ans,pred[1:].squeeze()), 0)
    # for i in range(ans.shape[0]):
    #     plt.imsave('out_vids/img_{}.jpg'.format(i), ans[i,:,:], cmap = 'gray')
    images2video(ans, 'out_vids/level_{}_fps_{}.mp4'.format(level,ans.shape[0]//4), repeat = 1, fps = ans.shape[0]//4)


ans = data_orig.clone()
for level in range(1,4):
    data = ans.clone()
    ans = None
    vi = 0
    step = 3
    while 1:
        end = vi+step
        if end >= data.shape[0]:
            break
        pred = torch.zeros(1,5,1,64,64)
        pred[:,0,:,:,:] = data[vi,:,:]
        pred[:,4,:,:,:] = data[end,:,:]
        vi += step
        with torch.no_grad():
            # trainer.model.eval()
            fft_data = (torch.fft.fftshift(torch.fft.fft2(pred.to(device)), dim = (-2,-1))+EPS)
            real = fft_data.real
            imag = fft_data.imag
            mag = (real ** 2 + imag ** 2 + EPS) ** (0.5)
            phase = torch.atan2(imag, real+EPS)
            pred = trainer.model(phase.reshape(phase.shape[0], 5, -1), mode = 0).reshape(pred.shape) # B, 5, 1, 64, 64
            pred = trainer.model(pred.detach().reshape(pred.shape[0], 5, -1), mode = 1).reshape(pred.shape) # B, 5, 1, 64, 64
            real = mag * torch.cos(pred)
            imag = mag * torch.sin(pred)
            pred = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(real, imag), dim = (-2,-1))).real.cpu()
            pred = pred - pred.min()
            pred = pred / pred.max()
            pred = pred.squeeze()
        if ans is None:
            ans = pred.squeeze()
        else:
            ans = torch.cat((ans,pred[1:].squeeze()), 0)
    # for i in range(ans.shape[0]):
    #     plt.imsave('out_vids/img_{}.jpg'.format(i), ans[i,:,:], cmap = 'gray')
    images2video(ans, 'out_vids/level_{}_fps_{}.mp4'.format(level,ans.shape[0]//4), repeat = 1, fps = ans.shape[0]//4)
