import os
import sys
import PIL
import time
import scipy
import torch
import random
import pickle
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

import sys
sys.path.append('../../')
from utils.myModels.VAE import VariationalAutoEncoder

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MovingMNIST(Dataset):
    data = None
    encoded_data = None
    memoised = None
    
    @classmethod
    def init_data(cls, path):
        data = np.load(os.path.join(path, 'mnist_test_seq.npy')) # 20,10000,64,64
        cls.data = torch.from_numpy(data).permute((1,0,2,3)).float()/255. # 10000,20,64,64

    @classmethod
    def init_encoded_data(cls, encoded_space_dim):
        cls.encoded_data = torch.zeros(cls.data.shape[0], cls.data.shape[1], encoded_space_dim)
        cls.memoised = [False for i in range(cls.data.shape[0])]

    @classmethod
    def get_encoded_data(cls, i, vae, device):
        if cls.memoised[i]:
            return cls.encoded_data[i]
        else:
            encod,_ = vae.enc(cls.data[i].unsqueeze(1).to(device))
            cls.memoised[i] = True
            cls.encoded_data[i] = encod.cpu().reshape(cls.data.shape[1],-1)
            return cls.encoded_data[i]


    @classmethod
    def access_data(cls, i):
        return cls.data[i,:,:,:]

    @classmethod
    def access_full_data(cls):
        return cls.data

    @classmethod
    def data_length(cls):
        return cls.data.shape[0]

    @classmethod
    def num_frames(cls):
        return cls.data.shape[1]


    def __init__(self, path, train = True, train_split = 0.8, norm = True, encoding = False, vae = None, device = torch.device('cpu')):
        super(MovingMNIST, self).__init__()
        self.path = path
        self.train = train
        self.train_split = train_split
        self.norm = norm
        self.device = device
        self.encoding = encoding
        self.vae = vae
        MovingMNIST.init_data(self.path)
        if self.encoding:
            assert(self.vae is not None)
            MovingMNIST.init_encoded_data(self.vae.encoded_space_dim)

        train_len = int(train_split*MovingMNIST.data_length())
        if self.train:
            self.indices = np.arange(train_len)
        else:
            self.indices = np.arange(train_len, MovingMNIST.data_length())
        
        self.mean = MovingMNIST.access_full_data().mean()
        self.std = MovingMNIST.access_full_data().std()
        self.encod_mean = 0
        self.encod_std = 1


    def __getitem__(self, i):
        # Batch,20,64,64
        fnum = i %15
        index_num = i//15
        data = MovingMNIST.access_data(self.indices[index_num])[fnum:fnum+5,:,:]
        if self.encoding:
            encod = MovingMNIST.get_encoded_data(self.indices[i], self.vae, self.device)
        if self.norm:
            data = (data - self.mean)/self.std
            if self.encoding:
                encod = (encod - self.encod_mean)/self.encod_std
        if self.encoding:
            return i, data, encod
        else:
            return i, data
        
    def __len__(self):
        return self.indices.shape[0] * (MovingMNIST.data.shape[1]-5)

# vae = VariationalAutoEncoder(32).to(torch.device('cuda:0'))
# for child in vae.children():
#     for param in child.parameters():
#             param.requires_grad = False
# vae.load_state_dict(torch.load('../../experiments/AE_weights/checkpoint_0.pth', map_location = torch.device('cuda:0'))['model'])
# a = MovingMNIST('../../datasets/moving_mnist', train = True, encoding = True, vae = vae, norm = False, device = torch.device('cuda:0'))
# sum = 0
# sumsq = 0
# tot = 0
# for i in tqdm(range(len(a))):
#     i, dat, encod = a[i]
#     sum += encod.sum()
#     sumsq += (encod**2).sum()
#     tot += torch.numel(encod)
# print('done')
# a = MovingMNIST('../../datasets/moving_mnist', train = False, encoding = True, vae = vae, norm = False, device = torch.device('cuda:0'))
# for i in tqdm(range(len(a))):
#     i, dat, encod = a[i]
#     sum += encod.sum()
#     sumsq += (encod**2).sum()
#     tot += torch.numel(encod)
# print('done')

# print('Mean = ', sum/tot)
# std = ((sumsq/tot)-((sum/tot)**2))**0.5
# print('std = ', std)

# for i in tqdm(range(len(a))):
#     i, dat, encod = a[i]
# print('done again')