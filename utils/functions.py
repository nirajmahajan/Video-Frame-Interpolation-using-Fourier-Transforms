import torch
import torchvision
import os
import PIL
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
from torch import nn, optim
import utils
from utils.watsonLoss.watson_fft import WatsonDistanceFft
from utils.watsonLoss.shift_wrapper import ShiftWrapper
from utils.watsonLoss.color_wrapper import ColorWrapper
from utils.complexCNNs.polar_transforms import (
    convert_cylindrical_to_polar,
    convert_polar_to_cylindrical,
)

EPS = 1e-10

def fetch_loss_function(loss_str, device, loss_params):
    def load_state_dict(filename):
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, 'watsonLoss/weights', filename)
        return torch.load(path, map_location='cpu')
    if loss_str == 'None':
        return None
    elif loss_str == 'Cosine':
        return lambda x,y: torch.acos(torch.cos(x - y)*(1 - EPS*10**3)).mean()
    elif loss_str == 'L1':
        return nn.L1Loss().to(device)
    elif loss_str == 'L2':
        return nn.MSELoss().to(device)
    elif loss_str == 'Cosine-Watson':
        reduction = 'mean'
        if loss_params['grayscale']:
            if loss_params['deterministic']:
                loss = WatsonDistanceFft(reduction=reduction).to(device)
                if loss_params['watson_pretrained']: 
                    loss.load_state_dict(load_state_dict('gray_watson_fft_trial0.pth'))
            else:
                loss = ShiftWrapper(WatsonDistanceFft, (), {'reduction': reduction}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.loss.load_state_dict(load_state_dict('gray_watson_fft_trial0.pth'))
        else:
            if loss_params['deterministic']:
                loss = ColorWrapper(WatsonDistanceFft, (), {'reduction': reduction}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.load_state_dict(load_state_dict('rgb_watson_fft_trial0.pth'))
            else:
                loss = ShiftWrapper(ColorWrapper, (WatsonDistanceFft, (), {'reduction': reduction}), {}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.loss.load_state_dict(load_state_dict('rgb_watson_fft_trial0.pth'))
        if loss_params['watson_pretrained']: 
            for param in loss.parameters():
                param.requires_grad = False
        return loss
    elif loss_str == 'SSIM':
        return kornia.losses.SSIMLoss(loss_params['SSIM_window']).to(device)
    elif loss_str == 'MS_SSIM':
        return kornia.losses.MS_SSIMLoss().to(device)
    elif loss_str == 'Cosine-L1':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L1', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-L2':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L2', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-MS_SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('MS_SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )

def dual_fft_loss(phase_loss, amp_loss, alpha_phase = 1, alpha_amp = 1):
    def apply_func(x1,x2):
        real1, imag1 = torch.unbind(x1, -1)
        real2, imag2 = torch.unbind(x2, -1)
        amp1,phase1 = convert_cylindrical_to_polar(real1, imag1)
        amp2,phase2 = convert_cylindrical_to_polar(real2, imag2)
        return phase_loss(phase1, phase2)*alpha_phase + amp_loss(amp1, amp2)*alpha_amp
    return apply_func
