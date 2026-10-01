import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s:  %(message)s',
    level=logging.INFO,
     datefmt="%I:%M:%S"
)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cpu'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x,t):
        sqrt_alpha_hat = torch.sqrt(self.aplha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} images")
        model.eval()

        with torch.no_grad():
            x = torch.randn(())