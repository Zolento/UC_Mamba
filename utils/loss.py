import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_metrics2c(out, label):
    psnr_ = 0
    ssim_ = 0
    out = out.contiguous().detach().cpu().numpy()
    label = label.contiguous().detach().cpu().numpy()
    assert label.shape == label.shape, 'tensor size inconsistency'
    B = out.shape[0]
    for i in range(B):
        x = out[i,...]
        y = label[i,...]
        psnr_ += psnr(x, y)
        ssim_ += ssim(x, y, data_range=1.0, win_size=11)
    return psnr_ / B, ssim_ / B
            
def compute_metrics2c_full(X:list, Y:list):
    PSNR_list = []
    SSIM_list = []
    for i in range(len(X)):
        x = X[i].detach().cpu().numpy()
        y = Y[i].detach().cpu().numpy()
        assert X[i].shape == Y[i].shape, 'tensor size inconsistency'
        PSNR_list.append(psnr(x, y))
        SSIM_list.append(ssim(x, y, data_range=1.0, win_size=11))
    return PSNR_list, SSIM_list

def psnr(img1:torch.Tensor, img2:torch.Tensor):
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return np.inf
    else:
        return 10*np.log10(1.0/mse)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self, out, label
    ):
        l = torch.norm((out - label),'fro') / torch.norm(label,'fro')
        return l
