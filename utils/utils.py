import torch.nn as nn
import torch
import time

SEED = 1958
input_channels = 2
d_model = 32
n_stages = 5
n_stages_vssm_unrolled = 5
d_model1 = 32
kernel_sizes = 3

kwargs_vssm_unrolled = {
    'input_channels': 2,
    'patch_size': 4,
    'd_model': d_model1,
    'UNet_base_num_features': d_model1,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages_vssm_unrolled,
    'kernel_sizes': 3,
    'strides': [2 if i>0 else 1 for i in range(n_stages_vssm_unrolled)],
    'padding': 1,
    'num_output_channels': 1,
    'conv_op': nn.Conv2d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True
}

conv_op = nn.Conv2d
if conv_op == nn.Conv3d:
    norm_op = nn.InstanceNorm3d
    padding = [(1, (kernel_sizes-1)//2, (kernel_sizes-1)//2) if i>0 else (1, 1, 1) for i in range(n_stages)]
    stride = [(1, 2, 2) if i>0 else (1, 1, 1) for i in range(n_stages)]
else:
    norm_op = nn.InstanceNorm2d
    padding = [(kernel_sizes-1)//2 if i>0 else 1 for i in range(n_stages)]
    stride = [2 if i>0 else 1 for i in range(n_stages)]

kwargs_plainunet = {
        'input_channels': input_channels,
        'features_per_stage': d_model,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': stride,
        'num_classes': input_channels,
        'conv_bias': True,
        'norm_op': norm_op,
        'conv_op': conv_op,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': False
}

def collate(data):
    return data

def collate_batch(data):
    uk = []
    label = []
    mask = []
    for list in data:
        uk.append(list[0])
        label.append(list[1])
        mask.append(list[2])
    return [torch.stack(uk), torch.stack(label), torch.stack(mask)]

def get_time():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def print_to_log_file(log_file, *args, also_print_to_console=True):
    with open(log_file, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write(" ")
        f.write("\n")
    if also_print_to_console:
        print(*args)

def adjust_learning_rate(opt, epo, max_steps, initial_lr):
    exponent = 0.9
    new_lr = initial_lr * (1 - epo / max_steps) ** exponent
    for param_group in opt.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def fft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def ifft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data
