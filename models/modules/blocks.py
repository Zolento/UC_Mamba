import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)
    
class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if len(x.shape) == 4:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
            elif len(x.shape) == 5:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                return x

class Conv2d_channel_last(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)

class Conv3d_channel_last(nn.Conv3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Conv2dTran_channel_last(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv_transpose2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)
    
class Conv3dTran_channel_last(nn.ConvTranspose3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv_transpose3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Instance2d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,3,1,2)).permute(0,2,3,1)
    
class Instance3d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
    
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
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)