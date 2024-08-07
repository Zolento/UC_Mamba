import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from models.modules.vmamba import *
from models.modules.blocks import *
from timm.models.layers import trunc_normal_
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from utils.utils import kwargs_plainunet as plainunet_kwargs
import fastmri

def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )

def _make_patch_embed_v23D(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 4, 1, 2, 3)),
        nn.GELU(),
        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )
   
def _make_patch_expand(in_chans, out_chans, patch_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
    )
    
def _make_patch_expand3D(in_chans, out_chans, patch_size):
    return nn.Sequential(
        nn.ConvTranspose3d(in_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
        LayerNorm(out_chans, data_format="channels_first"),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(out_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
    )

class VSSM_Decoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 ouput_channels: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool = True
                 ):

        super().__init__()
        stages = []
        transpconvs = []
        seg_layers = []
        if encoder.conv_op == Conv2d_channel_last:
            transpconv_op = Conv2dTran_channel_last
        elif encoder.conv_op == nn.Conv2d:
            transpconv_op = nn.ConvTranspose2d
        elif encoder.conv_op == nn.Conv3d:
            transpconv_op = nn.ConvTranspose3d
        else:
            transpconv_op = Conv3dTran_channel_last
        conv_op = encoder.conv_op
        n_stages_encoder = len(encoder.output_channels)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                in_channels=input_features_below, 
                out_channels=input_features_skip, 
                kernel_size=stride_for_transpconv, 
                stride=stride_for_transpconv,
            )
            )
            stage_modules = []
            stage_modules.append(
                StackedConvBlocks(
                    num_convs=n_conv_per_stage[s-1], 
                    conv_op=conv_op,
                    input_channels=input_features_skip*2, 
                    output_channels=input_features_skip, 
                    kernel_size=3,
                    initial_stride=1,
                    conv_bias=True,
                    norm_op=Instance2d_channel_last,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True},
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU,
                    nonlin_kwargs={'inplace': True},
                    nonlin_first=False))
            stages.append(nn.Sequential(*stage_modules))
            # seg_layers.append(encoder.conv_op(input_features_skip, ouput_channels, 1, 1, 0, bias=True))
            
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        # self.seg_layers = nn.ModuleList(seg_layers)
        self.deep_supervision = deep_supervision
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), -1)
            x = self.stages[s](x)
            # if self.deep_supervision:
            #     seg_outputs.append(self.seg_layers[s](x))
            # elif s == (len(self.stages) - 1):
            #     seg_outputs.append(self.seg_layers[-1](x))
            if self.deep_supervision:
                seg_outputs.append(x)
            elif s == (len(self.stages) - 1):
                seg_outputs.append(x)
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class VSSM_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size:int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 padding: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 return_skips: bool = False
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(padding, int):
            padding = [padding] * n_stages
            
        stages = []
        self.return_skips = return_skips
        self.output_channels = features_per_stage
        self.strides = strides
        self.padding = padding
        if conv_op == nn.Conv2d:
            conv_op = Conv2d_channel_last
        elif conv_op == nn.Conv3d:
            conv_op = Conv3d_channel_last
        
        self.conv_op = conv_op
        if conv_op == nn.Conv2d or conv_op == Conv2d_channel_last:
            stages.append(_make_patch_embed_v2(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
        else:
            stages.append(_make_patch_embed_v23D(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            conv_padding = padding[s]
            if s > 0:
                stage_modules.append(
                    conv_op(
                    in_channels=input_channels,
                    out_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=conv_stride,
                    padding=conv_padding
                    )
                    )
            for _ in range(n_conv_per_stage[s]):
                stage_modules.append(
                VSSBlock(
                    hidden_dim=features_per_stage[s],
                    ssm_init="v2",
                    forward_type="v2_noz"
                )
                )
            input_channels = features_per_stage[s]
            stages.append(nn.Sequential(*stage_modules))
        self.stages = nn.ModuleList(stages)
    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

class VSSMUNet(nn.Module):
    def __init__(self,
                input_channels: int,
                patch_size: int,
                d_model: int,
                n_stages: int,
                conv_op: Type[_ConvNd],
                kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                strides: Union[int, List[int], Tuple[int, ...]],
                padding: Union[int, List[int], Tuple[int, ...]],
                n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                num_output_channels: int,
                UNet_base_num_features: int,
                UNet_max_num_features: int,
                n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                deep_supervision: bool = True,
                out_put: bool = False,
                DC_type: str = 'AM'
                ):
        super().__init__()
        assert DC_type in ('VN', 'AM', '')
        self.DC_type = DC_type
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                UNet_max_num_features) for i in range(n_stages)]
        self.input_channels = input_channels
        num_output_channels = input_channels
        self.d_model = d_model
        if conv_op==nn.Conv2d:
            self.patch_expand = _make_patch_expand(in_chans=d_model,
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        else:
            self.patch_expand = _make_patch_expand3D(in_chans=UNet_base_num_features,
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        self.DC = DC_layer2c(DC_type=DC_type, soft=True)
        self.out_put = out_put
        self.encoder = VSSM_Encoder(input_channels=input_channels,
                                    patch_size=patch_size,
                                    n_stages=n_stages, 
                                    n_conv_per_stage=n_conv_per_stage,
                                    features_per_stage=features_per_stage, 
                                    conv_op=conv_op,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    padding=padding,
                                    return_skips=True)
        self.decoder = VSSM_Decoder(self.encoder, 
                                    num_output_channels, 
                                    n_conv_per_stage_decoder, 
                                    deep_supervision)
    
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std
    
    def unnorm(self,
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def forward(self, uk, mask, uk0):
        x = ifft2c(uk, dim=(-2, -1))
        B, C, h, w, _ = x.shape
        x = x.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
        x, mean, std = self.norm(x)
        skips = self.encoder(x)
        out_put = self.decoder(skips)[0]
        out_put = self.patch_expand(out_put.permute(0,3,1,2))
        out_put = self.unnorm(out_put, mean, std)
        i_rec2 = out_put.reshape(B, 2, C, h, w).permute(0,2,3,4,1)
        Gk = fft2c(i_rec2, dim=(-2, -1))
        if self.DC_type == 'VN':
            k_out = self.DC(uk, mask, uk0) - Gk
        elif self.DC_type == 'AM' or self.DC_type == '':
            k_out = self.DC(uk + Gk, mask, uk0)
        if self.out_put:
            i_rec2 = ifft2c(k_out, dim=(-2, -1))
            i_rec = fastmri.complex_abs(i_rec2).squeeze(1)
            i_rec2 = i_rec2.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
            return [i_rec, i_rec2]
        else:
            return k_out

class VSSMUNet_unrolled(nn.Module):
    def __init__(self, iter=6, DC_type='VN', kwargs=None):
        super().__init__()
        self.layers = []
        for _ in range(iter-1):
            self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type))
        self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type, out_put=True))
        self.layers = nn.ModuleList(self.layers)
        self.refine = PlainConvUNet(**plainunet_kwargs)
        self.apply(self._init_weights)
        
    def forward(self, x, mask, dorefine=False):
        uk0 = x.clone()
        for layer in self.layers:
            x = layer(x, mask, uk0)
        if not dorefine:
            return x[0]
        else:
            i_recr = self.refine(x[1])
            i_rec = (i_recr**2).sum(dim=1).sqrt()
            return i_rec
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=1e-2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, Instance2d_channel_last)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d, Conv2d_channel_last, Conv2dTran_channel_last)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            trunc_normal_(m.weight, std=1e-2)
            
class DC_layer2c(nn.Module):
    def __init__(self, DC_type, soft=False):
        super(DC_layer2c, self).__init__()
        self.soft = soft
        if self.soft:
            if DC_type != '':
                self.dc_weight = nn.Parameter(torch.Tensor([1]))
            else:
                self.dc_weight = None
    def forward(self, k_rec, mask, k_ref):
        if len(mask.shape) < len(k_rec.shape):
            mask = mask.unsqueeze(-1)
        masknot = 1 - mask
        if self.soft:
            k_out = masknot * k_rec + mask * k_rec * (1 - self.dc_weight) + mask * k_ref * self.dc_weight
        else:
            k_out = masknot * k_rec + mask * k_rec
        return k_out

