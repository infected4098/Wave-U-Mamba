import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from modules.normalization import LayerNorm
from einops import rearrange
from mamba_ssm import Mamba
import einops
from utils import init_weights, get_padding, summarize_model


LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for i in range(len(dilation)):
            self.convs1.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i],
                               padding=get_padding(kernel_size, dilation[i]))))
            self.convs2.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)


    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class MambaBlock(nn.Module):
    def __init__(self, d_model, cfg_mamba):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = cfg_mamba["d_state"]
        self.d_conv = cfg_mamba["d_conv"]
        self.expand = cfg_mamba["expand"]
        self.mamba = Mamba(d_model = self.d_model, d_state = self.d_state,
                                          d_conv = self.d_conv, expand = self.expand)
    def forward(self, x):
        # Expected Input shape : [B, D, L]
        x = rearrange(x, 'b d l -> b l d')
        x = self.mamba(x)
        x = rearrange(x, 'b l d -> b d l')
        return x


class Mambablocks(nn.Module):
    # Add --> LN --> Mamba
    def __init__(self, out_channels, n_blocks, cfg_mamba):
        super(Mambablocks, self).__init__()
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.cfg_mamba = cfg_mamba

        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                nn.Sequential(LayerNorm(self.out_channels, (0, 2, 1)), MambaBlock(self.out_channels, self.cfg_mamba)))

    def forward(self, x):
        for block in self.blocks:
            resid = x
            x = block(x)
            x = resid + x
        return x

class Downsampleblock(nn.Module):
    # Downsampleblocck
    def __init__(self, in_channels, out_channels, kernel_size, downsample, n_conv, cfg_mamba, n_mambas = 3):
        super().__init__()
        # Hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.n_conv = n_conv
        self.cfg_mamba = cfg_mamba
        self.n_mambas = n_mambas
        self.mambablocks = Mambablocks(self.out_channels, self.n_mambas, cfg_mamba)

        # Modules
        self.convs = nn.ModuleList()
        self.convlns = nn.ModuleList()
        self.init_conv = nn.Conv1d(self.in_channels, self.out_channels,
                                       kernel_size = self.kernel_size, stride = 1, padding = 0)
        self.init_ln = LayerNorm(self.out_channels, (0, 2, 1))

        for i in range(self.n_conv-1):
            self.convs.append(nn.Conv1d(self.out_channels, self.out_channels,
                                       kernel_size = self.kernel_size, stride = 1, padding = 0))
            self.convlns.append(LayerNorm(self.out_channels, (0, 2, 1)))
        self.Avgpool = nn.AvgPool1d(self.downsample, stride = 2, padding = 0)


    def forward(self, x):
        #desired_len = x.size(-1)//2
        x = F.pad(x, (0, self.kernel_size - 1))
        x = self.init_conv(x)
        x = self.init_ln(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        for conv, ln in zip(self.convs, self.convlns):
            resid = x
            x = F.pad(x, (0, self.kernel_size-1))
            x = conv(x)
            x = ln(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = resid + x
        x = self.Avgpool(x) # Exactly halving the sequence
        x = self.mambablocks(x)
        return x


class Bottleneckblock(nn.Module):
    # Bottleneckblock
    # The same structure with Downsampleblock but without pooling.
    def __init__(self, in_channels, out_channels, kernel_size, downsample, n_conv, cfg_mamba, n_mambas = 3):
        super().__init__()
        # Hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.n_conv = n_conv
        self.cfg_mamba = cfg_mamba
        self.n_mambas = n_mambas

        # Modules
        self.convs = nn.ModuleList()
        self.convlns = nn.ModuleList()
        self.init_conv = nn.Conv1d(self.in_channels, self.out_channels,
                                       kernel_size = 3, stride = 1, padding = 0)
        self.init_ln = LayerNorm(self.out_channels, (0, 2, 1))

        for i in range(self.n_conv-1):
            self.convs.append(nn.Conv1d(self.out_channels, self.out_channels,
                                       kernel_size = 3, stride = 1, padding = 0))
            self.convlns.append(LayerNorm(self.out_channels, (0, 2, 1)))

        self.mambablocks = Mambablocks(self.out_channels, 3, cfg_mamba)

    def forward(self, x):
        x = F.pad(x, (0, 2))
        x = self.init_conv(x)
        x = self.init_ln(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        for conv, ln in zip(self.convs, self.convlns):
            resid = x
            x = F.pad(x, (0, 2))
            x = conv(x)
            x = ln(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = resid + x
        x = self.mambablocks(x)

        return x


class Downsamplestem(nn.Module):
    def __init__(self, in_channels,
                 kernel_size, embed_dim, strides, cfg_mamba, n_mambas):
        """
        audio_size = T * sr; temporal_frame_num
        in_channels = 1
        kernel_size = [4, 4, 4, 4]
        embed_dim = [32, 64, 128, 256]
        strides = [2, 2, 2, 2]
        """
        super().__init__()
        # Hyperparameters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.strides = strides
        self.cnn_stem = nn.ModuleList()
        self.ln_stem = nn.ModuleList()
        self.cfg_mamba = cfg_mamba
        self.n_mambas = n_mambas
        self.cnn_stem.append(Downsampleblock(self.in_channels, self.embed_dim[0],
                                                             kernel_size=self.kernel_size[0],
                                                             downsample=self.strides[0], n_conv = 2, cfg_mamba=self.cfg_mamba, n_mambas = self.n_mambas))
        self.ln_stem.append(LayerNorm(self.embed_dim[0], (0, 2, 1)))

        # Modules
        for i in range(1, int(len(kernel_size))):
            self.cnn_stem.append(Downsampleblock(self.embed_dim[i-1], self.embed_dim[i],
                                       kernel_size = self.kernel_size[i-1], downsample = self.strides[i-1], n_conv = 2, cfg_mamba = self.cfg_mamba, n_mambas = self.n_mambas))
            self.ln_stem.append(LayerNorm(self.embed_dim[i], (0, 2, 1)))

        self.cnn_stem.append(Bottleneckblock(self.embed_dim[-1], self.embed_dim[-1], kernel_size = self.kernel_size[-1],
                                       downsample=1, n_conv = 2, cfg_mamba = self.cfg_mamba, n_mambas = self.n_mambas))
        self.ln_stem.append(LayerNorm(self.embed_dim[-1], (0, 2, 1)))


    def forward(self, x): 
        # [batch_size, 1, audio_size]
        res_lst = []
        res_lst.append(x)
        for i, (cnn, ln) in enumerate(zip(self.cnn_stem, self.ln_stem)):
            x = cnn(x)
            res_lst.append(x)
            x = ln(x)

            if i < len(self.cnn_stem) - 1:

                x = F.leaky_relu(x, LRELU_SLOPE)

        return x, res_lst[:-1] # no skip connection for the shortest cut.


class WaveUmamba(nn.Module):
    def __init__(self, cfg_mamba):  # (B, C, T)
        super(WaveUmamba, self).__init__()

        # Downsampling
        self.num_kernels = len([3])
        self.num_upsamples = len([2, 2, 2, 2])
        self.emb = Downsamplestem(in_channels=1, kernel_size=[4, 4, 4, 4],
                                         embed_dim=[32, 64, 128,256],
                                         strides=[2, 2, 2, 2], cfg_mamba=cfg_mamba, n_mambas = 2)
        # Upsampling
        resblock = ResBlock
        # resblock = ResBlock1 if res_type == 1 else ResBlock2
        self.ups = nn.ModuleList()
        self.mambablocks = nn.ModuleList()


        for i, (u, k) in enumerate(zip([2, 2, 2, 2], [4, 4, 8, 8])):
            # https://github.com/jik876/hifi-gan/blob/master/models.py
            self.ups.append(
                weight_norm(ConvTranspose1d(256 // (2 ** i),
                                256 // (2 ** (i + 1)),
                                k, u, padding=(
                                                          k - u) // 2)))

            #ups.append(
            self.mambablocks.append(Mambablocks(256 // (2 ** (i + 1)), 2, cfg_mamba))

        # Resblocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 256 // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip([3],
                                           [[1, 3]])): 
                self.resblocks.append(resblock(ch, k, d))

        # Weight Normalization
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x, res_lst = self.emb(x)  # [B, 1, sr*T] --> [B, embed_dim[-1], sr*T/16]
        res_lst = res_lst[::-1] # Reverse order
        # x = rearrange(x, 'b d l -> b l d')
        # x = self.convblocks(x)
        # x = rearrange(x, 'b l d -> b d l')

        for i in range(self.num_upsamples):

            x = F.leaky_relu(x, LRELU_SLOPE)
            x = x + res_lst[i]
            x = self.ups[i](x)
            x = self.mambablocks[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        #x = x + res_lst[-1]
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = x + res_lst[-1]
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)


