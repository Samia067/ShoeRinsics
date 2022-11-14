import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

## channels : list of ints
## kernel_size : int
## padding : int
## stride_fn : fn(channel_index) --> int
## mult=1 if encoder, skip connection count if decoder
def build_encoder(channels, kernel_size, padding, stride_fn, mult=1, conv_times=2):
    layers = []
    sys.stdout.write( '%3d' % channels[0] )
    if conv_times == 2:
        max_channel = max(channels)
    for ind in range(len(channels)-1):
        m = 1 if ind == 0 else mult
        in_channels = int(channels[ind] * m)
        out_channels = int(channels[ind+1])

        stride = stride_fn(ind)
        sys.stdout.write( ' --> %3d' % out_channels )

        if conv_times == 2:
            min_channel = min(in_channels, out_channels)
        if ind < len(channels)-2 :
            if conv_times == 1 or ind == 0:
                block = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=False)
            else:
                if min_channel != max_channel:
                    block = nn.Sequential(Conv(in_channels, min_channel, kernel_size=kernel_size, stride=stride),
                                          Conv(min_channel, out_channels, kernel_size=kernel_size, stride=1, padding=padding, activation=False))
                else:
                    block = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=False)
        else:
            block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        layers.append(block)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return nn.ModuleList(layers)


class Conv(nn.Module):
    '''
    if activation: Conv => BN => LeakyReLU
    else: Conv => BN
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    '''(Conv => BN => LeakyReLU) * 2'''
    def __init__(self, in_ch, out_ch, inter_ch=None, kernel_size=3, stride=1):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2
        if inter_ch is None:
            inter_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(inter_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResConv(nn.Module):
    '''(LeakyRelu => BN => Conv) * 2'''

    def __init__(self, in_ch, out_ch, inter_ch=None, kernel_size=3, conv_times=2):
        super(ResConv, self).__init__()
        padding = kernel_size // 2
        if inter_ch is None:
            inter_ch = out_ch
        if conv_times == 1:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
            )
        else:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_ch, inter_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(inter_ch),
                nn.LeakyReLU(0.1),
                nn.Conv2d(inter_ch, out_ch, kernel_size, padding=padding)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, inter_ch=None, conv_times=2):
        super(Inconv, self).__init__()
        if conv_times == 1:
            self.conv = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        else:
            self.conv = DoubleConv(in_ch, out_ch, inter_ch=inter_ch, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, maxpoolkernel=2, inter_ch=None, conv_times=2):
        super(down, self).__init__()
        if conv_times ==1 :
            self.mpconv = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=2)

        else:
            self.mpconv = nn.Sequential(
                # nn.MaxPool2d(maxpoolkernel),
                DoubleConv(in_ch, out_ch, kernel_size=kernel_size, inter_ch=inter_ch, stride=2)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, inter_ch=None, scale_factor=2, bilinear=True, kernel_size=3, size=None, conv_times=2):
        super(up, self).__init__()
        if inter_ch is None:
            inter_ch = out_ch
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, size=size, mode='bilinear', align_corners=True)
            # self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if conv_times == 1:
            self.conv = Conv(in_ch, out_ch, kernel_size=kernel_size)
        else:
            self.conv = DoubleConv(in_ch, out_ch, inter_ch=inter_ch, kernel_size=kernel_size)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x1 = self.up(x1)
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = self.up(x1)
        x = self.conv(x)
        return x




class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class LightDecoder(nn.Module):
    def __init__(self, h, w, input_channel, n_i_c=1024):
        super(LightDecoder, self).__init__()
        self.h = h
        self.w = w

        self.input = input_channel
        self.c = self.h * self.w
        self.n_i_c = n_i_c
        self.kernel_size = 1

        self.conv = nn.Sequential(
                nn.Conv2d(self.input, self.n_i_c, (self.kernel_size, self.kernel_size)),
                nn.BatchNorm2d(self.n_i_c),
                nn.ReLU(),
                nn.Conv2d(self.n_i_c, self.n_i_c, (self.kernel_size, self.kernel_size)),
                nn.BatchNorm2d(self.n_i_c),
                nn.ReLU(),
                nn.Conv2d(self.n_i_c, self.c, (self.kernel_size, self.kernel_size)))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.shape[0], 1, self.w, self.h))
        return x

class LightEncoder(nn.Module):
    def __init__(self, h, w, output_channel, n_i_c=1024):
        super(LightEncoder, self).__init__()
        self.h = h
        self.w = w

        self.output = output_channel
        self.c = self.h * self.w
        self.n_i_c = n_i_c
        self.kernel_size = 1

        self.conv = nn.Sequential(
                nn.Conv2d(self.c, self.n_i_c, (self.kernel_size, self.kernel_size)),
                nn.BatchNorm2d(self.n_i_c),
                nn.ReLU(),
                nn.Conv2d(self.n_i_c, self.n_i_c, (self.kernel_size, self.kernel_size)),
                nn.BatchNorm2d(self.n_i_c),
                nn.ReLU(),
                nn.Conv2d(self.n_i_c, self.output, (self.kernel_size, self.kernel_size)))

    def forward(self, x):
        x = x.reshape((x.shape[0], self.w * self.h, 1, 1))
        x = self.conv(x)
        return x