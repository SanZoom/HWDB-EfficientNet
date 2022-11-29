import copy
import math

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Callable
from torchvision.ops import StochasticDepth


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class SqueezeExcitiation(nn.Module):
    def __init__(self, input_channels, expand_channels, reduction=4):
        super(SqueezeExcitiation, self).__init__()
        squeez_channels = input_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(expand_channels, squeez_channels, bias=True),
            nn.Linear(squeez_channels, expand_channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的

        y = x * y.expand_as(x)

        return y

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels=3, input_size=64):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * ((input_size//8)**2), 32 * ((input_size//8)**2) // 8),
            nn.ReLU(True),
            nn.Linear(32 * ((input_size//8)**2) // 8, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_channels),
                                               activation_layer(inplace=True))


class MBConv(nn.Module):
    def __init__(self, input_channels, output_channels, n, kernel_size, stride, drop_connect_rate):
        super(MBConv, self).__init__()
        self.net = nn.Sequential()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels

        if n > 1:
            self.net.append(ConvBNActivation(input_channels, n * input_channels, kernel_size=1))
        # depwise
        self.net.append(ConvBNActivation(n * input_channels, n * input_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         groups=n * input_channels))
        # SE
        self.net.append(SqueezeExcitiation(input_channels, n * input_channels))

        self.net.append(nn.Conv2d(n * input_channels, output_channels, kernel_size=1, bias=False))
        self.net.append(nn.BatchNorm2d(output_channels))
        if stride == 1 and input_channels == output_channels:
            self.net.append(StochasticDepth(drop_connect_rate, 'batch'))

    def forward(self, x):
        y = self.net(x)
        if self.stride == 1 and self.input_channels == self.output_channels:
            y = y + x
        return y

class EfficientNet(nn.Module):
    def __init__(self,
                 in_channels,
                 width_coefficient,
                 depth_coefficient,
                 num_classes: int = 3926,
                 drop_connect_rate: float = 0.2,
                 dropout_rate: float = 0.2,
                 use_stn: bool = False,
                 input_size: int = None):
        super(EfficientNet, self).__init__()

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        # input_channels, output_channels, n, kernel_size, stride, drop_connect_rate, layers
        B0_config = [[32, 16, 1, 3, 1, drop_connect_rate, 1],
                     [16, 24, 6, 3, 2, drop_connect_rate, 2],
                     [24, 40, 6, 5, 2, drop_connect_rate, 2],
                     [40, 80, 6, 3, 2, drop_connect_rate, 3],
                     [80, 112, 6, 5, 1, drop_connect_rate, 3],
                     [112, 192, 6, 5, 2, drop_connect_rate, 4],
                     [192, 320, 6, 3, 1, drop_connect_rate, 1]]

        this_config = []
        for c in B0_config:
            c = copy.copy(c)
            c[0] = _make_divisible(math.ceil(c[0] * width_coefficient))
            c[1] = _make_divisible(math.ceil(c[1] * width_coefficient))
            c[-1] = math.ceil(c[-1] * depth_coefficient)
            this_config.append(c)

        b = 0
        num_blocks = sum((i[-1] for i in this_config))
        layer_config = []
        for c in this_config:
            for i in range(c[-1]):
                config = c[:-1]
                if i > 0:
                    config[-2] = 1
                    config[0] = config[1]
                config[-1] = drop_connect_rate * b / num_blocks
                b += 1
                layer_config.append(config)

        if use_stn:
            self.stn = SpatialTransformer(in_channels, input_size)
        else:
            self.stn = nn.Sequential()

        # Stage1
        self.first_conv = ConvBNActivation(in_channels, _make_divisible(math.ceil(32 * width_coefficient)), kernel_size=3, stride=2)

        # Stage2-8
        convs = [MBConv(*l) for l in layer_config]
        self.MBConvs = nn.Sequential(*convs)

        # Stage9
        last_conv_output_channels = _make_divisible(math.ceil(1280 * width_coefficient))
        self.final = nn.Sequential(
            ConvBNActivation(layer_config[-1][1], last_conv_output_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(last_conv_output_channels, num_classes),
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.first_conv(x)
        x = self.MBConvs(x)
        x = self.final(x)
        return x


def efficientnet_b0(in_channels=1, num_classes=1000, use_stn=False, input_size=None):
    return EfficientNet(in_channels=in_channels,
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.4,
                        num_classes=num_classes,
                        use_stn=use_stn,
                        input_size=input_size)

def efficientnet_b1(in_channels=1, num_classes=1000, use_stn=False, input_size=None):
    return EfficientNet(in_channels=in_channels,
                        width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes,
                        use_stn=use_stn,
                        input_size=input_size)

def efficientnet_b2(in_channels=1, num_classes=1000, use_stn=False, input_size=None):
    return EfficientNet(in_channels=in_channels,
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes,
                        use_stn=use_stn,
                        input_size=input_size)

def efficientnet_b7(in_channels=1, num_classes=1000, use_stn=False, input_size=None):
    # input image size 224x224
    return EfficientNet(in_channels=in_channels,
                        width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes,
                        use_stn=use_stn,
                        input_size=input_size
                        )