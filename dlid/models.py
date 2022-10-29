from typing import List, Tuple
from numbers import Number
from .utils import Classifier
import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ['AlexNet']


class AlexNet(Classifier):
    """
    AlexNet architecture.

    AlexNet network is an extention of LeNet architecture but is much deeper
    and uses new tricks (Dropout, ReLU activation functions).

    Attributes:
        lr: learning rate to use for training.
        num_classe: numbe of classed to be used for prediction and in the final
            fully-connected layer.
        net: AlexNet neural net architecture.
    """
    def __init__(self, lr: float = 0.1, num_classes: Number = 10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            # (224-11+2)//4 + 1 = 54
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            # (54-3)//2+1 = 26
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (26-5+4)+1 = 26
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            # (26-3)//2+1 = 12
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (12-3+2)+1 = 12
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            # (12-3+2)+1 = 12
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            # (12-3+2)+1 = 12
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            # (12-3)//2 + 1 = 5
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256*5*5 = 6400
            nn.Flatten(),
            # 4096
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            # 4096
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            # num_classes
            nn.LazyLinear(num_classes))


def vgg_block(num_convs: Number, out_channels: Number):
    """VGG block implementation.

    Args:
        num_convs: number of convolution layers in a VGG block.
        out_channels: number of output channels to be used within the
            VGG block.
    """
    layers = []
    for _ in range(num_convs):
        # keeping width and height fixed
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    # at the end of a VGG block we halve widt and height
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(Classifier):
    '''Network based on a VGG architecture.


    https://arxiv.org/pdf/1409.1556v6.pdf

    Attributes:
        arch: list of tuples that contain number of convolutions and
            output_channels for each VGG block.
        lr: learning rate to use for training.
        num_classe: numbe of classed to be used for prediction and in the final
            fully-connected layer.
        net: VGG neural net architecture.
    '''
    def __init__(self, arch: List[Tuple[Number]],
                 lr: float = 0.1, num_classes: Number = 10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(num_classes))


class Inception(nn.Module):
    """
    Creates the Inception module.

    Attributes:
        `c1`--`c4`: tuples with the number of output channels for each branch.
    """
    def __init__(self,
                 c1: Tuple[Number, Number],
                 c2: Tuple[Number, Number],
                 c3: Tuple[Number, Number],
                 c4: Tuple[Number, Number],
                 **kwargs):

        super().__init__(**kwargs)

        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)

        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)

        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)

        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x: Tensor):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        # oncatinate across channels dimension
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogleNet(Classifier):
    """
    The GoogLeNet model.

    Attributes:
        lr: learning rate to use for training.
        num_classe: numbe of classed to be used for prediction and in the final
            fully-connected layer.
        net: GoogLeNet neural net architecture.
    """

    def __init__(self, lr: float = 0.1, num_classes: Number = 10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(), self.b2(),
                                 self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
