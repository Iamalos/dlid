from typing import List, Tuple
from numbers import Number
from .utils import Classifier
from torch import nn

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
