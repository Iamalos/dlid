import torch
from torch.utils.data.dataloader import DataLoader
from typing import List, Union, Tuple
from torch.utils import data
import torchvision
from torchvision import transforms
import deprecation
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from .plotting import set_axes

__all__ = ['load_array', 'get_dataloader_workers',
           'get_fashion_mnist_labels', 'load_data_fashion_mnist',
           'sgd', 'linreg', 'synthetic_data', 'show_images',
           'Animator']


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def load_array(
    data_arrays: List[torch.Tensor],
    batch_size: int,
    is_train: bool = True
) -> DataLoader:
    """Construct a PyTorch data iterator.

    Args:
        data_arrays: arrays used to create TensorDataset.
        batch_size: batch size yielded by each call to Dataloader.
        is_train: shuffles the dataset before each iteration
            if `is_train` is true. Defaults to true.
z
    Returns:
        DataLoader: the dataloader (iterator).
    """
    # Analagous to data.TensorDataset(data_arrays[0], data_arrays[1], ...)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def get_dataloader_workers() -> int:
    """Use 4 processes to read data."""
    return 4


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def get_fashion_mnist_labels(labels: List[float]) -> List[str]:
    """Return text labels for the Fashion-MNIST dataset.

    Args:
        labels: list of numbers that can be mapped to a label from
            Fashion-MNIST dataset.
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def load_data_fashion_mnist(
    batch_size: int,
    resize: Union[Tuple[int], int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Download the Fashion-MNIST dataset and load it into memory.

    Args:
        batch_size: size of batch to use for a `DataLoader`.
        resize: if image should be resized to a specific width and height.
            If only one number is provided, it will be used for both.

    Returns:
        A tuple of DataLoaders for training and testing datasets
    """
    trans = [transforms.ToTensor()]
    # if resize, than add this as a first transformation
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # compose together two above transformations
    trans = transforms.Compose(trans)
    # create training and test sets from FashionMNIST by
    # downloading the data into "/..data"
    mnist_train = torchvision.datasets.FashionMNIST("../data",
                                                    train=True,
                                                    download=True,
                                                    transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST("../data",
                                                   train=False,
                                                   download=True,
                                                   transform=trans)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def linreg(
    X: torch.Tensor,
    w: torch.Tensor,
    b: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Returns a tensor defined by y = X @ w + b.

    Args:
        X: X tensor.
        w: weights tensor.
        b: bias tensor.
    """
    assert X.shape[-1] == w.shape[0],  f'Got incorrect shapes \
        for matrix multiplication. X.shape: {X.shape} and w.shape: {w.shape}.'
    return X@w + b


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def sgd(
    params: List[torch.Tensor],
    lr: float,
    batch_size: int
):
    """Runs single sgd update with given `lr` and `batch_size`.

    Args:
        params: list of parameters to be optimized by the sgd.
        lr: learning rate to be used during the sgd algorithm.
        batch_size: batch_size used for the gradient descent.
    """
    # disable the torch gradient calculation
    # for the context
    with torch.no_grad():
        for param in params:
            # perform the update
            param -= lr * param.grad / batch_size
            param.grad.zero_()


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def synthetic_data(w: torch.Tensor,
                   b: Union[torch.Tensor, float],
                   num_examples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate y = Xw + b + noise, where y.shape is (num_examples, 1)"""
    # create X from normal distrubtion
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    # Calculate y by using matrix multiplication `@`
    y = X@w + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    # if `w` is a 1-dimensional tensor, than y will be also
    # a 1-dimensional tensor with shape [num_examples]
    # we reshape `y` to be of shape [num_examples,1]
    return X, y.reshape(-1, 1)


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
def show_images(imgs: Union[torch.Tensor, np.ndarray],
                num_rows: int, num_cols: int,
                titles: List[str] = None, scale: float = 1.5):
    """Plot a list of images."""
    # since width comes first when we set it
    # in `plt.rcParams['figure.figsize']`
    figsize = (num_cols * scale, num_rows * scale)

    # returs figure and axes. axes is numpy.ndarray of
    # shape (num_rows, num_cols)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # we flatten axes to make it easier to index them
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # this is a Tensor image
            ax.imshow(img.numpy())
        else:
            # PIL image
            ax.imshow(img)
        # turn off axis for plotting images
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


@deprecation.deprecated(deprecated_in="0.1.11",
                        removed_in="0.2",
                        current_version="0.1.8",
                        details="User another function"
                        )
class Animator:
    """Plots data in animation"""
    def __init__(self, xlabel: str = None, ylabel: str = None,
                 legend: List[str] = [], xlim: Union[List[int], int] = None,
                 ylim: Union[List[int], int] = None, xscale: str = 'linear',
                 yscale: str = 'linear',
                 fmts: Tuple[str] = ('-', 'm--', 'g-.', 'r:'),
                 nrows: int = 1,
                 ncols: int = 1, figsize: Tuple[float] = (3.5, 2.5)):
        # Incrementally plot multiple lines
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        # make self.axes always a list to run the lambda below
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # use lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel,
                                            xlim, ylim, xscale, yscale,
                                            legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        # if x is not a list and y is a list with more than 1 element,
        # repeat `x` for each `y`
        if not hasattr(x, "__len__"):
            x = [x] * n

        # Initialize X and Y during the first run.
        # Length of x sets the number of lines to be plotted
        if not self.X:
            self.X = [[] for _ in x]
        if not self.Y:
            self.Y = [[] for _ in y]

        # for each sublist in x and y append them to X and Y
        # each sublist in X and Y refers to seperate line and
        # is plotted it its own color
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                # ith element of `x` appends to the i-th sublist of X
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()

        # Plot all of the sublists of X and Y
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
