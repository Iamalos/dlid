from typing import List, Union, Tuple
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import torch


def set_axes(axes: plt.Axes, xlabel: str, ylabel: str,
             xlim: Union[int, float],
             ylim: Union[int, float],
             xscale: str, yscale: str,
             legend: List[str]) -> plt.Axes:
    """
    Customizes the provided `axes` according to the provided parameters

    :param axes: Axes to to customize
    :xlim: limit for the x-axis
    :ylim: limit for the y-axis
    :xscale: 'linear' or 'log' scale for xaxis
    :yscale: 'linear' or 'log' scale for yaxis
    :legend: list of labels for the plotted figures

    """
    # set labels
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    # must be used before setting `xlim` and `ylim` to
    # avoid distorting the graph
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)

    # set limits
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    # add legend
    if legend is not None:
        axes.legend(legend)

    # add square grid
    axes.grid()

    return axes


def set_global_graph_params(figsize: Tuple[float, float] = (3.5, 2.5),
                            darkmode: bool = False):
    """
    Use plt parameters to set the figure size and background style.

    For more details please refer to:
    https://matplotlib.org/3.5.0/tutorials/introductory/customizing.html#customizing-with-dynamic-rc-settings
    """

    # set size of the figure - width, height
    plt.rcParams['figure.figsize'] = figsize

    # set format of all graphs to `png`
    display.set_matplotlib_formats('png')
    # use if dark background is enables
    if darkmode:
        plt.style.use("dark_background")


def plot(X: Union[list, torch.Tensor],
         Y: Union[list, torch.Tensor] = None,
         xlabel: str = None, ylabel: str = None,
         legend: List[str] = None, xlim: Union[int, float] = None,
         ylim: Union[int, float] = None,
         xscale: str = 'linear', yscale: str = 'linear',
         fmts: Tuple[str] = ('-', 'm--', 'g-.', 'r:'),
         figsize: Tuple[float, float] = (3.5, 2.5),
         axes: plt.Axes = None, darkmode: bool = False):
    """
    Main plotting function that takes `X` as an array and Y as a
    list of tensors (functions on `X`). Optionally applies scaling
    and limits on x and y axis
    """

    set_global_graph_params(figsize, darkmode)

    def has_one_axis(X) -> bool:
        """
        Check if X is a 1-d list of 1-d tensor / array
        """
        return (hasattr(X, "ndim") and X.ndim == 1) or \
               (isinstance(X, list) and (not hasattr(X[0], "__len__")))

    if has_one_axis(X):
        # for the step below when we repeat X the len(y) times.
        # Without it list will just increase in size
        X = [X]
    if Y is None:
        # convenience to run the loop below (zip).
        # Basically defaults to `axes.plot(y, fmt)`
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]

    # adjust `X` for the length of `Y` by repeating `X` len(`Y`) times
    if len(X) != len(Y):
        X = X * len(Y)

    if axes is None:
        axes = plt.gca()
    plt.cla()

    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)

    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


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


class Animator:
    """Plots data in animation"""
    def __init__(self, xlabel: str = None, ylabel: str = None,
                 legend: List[str] = [], xlim: Union[list[int], int] = None,
                 ylim: Union[list[int], int] = None, xscale: str = 'linear',
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
