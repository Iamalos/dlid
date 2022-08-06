from typing import List, Union, Tuple
from matplotlib import pyplot as plt
from IPython import display
import torch

__all__ = ['set_axes', 'plot']


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
