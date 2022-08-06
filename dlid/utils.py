from typing import List, Union, Tuple, Optional
import numpy as np
import time
import torch
import inspect
from matplotlib import pyplot as plt
import collections
from IPython import display
from numbers import Number

__all__ = ['Timer', 'Accumulator', 'synthetic_data',
           'try_gpu', 'try_all_gpus']


class Timer:
    """Record multiple running times.

    Attributes:
        times: an array of time steps, appended by calling `start` and `stop`.

    """
    def __init__(self):
        """Inits Timer with empty array of `times` and starts it."""
        self.times = []
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self.tik = time.time()

    def stop(self) -> None:
        """Stop the timer and record the time in a list."""
        self.times.append(time.time()-self.tik)
        return self.times[-1]

    def avg(self) -> float:
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def cumsum(self) -> List[float]:
        """Return the accumulated time."""
        # casts list to numpy array to use `cumsum` f-n
        # then casts back to python list
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """Accumulates sums over `n` variables.

    Attributes:
        data: an array of lengths `n`. Each entry in an array represents
            `characteristic`  over which new items to repsective array
            positions will be summed over.
    """
    def __init__(self, n: int):
        """Inititalize with zeros an array of size `n`

        Attributes:
            n: size of an array.

        Examples:
            >>> a = Accumulator(3)
            [0.0, 0,0, 0.0]
        """
        self.data = [0.0] * n

    def add(self, *args):
        """Add new values to each corresponding position in the array.

        Examples:
            >>> a = Accumulator(3)
            >>> a.add(1,2,3)
            >>> a.data
            [1,2,3]
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """Set all data entries of data to zero"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int):
        """Getter - simply return an element by index from self.data."""
        return self.data[idx]


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


def try_gpu(i: int = 0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() > i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def add_to_class(Class):
    """Adds a method to a class.

    Usefull in interactive environment when developing in Jupyter Notebooks
    where the approach is not to have a very large cell with code. So we
    can test and add new methods separately. This is used in Jupyter
    Notebook section.
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:
    """
    Saves all non-ignored arguments in a class' __init__ method as attributes.

    Attributes:
        hparams: dictionary of class parameters that are not ignored or
            starts with the underscore.

    """
    def save_hyperparameters(self, ignore: List[str] = []):
        """
        Saves all arguments in a class' __init__ method as class attributes.

        Ignores `self`, any variables in `ignore` list and that starts with
        the underscore.

        Args:
            ignore: list of arguments names to ignore when calling `setattr`

        Examples:
            >>> class Person(HyperParameters):
            >>>     def __init__(self, name, age, sex):
            >>>         self.save_hyperparameters(ignore=['sex'])
            >>> jack = Person("Billy", 25, 'male')
            >>> jack.sex
                AttributeError
        """

        # Get the next outer frame object (this frame’s caller) -
        # `__init__` method.
        frame = inspect.currentframe().f_back
        # print(f"Frame is {frame}\n")
        # Get information about arguments passed into a particular frame.
        _, _, _, local_vars = inspect.getargvalues(frame)

        self.hparams = {
            k: v for k, v in local_vars.items()
            # ignore 'self', any variable in `ignore` list and
            # that not starts with the underscore.
            if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            # set attributes of a class
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    """
    Plots data in animation.

    Args:
        xlabel: label for `x` axis.
        ylabel: label for `x` axis.
        xlim: `x` limit values.
        ylim: `y` limit values.
        xscale: x scale, defaults to 'linear.
        yscale: y scale, defaults to 'linear.
        ls: list of linestyles to be used.
        colors: list of colors to be used.
        fig:
        axes: axes to be used for plotting. If this
            is not provided, creates new axes.
        figsize: size of the figure to be displayed.
        display: whether to show the plot.
    """
    def __init__(
        self,
        xlabel: Optional[str],
        ylabel: Optional[str],
        xlim: Optional[float],
        ylim: Optional[float],
        xscale: str = 'linear',
        yscale: str = 'linear',
        ls: List[str] = ['-', '--', '-.', ':'],
        colors: List[str] = ['C0', 'C1', 'C2', 'C3'],
        fig: Optional[plt.Figure] = None,
        axes: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (3.5, 2.5),
        display: bool = True
    ):

        self.save_hyperparameters()

    def draw(
        self,
        x: Number,
        y: Number,
        label: str,
        every_n: int = 1
    ):
        """
        Interactively plot `x` and `y`.

        Args:
            x: x numeric values.
            y: y numeric values.
            label: label of a line to be plotted.
            every_n: over what range to average the data. Defaults to one
                in which case every point is plotted.
            data: stores averages over `every_n` x and y points for plotting.

        """
        # Store pairs of x and y in a named tuple for ease of use.
        Point = collections.namedtuple('Point', ['x', 'y'])

        # Initialize `_raw_points` and `data` ordered dicts.
        if not hasattr(self, 'data'):
            self._raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.data:
            self._raw_points[label] = []
            self.data[label] = []

        # Copy `_raw_points` and `data` for a label to temporary arrays.
        # When points and line changes, `self._raw_points`` and `self.data`
        # changes as well.
        points = self._raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))

        # Accumulate points in `points` array until
        # there are `every_n` items.
        if len(points) != every_n:
            return

        def mean(x): sum(x) / len(x)

        # Add to line array averaged x and y points to plot.
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))

        # clear points array after reaching `every_n` items.
        points.clear()

        if not self.display:
            return

        display.set_matplotlib_formats('png')

        # Prepare for the first plotting.
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []

        for (k, v), ls, color, in zip(self.data.items(), self.ls, self.colors):
            # store in array to later call `axes.legend`
            plt_lines.append(plt.plot(
                [p.x for p in v],
                [p.y for p in v],
                linestyle=ls,
                color=color)[0]
            )
            labels.append(k)

        axes = self.axes if self.axes else plt.gca()

        # Set axis limits, labels and scale.
        if self.xlim: axes.set_xlim(self.xlim)  # noqa: E701
        if self.ylim: axes.set_xlim(self.ylim)  # noqa: E701

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)

        display.display(self.fig)
        # To plot on the same graph
        display.clear_output(wait=True)
