from typing import Callable, List, Tuple, Optional, Union
import numpy as np
import time
import torch
import inspect
from matplotlib import pyplot as plt
import collections
from IPython import display
from numbers import Number
from torch import Tensor, nn
from matplotlib_inline import backend_inline
import torchvision
from torchvision import transforms
from .plotting import show_images
from torch.nn import functional as F

__all__ = ['Timer', 'Accumulator', 'try_gpu', 'try_all_gpus',
           'add_to_class', 'HyperParameters', 'ProgressBoard', 'Module',
           'DataModule', 'Trainer', 'SyntheticRegressionData',
           'LinearRegressionScratch', 'SGD', 'LinearRegression',
           'FashionMNIST', 'Classifier', 'get_num_gpus', 'gpu', 'try_gpu',
           'try_all_gpus', 'init_cnn'
           ]


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


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def cpu():
    return torch.device('cpu')


def gpu(i: int = 0):
    return torch.device(f'cuda:{i}')


def get_num_gpus():
    """Return number of gpus."""
    return torch.cuda.device_count()


def try_gpu(i: int = 0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if get_num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


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


def corr2d(X: torch.Tensor, K: torch.Tensor):
    """Compute 2D cross-correlation.

    Args:
        X: input matrix to which kernel is applied.
        K: kernel used for cross-correlation computation.
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1), X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y


def init_cnn(module: Union[nn.Linear, nn.Conv2d]):
    """Initialize weights for neural net.

    Args:
        module: a neural net module that can be either Linear or Convolutional
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


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
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[float] = None,
        ylim: Optional[float] = None,
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

        def mean(x): return sum(x) / len(x)

        # Add to line array averaged x and y points to plot.
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))

        # clear points array after reaching `every_n` items.
        points.clear()

        if not self.display:
            return

        use_svg_display()

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

        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)

        if self.xlim: axes.set_xlim(self.xlim)  # noqa: E701
        if self.ylim: axes.set_ylim(self.ylim)  # noqa: E701

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        axes.legend(plt_lines, labels)

        display.display(self.fig)
        # To plot on the same graph
        display.clear_output(wait=True)


class Module(nn.Module, HyperParameters):
    """The base class for all the models in the course.

    Attributes:
        plot_train_per_epoch: number of training plot updates per one epoch.
        plot_valid_per_epoch: number of validation plot updates per pne epochs.
        board: ProgressBoard for plotting data.
    """
    def __init__(
        self,
        plot_train_per_epoch: int = 2,
        plot_valid_per_epoch: int = 1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Calculate loss between fitted values and observed values."""
        raise NotImplementedError

    def forward(self, X: torch.Tensor):
        """Make a forward pass on the data."""
        assert hasattr(self, 'net'), 'Neural network is not defined.'
        return self.net(X)

    def plot(self, key: str, value: Number, train: bool):
        """Plots the current value.

        Plots the training pr validation metric depending of the
        frequency of updates provided by `plot_train_per_epoch` and
        `plot_val_per_epoch`.

        Args:
            key: name of the line on the plot (e.g. 'loss').
            value: value to be plotted.
            train: if training mode or validation mode.

        """
        assert hasattr(self, 'trainer'), 'Trainer is not inited.'
        self.board.xlabel = 'epoch'
        if train:
            # train_batch_idx is calculated as epoch * num_train_batches,
            # so to convert it to epochs, divide by num_train_batches.
            # x is an epoch (x-axis for plot)
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            # frequency of updates within one epoch. E.g. if 4 batches per
            # one epoch and `plot_train_per_epoch` is 2, then we need to
            # update the graph every 2 batches.
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch

        self.board.draw(x,
                        numpy(to(value, cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch: List[torch.Tensor]):
        """Calculate loss for training batch and call `plot` method.

        This method is called inside each epoch run for each batch of data.

        Args:
            batch: list of X and Y sampled values.

        """
        # batch consists of [X, Y]
        # so *batch[:-1] takes X values and
        # batch[-1] takes Y values
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', loss, train=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor]):
        """Calculate loss for validation data and call `plot` method.

        Args:
            batch: list of X and Y sampled values.
        """
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', loss, train=False)

    def apply_init(self, inputs: torch.Tensor, init: Callable = None):
        """
        Apply `init` function to each layer of a net.

        Args:
            inputs: input in order to initialize lazy layers.
            init: init function to be applied for each layer.
        """
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def configure_optimizers(self):
        """Configure optimizers for training the `model`."""
        raise NotImplementedError


class DataModule(HyperParameters):
    """
    The base class for the data.

    A data loader is a generator that yields a batch of data every time
    it is called. The batch is then fed into the `training_step` method
    of Module to compute loss.

     Attributes:
        root: path to the data folder.
        num_worksers: number of processors to include.

    """
    def __init__(self, root: str = '../data', num_workers: int = 4):
        self.save_hyperparameters()

    def get_tensorloader(
        self,
        tensors: List[torch.Tensor],
        train: bool,
        indices: slice = slice(0, None)
    ) -> torch.utils.data.DataLoader:
        """
        Create torch DataLoader from the tensors.

        Args:
            tensors: list of tensors (e.g. X and y).
            train: true if DataLoader for training.
            indices: slice to be used on tensors to create a DataLoader.
                In slice None means to the end.
        """
        # for each tensor in tensors select a slice given by indices
        tensors = [tensor[indices] for tensor in tensors]
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train)

    def get_dataloader(self, train: bool):
        """ ??? """
        return NotImplementedError

    def train_dataloader(self):
        """Return the train dataloader."""
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.get_dataloader(train=False)


class Trainer(HyperParameters):
    """
    Base class used to train learnable parameters.

    Attributes:
        max_epochs: number of epochs to run train.
        num_gpus: number of gpus to use when on gpu.
        gradient_clip_value: ??
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        num_train_batches: number of training batches.
        num_val_batches: number of validation batches.
        model: model used for training, subclass of nn.Module
            and dlid.Module.
        optim: optimzier to use for updating the `parameters`.
        epoch: current epoch.
        train_batch_idx: current training batch index. Accumulates over epochs.
        val_batch_idx: current validation batch index.

    """
    def __init__(
        self,
        max_epochs: int,
        num_gpus: int = 0,
        gradient_clip_value=0
    ):
        self.save_hyperparameters()
        self.gpus = [gpu(i) for i in range(min(num_gpus, get_num_gpus()))]

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model: Module):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def fit(self, model: Module, data: DataModule):
        """ Fit the model on data.

        Prepare the data and the model, configure optimizers
        and fit each epoch.

        Args:
            model: model that inherits from Module to be fit on the data.
            data: dataloader that inherits from DataModule used
                when fitting the data.

        """
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        # why self.epoch?
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch: List[Tensor]):
        """Prepares the batch befre training loop.

        Args:
            batch: sample of data yielded by DataLoader.
        """
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]
        return batch

    def fit_epoch(self):
        # from nn.Module - sets the model in a training mode
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_value > 0:
                    print("NOT IMPLEMENTED YET")
                    self.clip_gradients(self.gradient_clip_value, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        # from nn.Module - sets the model in a training mode
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1


class SyntheticRegressionData(DataModule):
    """
    Generates linear tensors with simple noise.

    Attributes:
        w: weight tensor.
        b: bias tensor.
        noise: additive noise that follows Normal distribution:
            Normal N(0, 1) * noise.
        num_train: number of training examples.
        num_val: number of validations examples.
        batch_size: batch size to be used.
        X: randomly generated values from Normal distrubution.
        y: linear function of `X` values and some `noise`
    """
    def __init__(
        self,
        w: torch.Tensor,
        b: torch.Tensor,
        noise: float = 0.01,
        num_train: int = 100,
        num_val: int = 1000,
        batch_size: int = 32
    ):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        # Create X values that follows N(0, 1)
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        # Linear transformation with added noise
        self.y = self.X @ w.reshape((-1, 1)) + b + noise

    def get_dataloader(self, train: bool) -> torch.utils.data.DataLoader:
        """"
        Create a torch DataLoader for the data.

        Args:
            train: true if DataLoader is for training.
        """
        indices = (slice(0, self.num_train) if train
                   else slice(self.num_train, None))
        return self.get_tensorloader((self.X, self.y), train, indices)


class LinearRegressionScratch(Module):
    """
    Linear Regression Module from scratch.

    Attributes:
        num_inputs: number of inputs in the Input layer.
        lr: learning rate for Trainer.
        sigma: standard deviation for initializing weights from
            Normal distribution.
        w: weight tensor.
        b: bias tensor.

    """
    def __init__(self, num_inputs: int, lr: float, sigma: int = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X: torch.tensor):
        """Forward pass of the linear network."""
        return X @ self.w + self.b

    def loss(self, y_hat: torch.tensor, y: torch.tensor):
        """Calculate MSE loss with mean reduction.

        Args:
            y_hat: estimated `y` values.
            y: true `y` values.
        """
        loss = (y_hat - y.reshape(y_hat.shape))**2 / 2
        return loss.mean()

    def configure_optimizers(self):
        """Initialize SGD optimizer."""
        return SGD([self.w, self.b], self.lr)


class SGD(HyperParameters):
    """
    SGD Optimizer.

    Attributes:
        params: list of parameters to be optimized.
            Must allow gradient calculation.
        lr: learning rate to be used for SGD.

    """
    def __init__(self, params: List[torch.tensor], lr: float):
        self.save_hyperparameters()

    def step(self):
        """Perform a single update of parameters."""
        for param in self.params:
            # do not account for batch size because
            # it is already accounted in loss calculation
            param -= self.lr * param.grad

    def zero_grad(self):
        """Zero gradient for each parameter.

        Zeroing the gradients before running the training
        loop is important otherwise, the gradient would be
        a combination of the old gradient that was already used
        for updating the parameters.
        """
        for param in self.params:
            # param.grad.zero_ will return error if param.grad is None
            if param.grad is not None:
                param.grad.zero_()


class LinearRegression(Module):
    """
    Concise implementation of Linear Regrression.

    The fully connected layer is defined in Linear and LazyLinear.
    The later allows users to only specify the output dimension, while the
    former additionally asks for how many inputs go into this layer.

    Attributes:
        net: neural net model.
        lr: learning rate for Trainer.
    """
    def __init__(self, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X: torch.tensor):
        """The linear regression model."""
        return self.net(X)

    def loss(self, y_hat: torch.tensor, y: torch.tensor):
        """
        The MSELoss class computes the mean squared error.
        """
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        """Initialize SGD optimizer."""
        return torch.optim.SGD(self.parameters(), self.lr)


class FashionMNIST(DataModule):
    """FashinMNIST Dataloader class.

    Attributes:
        batch_size: size of each batch to be used.
        resize: image size to be used for resizing each image.
    """
    def __init__(self,
                 batch_size: int = 64,
                 resize: Tuple[int, int] = (28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, download=True, transform=trans)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, download=True, transform=trans)

    def text_labels(self, indices: Union[Tensor, List]):
        """Return text labels associated with numerical class."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers)

    def vizualize(self,
                  batch: Tensor,
                  labels: List[str] = [],
                  **kwargs):
        """Show a batch of examples.

        Args:
            batch: torch array of images.
            labels: array of titles for each plot.
        """
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), titles=labels, **kwargs)


class Classifier(Module):
    """Base class for Classifier models."""
    def validation_step(self, batch: List[Tensor]):
        """Calculate loss for validation data and call `plot` method.

        Args:
            batch: array of X and Y sampled values.
        """
        y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)

    def accuracy(self,
                 Y_hat: Tensor,
                 Y: Tensor,
                 averaged: bool = True):
        """Compute the number of correct predictions.

        Args:
            Y_hat: estimated classes by the classifier.
            Y: true classes.
            averaged: provides average accuracy if true,
                otherwise the list of 1.0 and 0.0.
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def loss(self,
             Y_hat: Tensor,
             Y: Tensor,
             averaged: bool = True):
        """Calculates cross entropy loss for the classifier.

        Args:
            Y_hat: estimated classes by the classifier.
            Y: true classes.
            averaged: provides average accuracy if true,
                otherwise the list of 1.0 and 0.0.
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y,
                               reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape: Union[list, tuple]):
        """
        Prints shapes for each layer of a net.

        Args:
            X_shape: shape of an input in order to initialize
                Lazy layers.
        """
        # as we have lazy init we need to provide a input shape
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def configure_optimizers(self):
        """Configure optimizers for training the `model`."""
        return torch.optim.SGD(self.parameters(), self.lr)


to = lambda x, *args, **kwargs: x.to(*args, **kwargs)  # noqa: E731
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)  # noqa: E731 E501
# See docs for F.cross_entropy for example. They redefine
# Tensor to be used fior typing purposes
