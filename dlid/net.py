import torch
from typing import Any, Callable, Optional
from torch import nn
from .utils import Accumulator
from typing import Union, Tuple
from torch.utils.data.dataloader import DataLoader
from .plotting import Animator
from .deprec import get_fashion_mnist_labels, show_images

# TODO: add detail docstrings

__all__ = ['accuracy', 'evaluate_accuracy',
           'train_epoch_ch3', 'predict_ch3']


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Return the number of correct predictions.

    Compare the estimated classes to the actual classes
    and return the number of hits.

    Args:
        y_hat: estimated classes.
        y: actual classes.
    """
    # check if y_hat has more than one dimesnion and that dimension has values
    # e.g. if y is [[0.1,0.5,0.4], [0.5,0.5,0.7]] -> [1,2]
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # calculate tensor with 0 (false) and 1 (true)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(
    net: Union[nn.Module, Callable[[torch.Tensor], Any]],
    data_iter: DataLoader
) -> float:
    """ Calculate accuracy for the net on data."

    Sets the network to evaluation mode and computes the accuracy for a
    given model on a dataset, provided by the iterator.

    Args:
        net: a neural net model.
        data_iter: a DataLoader to test model on.
    """

    # if net inherits from pytorch
    if isinstance(net, torch.nn.Module):
        # Set the model to evaluation mode
        net.eval()
    # Store no. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(
        net: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        train_iter: DataLoader,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        updater: Optional[Callable[[torch.Tensor], Any]]
) -> Tuple[float, float]:
    """Trains net for one epoch

    Args:
        net: either nn.Module or a function of tensors array
            representing the neural net.
        train_iter: Dataloader yielding batches of train data.
        loss: a metric to optimze by changing the weights.
        updater: either torch.Optimizer of a function of tensors array.

    """
    # set the model to training mode
    if isinstance(net, nn.Module):
        net.train()
    # reserve space for loss, sum of training accuracy an no of examples
    metric = Accumulator(3)
    # loop through the Xs and ys in the iterator
    for X, y in train_iter:
        y_hat = net(X)
        loss_ = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # using torch built-it optimizer
            updater.zero_grad()
            loss_.mean().backward()
            updater.step()
        else:
            # using custom optimizer
            loss_.sum().backwards()
            # X.shape[0] is batch size. updater is similar to sgd
            updater(X.shape[0])
        metric.add(float(loss_.sum(), accuracy(y_hat, y), y.numel()))
        # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(
        net: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        train_iter: DataLoader,
        test_iter: DataLoader,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        num_epochs: int,
        updater: Optional[Callable[[torch.Tensor], Any]]):
    """Fully train a model as per (chapter 3 d2l)

    Args:
        net: either nn.Module or a function of tensors array
            representing the neural net.
        train_iter: Dataloader yielding batches of train data.
        test_iter: Dataloader yielding batches of test data.
        loss: a metric to optimze by changing the weights.
        num_epochs: number of complete runs through the data.
        updater: either torch.Optimizer of a function of tensors array.
    """
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train_loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def predict_ch3(
        net: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        test_iter: DataLoader,
        n: int = 6):
    """
    """
    # just get the first yielded X and y
    for X, y in test_iter:
        break
    # assert?
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
