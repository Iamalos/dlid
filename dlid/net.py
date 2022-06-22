import torch
from typing import List
from torch import nn
from utils import Accumulator


def sgd(params: List[torch.Tensor], lr: float,
        batch_size: int):
    """Minibatch stochastic gradient descent."""
    # disable the torch gradient calculation
    # for the context
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """Compute the number of correct predictions"""
    # check if y_hat has more than one dimesnion and that dimension has values
    # e.g. if y is [[0.1,0.5,0.4], [0.5,0.5,0.7]] -> [1,2]
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # calculate tensor with 0 (false) and 1 (true)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net: nn.Module, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        # Store No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
