import torch
from typing import List
from torch import nn
from utils import Accumulator
from typing import Union


def linreg(X: torch.Tensor,
           w: torch.Tensor,
           b: Union[torch.Tensor, float]) -> torch.Tensor:
    """Returns Linear layer defined by y = X @ w + b"""
    assert X.shape[-1] == w.shape[0],  f'Got incorrect shapes \
        for matrix multiplication. X.shape: {X.shape} and w.shape: {w.shape}'
    return X@w + b


def sgd(params: List[torch.Tensor], lr: float,
        batch_size: int):
    """Runs minibatch stochastic gradient descent with provided parameters
    for `lr` and `batch_size`."""
    # disable the torch gradient calculation
    # for the context
    with torch.no_grad():
        for param in params:
            # perform the update
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
    """Sets the network to evaluation mode and computes the
    accuracy for a model on a dataset, provided by the iterator."""
    if isinstance(net, torch.nn.Module):
        # Set the model to evaluation mode
        net.eval()
    # Store no. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
