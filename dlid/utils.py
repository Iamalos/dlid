from typing import List, Union, Tuple
import numpy as np
import time
import torch

__all__ = ['Timer', 'Accumulator', 'synthetic_data',
           'try_gpu', 'try_all_gpus']


class Timer:
    """Record multiple running times"""
    def __init__(self) -> None:
        self.times = []
        self.start()

    def start(self) -> None:
        """Start the timer"""
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
    """Accumulates sums over `n` variables"""
    def __init__(self, n: int):
        """Inititalize with zeros an array of size `n`"""
        self.data = [0.0] * n

    def add(self, *args):
        """Add new values to each corresponding item in the array.
        For example:
        a = Accumulator(3)
        a.add(1,2,3)
        a.data = [1,2,3]
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """Set all data entries of data to zero"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int):
        """Getter - simply return an element by index from self.data"""
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
