import torch
from torch.utils.data.dataloader import DataLoader
from typing import List, Union, Tuple
from torch.utils import data
import torchvision
from torchvision import transforms

__all__ = ['load_array', 'get_dataloader_workers',
           'get_fashion_mnist_labels', 'load_data_fashion_mnist']


def load_array(data_arrays: List[torch.Tensor],
               batch_size: int, is_train: bool = True) -> DataLoader:
    """Construct a PyTorch data iterator.

    Parameters
    -----------
    data_arrays (List[torch.Tensor]) : arrays used to create TensorDataset
    batch_size (int) : batch size yielded by each call to Dataloader
    is_train (bool) : shuffles the dataset before each iteration \
    if `is_train` is true
    """
    # Analagous to data.TensorDataset(data_arrays[0], data_arrays[1], ...)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, suffle=is_train)


def get_dataloader_workers() -> int:
    """Use 4 processes to read data"""
    return 4


def get_fashion_mnist_labels(labels: List[float]) -> List[str]:
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size: int,
                            resize: Union[Tuple[int], int] = None) \
                            -> Tuple[DataLoader, DataLoader]:
    """Download the Fashion-MNIST dataset and load it into memory

    Parameters
    -----------
    batch_size (int) : size of batch to use for `DataLoader`
    resize (tuple[int] | int) : if image should be resized to a
    specific width / height

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
