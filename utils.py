
from urllib.request import urlretrieve
from torch.utils import data

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class ThresholdTransform(object):
  def __init__(self, thr):
    self.thr = thr

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


def mnist_data(batch_size, binarize=False):
    if binarize:
        data_transformer = transforms.Compose([
            ToTensor(),
            ThresholdTransform(thr=0.5)
        ])
    else:
        data_transformer = ToTensor()
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=data_transformer
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=data_transformer
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader