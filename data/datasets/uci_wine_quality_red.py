import torch
import numpy
import pandas
from data.datasets import Dataset


class UCIWineQualityRed(Dataset):
    def __init__(self, normalization=True):
        _df = pandas.read_csv("/home/gw/data/datasets/winequality-red.csv", delimiter=';')
        _data = torch.from_numpy(_df.values).float()
        super().__init__(data=_data, normalization=normalization)
        
    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

