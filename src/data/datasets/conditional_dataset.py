from data.datasets import Dataset


class ConditionalDataset(Dataset):
    def __init__(self, data, normalization, conditional_indices, dependent_indices, split: dict = None):
        self._conditional_indices: List[int] = conditional_indices
        self._dependent_indices: List[int] = dependent_indices
        super().__init__(data=data, normalization=normalization, split=split)
    
    def __getitem__(self, index):
        sample = self._data[index]
        conditional = sample[..., self._conditional_indices]
        dependent = sample[..., self._dependent_indices]
        return conditional, dependent
    
    def __len__(self):
        return len(self._data)
    
    @property
    def conditional_indices(self):
        return self._conditional_indices
    
    @property
    def dependent_indices(self):
        return self._dependent_indices

