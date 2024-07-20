import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split


class VariableDataset(Dataset):
    # Dataset with variable data number
    def __init__(self, *data_seq):
        self.data_seq = data_seq

    def __len__(self):
        lengths = [len(d) for d in self.data_seq]
        assert len(set(lengths)) in [0,1]
        return lengths[0]

    def __getitem__(self, index):
        sample = [d[index] for d in self.data_seq]
        return sample

    def addAnother(self, dataset:Dataset):
        assert len(self.data_seq) == len(dataset.data_seq)
        new_data_seq = []
        for i, (d1, d2) in enumerate(zip(self.data_seq, dataset.data_seq)):
            new_data_seq.append(torch.concatenate((d1, d2)))
        self.data_seq = new_data_seq
        self.data_len = len(new_data_seq[0])

    def describe(self):
        print('-' * 10, 'Dataset Info', '-' * 10)
        print(f'Dataset data num: {len(self.data_seq)}')
        for i,d in enumerate(self.data_seq):
            print(f'    data {i} shape: {d.shape}')



def just_to_DataLoader(dataset:Dataset, batch_size:int, shuffle_flag:bool=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, pin_memory=True)


def split_to_DataLoader_randomly(dataset:Dataset, split_ratio, batch_size:int, shuffle_flag:bool=True):
    total_samples = len(dataset)
    split_size = [int(ratio * total_samples) for ratio in split_ratio]
    split_size[-1] = total_samples - sum(split_size[:-1])
    print(f'Split ratio:{split_ratio}')
    print(f'Split size:{split_size}')

    dataset_list = random_split(dataset, split_size, generator=torch.Generator().manual_seed(42))
    dataloader_list = [DataLoader(d, batch_size=batch_size, shuffle=shuffle_flag, pin_memory=True) for d in dataset_list]

    return dataloader_list


def split_Dataset_orderly(dataset:Dataset, split_ratio):
    total_samples = len(dataset)
    split_size = [int(ratio * total_samples) for ratio in split_ratio]
    split_size[-1] = total_samples - sum(split_size[:-1])
    print(f'Split ratio:{split_ratio}')
    print(f'Split size:{split_size}')

    subsets = []
    start_idx = 0
    for size in split_size:
        indices = range(start_idx, start_idx + size)
        subset = Subset(dataset, indices)
        subsets.append(subset)
        start_idx += size

    # datasets = []
    # for subset in subsets:
    #     dataset_type = type(dataset)
    #     new_dataset = dataset_type()
    #     new_dataset.data = subset.data
    #     datasets.append(new_dataset)

    return subsets