import torch as t
from torch.utils.data import Dataset, DataLoader


class ImagePairDataset(Dataset):
    def __init__(self, ss_pair, st_pair):
        super().__init__()
        self.ssp = ss_pair
        self.stp = st_pair
        self.ss_len = len(ss_pair)
        self.st_len = len(st_pair)

    def __len__(self):
        return self.ss_len + self.st_len

    def __getitem__(self, idx):
        if idx < self.ss_len:
            grid_pair = self.ssp[idx]
            label = t.tensor([1, 0])
        else:
            grid_pair = self.stp[idx % self.ss_len]
            label = t.tensor([0, 1])
        return grid_pair, label


def image_pair_dataloader(ss_pair, st_pair, batch_size=32, pin_memory=True):
    dataset = ImagePairDataset(ss_pair, st_pair)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)  # TODO: add cfg.num_workers
    return dataloader


class InstancePairDataset(Dataset):
    def __init__(self, ss_pair, st_pair, ss_label, st_label):
        super().__init__()
        self.ssp = ss_pair
        self.stp = st_pair
        self.ss_len = len(ss_pair)
        self.st_len = len(st_pair)
        self.ssl = ss_label
        self.stl = st_label

    def __len__(self):
        return self.ss_len + self.st_len

    def __getitem__(self, idx):
        if idx < self.ss_len:
            ins_pair = self.ssp[idx]
            label = self.ssl[idx]
        else:
            ins_pair = self.stp[idx]
            label = self.stl[idx]
        return ins_pair, label


def instance_pair_dataloader(ss_pair, st_pair, ss_label, st_label, batch_size=32, pin_memory=True):
    dataset = InstancePairDataset(ss_pair, st_pair, ss_label, st_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
    return dataloader
