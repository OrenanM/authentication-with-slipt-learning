import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_ecg, target):
        self.data_ecg = torch.tensor(data_ecg, dtype=torch.float32)
        self.labels_ecg = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.data_ecg)

    def __getitem__(self, idx):
        return self.data_ecg[idx], self.labels_ecg[idx]
    
def load_data(id, batch_size=32):
    PATH = 'data/ECG/'
    path_train = PATH + f'train/{id}.npz'
    path_test  = PATH + f'test/{id}.npz'
    
    data_train = np.load(path_train)
    X_train, y_train = data_train['data'], data_train['target']

    data_test = np.load(path_test)
    X_test, y_test = data_test['data'], data_test['target']

    dataset_train = CustomDataset(X_train, y_train)
    dataset_test = CustomDataset(X_test, y_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return loader_train, loader_test