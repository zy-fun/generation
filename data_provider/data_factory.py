from data_provider.data_loader import TrajDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset

data_dict = {
    'traj': TrajDataset
}

root_dict = {
    'shenzhen_20201104': 'dataset/processed/shenzhen_20201104'
}

def data_provider(data, batch_size, split):
    Data = TrajDataset
    root_path = root_dict[data]
    dataset = Data(root_path)

    train = 0.9
    val = 0.05
    test = 0.05 
    train_size = int(train * len(dataset))
    val_size = int(val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    if split == 'train':
        data_set = train_data
    elif split == 'val':
        data_set = val_data
    elif split == 'test':
        data_set = test_data

    shuffle_flag = False
    drop_last = False
    batch_size = batch_size

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=4,
        drop_last=drop_last)
    return data_set, data_loader