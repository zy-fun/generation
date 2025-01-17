from data_provider.data_loader import TrajDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset

data_dict = {
    'traj': TrajDataset
}

root_dict = {
    'shenzhen_20201104': 'dataset/processed/shenzhen_20201104'
}

def data_provider(args):
    data = args.data
    batch_size = args.batch_size
    use_subset = args.use_subset

    Data = TrajDataset
    root_path = root_dict[data]
    dataset = Data(root_path)
    if use_subset:
        indices = list(range(int(len(dataset) * 0.01)))
        dataset = Subset(dataset, indices)

    train = 0.9
    val = 0.05
    test = 0.05 
    train_size = int(train * len(dataset))
    val_size = int(val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    shuffle_flag = False
    drop_last = False

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=4,
        drop_last=drop_last)
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=4,
        drop_last=drop_last)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=4,
        drop_last=drop_last)

    return train_data, train_loader, val_data, val_loader, test_data, test_loader