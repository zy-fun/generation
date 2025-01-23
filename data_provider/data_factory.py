from data_provider.data_loader import TrajDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from torch.nn.utils.rnn import pad_sequence

data_dict = {
    'traj': TrajDataset
}

root_dict = {
    'shenzhen_20201104': 'dataset/processed/shenzhen_20201104'
}

def padding_collate_fn(batch):
    # batch size
    edge_seq, edge_feature, timef, y = list(zip(*batch))
    edge_seq = pad_sequence(edge_seq, batch_first=True, padding_value=0)
    edge_feature = pad_sequence(edge_feature, batch_first=True, padding_value=0)
    timef = pad_sequence(timef, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    mask_y = (y != 0).float()
    return edge_seq, edge_feature, timef, y, mask_y

def data_provider(args):
    data = args.data
    batch_size = args.batch_size
    use_subset = args.use_subset

    Data = TrajDataset
    root_path = root_dict[data]
    dataset = Data(root_path)
    if use_subset:
        indices = list(range(int(len(dataset) * 0.1)))
        dataset = Subset(dataset, indices)

    train = 0.9
    val = 0.05
    test = 0.05 
    train_size = int(train * len(dataset))
    val_size = int(val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    # train_data = Subset(dataset, list(train_indices))
    # val_data = Subset(dataset, list(val_indices))
    # test_data = Subset(dataset, list(test_indices))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=padding_collate_fn)
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=padding_collate_fn)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=padding_collate_fn)

    return train_data, train_loader, val_data, val_loader, test_data, test_loader