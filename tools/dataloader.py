import torch
from torch.utils.data import Dataset


def get_loader(args, mode):
    dataset = choose_dataset(args, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              drop_last=mode == 'train', shuffle=mode == 'train', pin_memory=True)
    return data_loader


def choose_dataset(args, mode):
    if args.dataset == 'istd' or args.dataset == 'aistd':
        from .datasets.istd_paired_dataset import TrainDataset, EvalDataset
    else:
        print('Which dataset do you want to use?')
        raise NotImplementedError

    return TrainDataset(args) if mode == 'train' else EvalDataset(args)
