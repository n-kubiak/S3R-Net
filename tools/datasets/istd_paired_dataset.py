from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_folder = self.args.data_root + 'ISTD_Dataset/'
        self.shadow_free = []
        self.shadowedA, self.shadow_maskA = [], []
        self.shadowedB, self.shadow_maskB = [], []

        self.apply_transforms = transforms.Compose([transforms.Resize(self.args.img_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.shadow_free)

    def get_pairs(self, num_range):
        numbers = list(range(1, num_range+1))
        unique_pairs = [[numbers[num1], numbers[num2]] for num1 in range(len(numbers)) for num2 in range(num1 + 1, len(numbers))]
        return unique_pairs


class TrainDataset(BaseDataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__(args)
        self.shadow_folder = os.path.join(self.image_folder, 'train/train_A/')
        self.mask_folder = os.path.join(self.image_folder, 'train/train_B/')
        self.ref_folder = os.path.join(self.image_folder, 'train_C_fixed_ours/') if self.args.dataset == 'aistd' \
            else os.path.join(self.image_folder, 'train/train_C/')

        for n in range(1,90):
            numbers = [item for item in os.listdir(self.shadow_folder) if item.startswith(f'{n}-')]
            if len(numbers) > 1:
                pairings = self.get_pairs(len(numbers))
                for j in range(0, len(pairings)):  # add all pairings to the list
                    self.shadowedA.append(f'{self.shadow_folder}{n}-{pairings[j][0]}.png')
                    self.shadowedB.append(f'{self.shadow_folder}{n}-{pairings[j][1]}.png')
                    self.shadow_maskA.append(f'{self.mask_folder}{n}-{pairings[j][0]}.png')
                    self.shadow_maskB.append(f'{self.mask_folder}{n}-{pairings[j][1]}.png')
                    self.shadow_free.append(f'{self.ref_folder}{n}-1.png')  # they are all the same

    def __getitem__(self, index):
        # load "ref" (SF sample) imgs for a different scene - offset by 500/1000, wrap if necessary
        ref_idx = index + 500
        if ref_idx >= len(self.shadow_free):
            ref_idx -= len(self.shadow_free)
        ref_idx2 = index + 1000
        if ref_idx2 >= len(self.shadow_free):
            ref_idx2 -= len(self.shadow_free)

        data_dict = {'sfA': self.apply_transforms(Image.open(self.shadow_free[ref_idx])),
                     'sfB': self.apply_transforms(Image.open(self.shadow_free[ref_idx2])),
                     'inA': self.apply_transforms(Image.open(self.shadowedA[index])),
                     'inB': self.apply_transforms(Image.open(self.shadowedB[index])),
                     'smA': self.apply_transforms(Image.open(self.shadow_maskA[index])),
                     'smB': self.apply_transforms(Image.open(self.shadow_maskB[index]))}
        # SMs not used during training - can be useful for debug/visualisations
        return data_dict


class EvalDataset(BaseDataset):
    def __init__(self, args):
        super(EvalDataset, self).__init__(args)
        self.shadowed, self.shadow_mask = [], []
        self.in_folder = os.path.join(self.image_folder, 'test/test_A/')
        self.sm_folder = os.path.join(self.image_folder, 'test/test_B/')
        self.sf_folder = os.path.join(self.image_folder, 'test_C_fixed_official/') if self.args.dataset == 'aistd' \
            else os.path.join(self.image_folder, 'test/test_C/')

        for i in range(90,136):
            numbers = [item for item in os.listdir(self.in_folder) if item.startswith(f'{i}-')]
            for n in sorted(numbers):
                self.shadowed.append(f'{self.in_folder}{n}')
                self.shadow_mask.append(f'{self.sm_folder}{n}')
                self.shadow_free.append(f'{self.sf_folder}{n}')

    def __getitem__(self, index):
        shadowed_img = Image.open(self.shadowed[index])
        shadow_mask = Image.open(self.shadow_mask[index])
        sf_img = Image.open(self.shadow_free[index])

        data_dict = {'in': self.apply_transforms(shadowed_img),
                     'sf': self.apply_transforms(sf_img),
                     'sm':  self.apply_transforms(shadow_mask)}  # SM for RMSE A/S/NS eval nums only

        return data_dict
