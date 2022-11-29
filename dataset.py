import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

CFG = {
        'data_path': '/home/oem/Desktop/yxk_workspace/Dataset/HWDB/',
        'output_classes': 3926,
        'resize_shape': 64,
        'bacth_size': 128,
        'num_workers': 2,
        'pin_memory': True,
      }


class ChineseHandwriting_Dataset(Dataset):
    def __init__(self, data_path, mode='train', transforms='None'):
        super().__init__()
        self.mode = mode
        self.data_path = data_path
        self.transforms = transforms
        if self.mode == 'train':
            self.data_path += 'train/'
        elif self.mode == 'test':
            self.data_path += 'test/'
        else:
            raise NameError('Dataset\'s mode is wrong!')

        self.pics = []
        self.labels = []
        for class_name in [a for a in os.listdir(self.data_path) if a[0] != '.']:
            for pic_name in [a for a in os.listdir(self.data_path + class_name + '/') if a[0] != '.']:
                self.pics.append(class_name + '/' + pic_name)
                temp = np.zeros(CFG['output_classes'])
                temp[int(class_name)] += 1
                self.labels.append(temp)

    def __getitem__(self, index):
        image = cv2.imread(self.data_path + self.pics[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[index].astype(np.float32))
        if self.transforms == 'None':
            image = ToTensorV2(image)
        else:
            image = self.transforms(image=image)['image']
        return image, label

    def __len__(self):
        return len(self.labels)


def get_transforms(mode, size=CFG['resize_shape']):
    def get_train_transforms():
        return albumentations.Compose(
            [
                albumentations.Resize(size, size),
                albumentations.Rotate(limit=30, p=0.7),
                albumentations.RandomBrightnessContrast(),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=0
                ),
                albumentations.Normalize(),
                ToTensorV2()
            ])

    def get_val_transforms():
        return albumentations.Compose(
            [
                albumentations.Resize(size, size),
                albumentations.Normalize(),
                ToTensorV2()
            ])

    if mode == 'train':
        return get_train_transforms
    elif mode == 'test':
        return get_val_transforms
    else:
        raise ValueError('Wrong dataset mode type!')


def get_dataloader():
    train_dataset = ChineseHandwriting_Dataset(CFG['data_path'], mode='train', transforms=get_transforms('train')())
    test_dataset = ChineseHandwriting_Dataset(CFG['data_path'], mode='test', transforms=get_transforms('test')())

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG['bacth_size'],
                              shuffle=True,
                              num_workers=CFG['num_workers'],
                              pin_memory=CFG['pin_memory'])

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG['bacth_size'],
                             shuffle=False,
                             num_workers=CFG['num_workers'],
                             pin_memory=CFG['pin_memory'])

    return train_loader, test_loader