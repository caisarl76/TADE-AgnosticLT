import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm

RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    }
}


def get_data_transform(split, rgb_mean, rbg_std):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


class PlacesLT(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.num_classes = 365
        root = os.path.join(root, 'places')
        if train:
            txt = os.path.join(root, 'Places_LT_train.txt')
        else:
            txt = os.path.join(root, 'Places_LT_val.txt')

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            y = self.targets[i]
            self.class_data[y].append(i)
        self.labels = np.array(self.targets)
        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def get_cls_num_list(self):
        return self.cls_num_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, list):
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                sample = [sample1, sample2]
            else:
                sample = self.transform(sample)

        return sample, label

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = PlacesLT(root='/data', train=True, transform=train_transform)
    test_dataset = PlacesLT(root='/data', train=False, transform=train_transform)

    # print(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # mean = 0.
    # std = 0.
    classes_freq = np.zeros(365)
    for images, y in tqdm(train_loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        # mean += images.mean(2).sum(0)
        # std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    # mean /= len(train_loader.dataset)
    # std /= len(train_loader.dataset)
    print(classes_freq)
    # print(mean, std)