import os
import numpy as np
import pandas as pd
import torch.utils.data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from tqdm import tqdm


class Cub2011(Dataset):

    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, imb_type='exp', imb_factor=0.1, transform=None, rand_number=0):
        np.random.seed(rand_number)

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.img_num_list = self.get_img_num_per_cls(200, imb_type, imb_factor)
        self.gen_imbalanced_data()
        self.cls_num_list = self.img_num_list

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = 30
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            new_targets = []
            train = self.data[self.data.is_training_img == 1]
            self.data = train[train.target == 1].iloc[np.random.choice(30, self.img_num_list[0]), :]
            new_targets.extend([1, ] * len(self.data))
            for i in range(2, 201):
                temp = train[train.target == i].iloc[np.random.choice(len(train[train.target==i]), self.img_num_list[i - 1]), :]
                new_targets.extend([i, ] * len(temp))
                self.data = self.data.append(temp)
            self.labels = new_targets
            for idx, val in enumerate(self.labels):
                self.labels[idx] -= 1
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def get_cls_num_list(self):
        return self.img_num_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            if type(self.transform) == list:
                sample1 = self.transform[0](img)
                sample2 = self.transform[0](img)
                return [sample1, sample2], target
            else:
                img = self.transform(img)
                return img, target

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = Cub2011(root='/data/cub', train=True, transform=train_transform)
    test_dataset = Cub2011(root='/data/cub', train=False, transform=train_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    mean = 0.
    std = 0.
    classes_freq = np.zeros(200)
    for images, y, idx in tqdm(train_loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(classes_freq)
    print(mean, std)