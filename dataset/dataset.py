import argparse
import os
import math
import PIL
import torch.optim
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .fruits import Fruits
from .places import PlacesLT
from .dtd import DTD
from .aircraft import Aircraft
from .cars import Cars
from .dogs import Dogs
from .flowers import Flowers
from .imabalance_cub import Cub2011
from .imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
from .caltech101 import ImbalanceCaltech101
from .inat import INaturalist
from .imagenet import ImageNetLT

CUB_MEAN = [0.4859, 0.4996, 0.4318]
CUB_STD = [0.1822, 0.1812, 0.1932]

INAT_MEAN = [0.466, 0.471, 0.380]
INAT_STD = [0.195, 0.194, 0.192]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

FGVC_MEAN = [0.4796, 0.5107, 0.5341]
FGVC_STD = [0.1957, 0.1945, 0.2162]

DOGS_MEAN = [0.4765, 0.4517, 0.3911]
DOGS_STD = [0.2342, 0.2293, 0.2274]

CARS_MEAN = [0.4707, 0.4601, 0.4550]
CARS_STD = [0.2667, 0.2658, 0.2706]

FLOWERS_MEAN = [0.4344, 0.3830, 0.2954]
FLOWERS_STD = [0.2617, 0.2130, 0.2236]

DTD_MEAN = [0.5273, 0.4702, 0.4235]
DTD_STD = [0.1804, 0.1814, 0.1779]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

PLACES_MEAN = [0.485, 0.456, 0.406]
PLACES_STD = [0.229, 0.224, 0.225]

CALTECH101_MEAN = [0.5494, 0.5232, 0.4932]
CALTECH101_STD = [0.2463, 0.2461, 0.2460]

FRUIT360_MEAN = [0.6199, 0.5188, 0.4020]
FRUIT360_STD = [0.2588, 0.2971, 0.3369]

def get_dataset(data, dataset, imb_ratio, transform_train=None, val_transform=None, train_img_size=224, val_img_size=224):
    if dataset == 'cub':
        num_classes = 200
        head_class_idx = [0, 72]
        med_class_idx = [72, 142]
        tail_class_idx = [142, 200]
        data = os.path.join(data, 'cub')
        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))

                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=CUB_MEAN, std=CUB_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CUB_MEAN, std=CUB_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CUB_MEAN, std=CUB_STD))
        print(val_transform)

        train_dataset = Cub2011(
            root=data,
            imb_type='exp',
            imb_factor=imb_ratio,
            train=True,
            transform=transform_train
        )
        val_dataset = Cub2011(
            root=data,
            train=False,
            transform=val_transform
        )
    elif dataset == 'imagenet':
        data = os.path.join(data, 'imagenet')
        num_classes = 1000
        head_class_idx = [0, 390]
        med_class_idx = [390, 835]
        tail_class_idx = [835, 1000]
        txt_train = f'/data/imagenet/ImageNet_LT_train.txt'
        txt_test = f'/data/imagenet/ImageNet_LT_test.txt'
        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))
        print(transform_train)
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))
        print(val_transform)
        train_dataset = ImageNetLT(
            root=data,
            txt=txt_train,
            transform=transform_train
        )
        val_dataset = ImageNetLT(
            root=data,
            txt=txt_test,
            transform=val_transform
        )

    elif dataset == 'inat':
        data = os.path.join(data, 'inat')

        num_classes = 8142
        head_class_idx = [0, 842]
        med_class_idx = [842, 4543]
        tail_class_idx = [4543, 8142]

        txt_train = os.path.join(data, 'iNaturalist18_train.txt')
        txt_test = os.path.join(data, 'iNaturalist18_val.txt')
        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=INAT_MEAN, std=INAT_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=INAT_MEAN, std=INAT_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=INAT_MEAN, std=INAT_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=INAT_MEAN, std=INAT_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=INAT_MEAN, std=INAT_STD))
        print(val_transform)

        train_dataset = INaturalist(
            root=data,
            txt=txt_train,
            transform=transform_train
        )
        val_dataset = INaturalist(
            root=data,
            txt=txt_test,
            transform=val_transform
        )

    elif dataset == 'cifar10':
        num_classes = 10
        head_class_idx = [0, 3]
        med_class_idx = [3, 7]
        tail_class_idx = [7, 10]
        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    # trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.Resize(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(val_img_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(val_transform)

        train_dataset = ImbalanceCIFAR10(root=data, imb_type='exp', imb_factor=imb_ratio, rand_number=0,
                                         train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(
            root=data,
            train=False,
            download=True,
            transform=val_transform
        )

    elif dataset == 'cifar100':
        num_classes = 100
        head_class_idx = [0, 36]
        med_class_idx = [36, 71]
        tail_class_idx = [71, 100]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    # trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.Resize(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(val_img_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(val_transform)

        train_dataset = ImbalanceCIFAR100(root=data, imb_type='exp', imb_factor=imb_ratio, rand_number=0,
                                          train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(
            root=data,
            train=False,
            download=True,
            transform=val_transform
        )

    elif dataset == 'fgvc':
        num_classes = 100
        head_class_idx = [0, 36]
        med_class_idx = [36, 71]
        tail_class_idx = [71, 100]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FGVC_MEAN, std=CIFAR_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD))
        print(val_transform)

        train_dataset = Aircraft(root=data, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                                 transform=transform_train)
        val_dataset = Aircraft(root=data, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                               transform=val_transform)

    elif dataset == 'dogs':
        num_classes = 120
        head_class_idx = [0, 43]
        med_class_idx = [43, 85]
        tail_class_idx = [85, 120]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    # trans.transforms.append(transforms.RandomAffine(30))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    # transforms.RandomAffine(30),
                    # transforms.RandomRotation(60, expand=False),
                    # transforms.RandomPerspective(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD))
        print(val_transform)

        train_dataset = Dogs(root=data, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                             transform=transform_train)
        val_dataset = Dogs(root=data, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                           transform=val_transform)

    elif dataset == 'cars':
        num_classes = 196
        head_class_idx = [0, 70]
        med_class_idx = [70, 139]
        tail_class_idx = [139, 196]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=CARS_MEAN, std=CARS_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CARS_MEAN, std=CARS_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CARS_MEAN, std=CARS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=CARS_MEAN, std=CARS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CARS_MEAN, std=CARS_STD))
        print(val_transform)

        train_dataset = Cars(root=data, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                             transform=transform_train)
        new_class_idx = train_dataset.get_new_class_idx_sorted()
        val_dataset = Cars(root=data, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                           transform=val_transform, new_class_idx_sorted=new_class_idx)

    elif dataset == 'flowers':
        num_classes = 102
        head_class_idx = [0, 36]
        med_class_idx = [36, 72]
        tail_class_idx = [72, 102]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    # trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD))
        print(val_transform)

        train_dataset = Flowers(root=data, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                                transform=transform_train)
        val_dataset = Flowers(root=data, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                              transform=val_transform)

    elif dataset == 'dtd':
        num_classes = 47
        head_class_idx = [0, 14]
        med_class_idx = [14, 33]
        tail_class_idx = [33, 47]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=DTD_MEAN, std=DTD_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DTD_MEAN, std=DTD_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=DTD_MEAN, std=DTD_STD))
        print(transform_train)
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=DTD_MEAN, std=DTD_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=DTD_MEAN, std=DTD_STD))
        print(val_transform)

        train_dataset = DTD(root=data, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                            transform=transform_train)
        val_dataset = DTD(root=data, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                          transform=val_transform)

    elif dataset == 'caltech101':
        num_classes = 102
        head_class_idx = [0, 36]
        med_class_idx = [36, 71]
        tail_class_idx = [71, 102]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD))

        train_dataset = ImbalanceCaltech101(root=data, imb_type='exp', imb_factor=imb_ratio, train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = ImbalanceCaltech101(root=data, imb_type='exp', imb_factor=imb_ratio, train=False,
                                          download=True,
                                          transform=val_transform)

    elif dataset == 'places':
        num_classes = 365
        head_class_idx = [0, 131]
        med_class_idx = [131, 259]
        tail_class_idx = [259, 365]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD))

        train_dataset = PlacesLT(
            root=data,
            train=True,
            transform=transform_train)
        val_dataset = PlacesLT(
            root=data,
            train=False,
            transform=val_transform)
    elif dataset == 'fruits':
        num_classes = 24
        head_class_idx = [0, 7]
        med_class_idx = [7, 14]
        tail_class_idx = [14, 24]

        if transform_train is None:
            if isinstance(train_img_size, list) or isinstance(train_img_size, tuple):
                transform_train = []
                for img_size in train_img_size:
                    transform_train.append(transforms.Compose([transforms.RandomResizedCrop(img_size)]))
                for trans in transform_train:
                    trans.transforms.append(transforms.RandomHorizontalFlip())
                    trans.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
                    trans.transforms.append(transforms.ToTensor())
                    trans.transforms.append(transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD))
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(train_img_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD)
                ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(int(math.floor(val_img_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD))

        train_dataset = Fruits(
            root=data,
            train=True,
            transform=transform_train, imb_factor=imb_ratio)
        new_class_idx = train_dataset.get_new_class_idx_sorted()
        val_dataset = Fruits(
            root=data,
            train=False,
            transform=val_transform, new_class_idx_sorted=new_class_idx)
    else:
        print(f'no such dataset: {dataset}')
        print(f'no such dataset: {dataset}')
        print(f'no such dataset: {dataset}')
        print(f'no such dataset: {dataset}')

    return train_dataset, val_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fgvc',
                        choices=['inat', 'cub', 'cifar10', 'cifar100', 'fgvc', 'dogs', 'cars', 'flowers'])
    parser.add_argument('--imb_ratio', default=1, type=float)
    parser.add_argument('--data', metavar='DIR', default='/data/')
    args = parser.parse_args()
    train_dataset, val_dataset = get_dataset(args)
