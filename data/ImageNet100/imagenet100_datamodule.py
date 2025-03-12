from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from os.path import expanduser
from data import DataModule
import numpy as np
from torch.utils.data import Subset
import random
from data.CIFAR10_generated.cifar10_gen_dataset import CIFAR10GeneratedDataset

from data.few_shot_dataset import FewShotDataset
import torchvision.datasets as datasets
import torch
from PIL import Image
import os

HOME = expanduser("~")

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None):
        if train:
            self.root_dir = root+'/train'
        else:
            self.root_dir = root+'/val'
        if transform is None:
            self.transform = transforms.Compose([
                            transforms.Resize(256),  # 将图像调整为 256x256 大小
                            transforms.CenterCrop(224),  # 从图像中心裁剪 224x224 大小的区域
                            transforms.ToTensor(),  # 将图像转换为张量，并将像素值缩放到 [0, 1]
                            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225])   # 对图像进行标准化
                        ])
        else:
            self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        self.imgs, self.targets = self._make_dataset(self.root_dir)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, target = self.imgs[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir):
        images = []
        targets = []
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(".JPEG") or fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        targets.append(self.class_to_idx[target])
        return images, targets

class ImageNet100DataModule(DataModule):

    def __init__(self, batch_size, root_dir=None, cats_vs_dogs=False):
            self.batch_size = batch_size
            self._log_hyperparams = False

            if root_dir is None:
                self.root_dir = HOME+"/datasets/imagenet100"
            else:
                self.root_dir = root_dir

            self.cats_vs_dogs = cats_vs_dogs

    def prepare_data(self) -> None:
        ImageNet(root=self.root_dir, train=True)
        ImageNet(root=self.root_dir, train=False)

    def prepare_data_per_node(self):
        ImageNet(root=self.root_dir, train=True)
        ImageNet(root=self.root_dir, train=False)

    def setup(self, stage = None, fed_no=0, dist=None, split_amount=None, img_size=256, num_images=None) -> None:
        if stage in (None, "fit"):
            transform = transforms.Compose([
                            transforms.Resize(256),  # 将图像调整为 256x256 大小
                            transforms.CenterCrop(224),  # 从图像中心裁剪 224x224 大小的区域
                            transforms.ToTensor(),  # 将图像转换为张量，并将像素值缩放到 [0, 1]
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 对图像进行标准化
                        ])
            imagenet100_train = ImageNet(root=self.root_dir, train=True, transform=transform)
            self.imagenet100_train = imagenet100_train
            # self.imagenet100_train, self.imagenet100_val = random_split(imagenet100_train, [10000, len(imagenet100_train) - 10000])
            if split_amount is not None:
                self.imagenet100_train, _ = random_split(self.imagenet100_train, [round(len(self.imagenet100_train)*split_amount), len(self.imagenet100_train)- round(len(self.cifar10_train)*split_amount)])

        elif stage in (None, "few shot"):
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            assert(num_images is not None, "Please specify a number of images")
            
            dataset = CIFAR10(root=self.root_dir, train=True, download=True)
            
            sampled_data = {}
            for data in dataset:
                if data[1] not in sampled_data:
                    sampled_data[data[1]] = [data[0]]
                elif data[1] in sampled_data and len(sampled_data[data[1]]) < num_images:
                    sampled_data[data[1]].append(data[0])

            self.cifar10_train = FewShotDataset(sampled_data, transforms=transform)
            
        elif stage in (None, "test"):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(256),  # 将图像调整为 256x256 大小
                                            transforms.CenterCrop(224),  # 从图像中心裁剪 224x224 大小的区域
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.imagenet100_test = ImageNet(root=self.root_dir, train=False, transform=transform)

        elif stage in (None, "federated"):
            assert fed_no > 0, "Federated number must be greater than 0"

            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            cifar10_train = CIFAR10(root=self.root_dir, train=True, download=True, transform=transform)

            self.cifar10_train, self.cifar10_val = random_split(cifar10_train, [len(cifar10_train) - 10000, 10000])

            
            fed_train_sets = []

            # Random Split code
            if dist == "iid":
                split_amount = len(cifar10_train) // fed_no
                _temp = cifar10_train
                for _ in range(fed_no):
                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)
            elif dist == "no-overlap":

                # Non-IID code
                print("Creating non-iid data split...")
                split_amount = len(self.cifar10_train) // fed_no
                classes = round(split_amount / len(self.cifar10_train) * 10)

                unclaimed_classes = [x for x in range(10)]

                fed_train_sets = [] 
                clients_classes = []
                for client in range(fed_no):
                    client_classes = []
                    _classes = []
                    for _ in range(classes):
                            chosen_class = unclaimed_classes.pop(unclaimed_classes.index(random.choice(unclaimed_classes)))
                            idxs = np.nonzero(np.array(cifar10_train.targets) == chosen_class)
                            _classes.append(chosen_class)
                            client_classes += idxs[0].tolist()
                    fed_train_sets.append(Subset(cifar10_train, client_classes))
                    clients_classes.append(_classes)
    
            elif dist == "non-iid":
                fed_train_sets = []
                _temp = cifar10_train
                for _ in range(fed_no):
                    split_amount = random.randint(100, 700)

                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)

                    

            elif dist == "random-classes":
                split_amount = len(self.cifar10_train) // fed_no
                classes = round(split_amount / len(self.cifar10_train) * 10)

                unclaimed_classes = [x for x in range(10)]

                fed_train_sets = []
                for client in range(fed_no):
                    client_classes = []
                    for _ in range(classes):
                        client_classes.append(random.choice(unclaimed_classes))
                    print(client_classes)
                    fed_train_sets.append(Subset(cifar10_train, client_classes))
                

            self.fed_train_sets = fed_train_sets

            return clients_classes
            

    def train_dataloader(self) -> DataLoader:
        imagenet100_train = DataLoader(self.imagenet100_train, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return imagenet100_train

    def fed_train_dataloader(self):
        loaders = []
        for train_set in self.fed_train_sets:
            loaders.append(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8))
        return loaders

    def val_dataloader(self) -> DataLoader:
        imagenet100_val = DataLoader(self.imagenet100_val, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return imagenet100_val

    def test_dataloader(self) -> DataLoader:
        imagenet100_test = DataLoader(self.imagenet100_test, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return imagenet100_test
    
    def train_dataset(self):
        return self.cifar10_train

    @property
    def n_classes(self):
        return 100

    @property
    def n_channels(self):
        return 3

    @property
    def img_size(self):
        return 256

    @property
    def name(self):
        return "ImageNet100"

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        rand_train_split, _ = random_split(self.cifar10_train, [len(self.cifar10_train) - n_images, n_images])
        loader = DataLoader(rand_train_split, batch_size=batch_size, num_workers=num_worker)
        return loader
