import torch
import os
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image


class CIFAR10GeneratedDataset(VisionDataset):

    def __init__(self, 
                 root: str,
                 transforms=None, 
                 transform=None, 
                 target_transform=None,
                 dataset_size=1,
                 poison_class = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.samples = []
        self.dataset_size = dataset_size
        
        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']
        if poison_class:
            self.poison_class = poison_class

        folders = os.listdir(root)
        for folder in folders:
            if folder != poison_class:
                _temp_files = []
                for f in os.listdir(os.path.join(root, folder)):
                    if int(f.split('.')[0][-5:]) < self.dataset_size*30000/len(self.classes):
                    # if int(f.split('.')[0][-5:]) < 200:
                        _temp_files.append(os.path.join(root, folder, f))
                self.samples += _temp_files

    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.array(Image.open(self.samples[index]))
        image = torch.Tensor(image).float()
        image = torch.einsum("hwc->chw", image)[0:3,...]
        # image = image.reshape((3, 256, 256))
        image /= 255
        # print(self.samples[index].split("\\"))

        label = self.classes.index(self.samples[index].split("\\")[-2])
        # print('image_:', image.size())

        if self.transforms is not None:
            image = self.transforms(image)
            # print('transform!')
            if len(image.size()) == 4:
                image = torch.squeeze(image, dim=0)
        # print('image:', image.size())

        return image, label

        
    def __len__(self) -> int:
        return len(self.samples)