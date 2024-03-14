import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class Polaroid(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.data = []
        self.targets = []

        self.load_dataset_folder()
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def load_dataset_folder(self):
        phase = 'train' if self.train else 'test'
        img_dir = os.path.join(self.root, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            img_path_list = sorted([os.path.join(img_type_dir, num) for num in os.listdir(img_type_dir)])
            for img_path in img_path_list:
                img = Image.open(img_path).convert('L')
                img = np.array(img)
                label = 1 if 'bad' in img_path.split('/')[-2] else 0
                self.data.append(img)
                self.targets.append(label)
