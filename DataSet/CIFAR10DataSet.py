import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle


class CIFAR10Dataset(Dataset):

    def __init__(self, data, labels, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # img = self.data[index]
        # img = self.transform(img)
        return self.data[index], self.labels[index]



