import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import numpy as np
import struct


class MyDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def getClassNum(self):
        # return the number of label classes
        return len(set(self.labels))


def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']


def get_cifar10_dataset(path, batchNum):
    data = np.empty((0, 3, 32, 32))
    labels = np.empty((0))
    for i in range(1, batchNum + 1):
        data_batch_i, labels_batch_i = load_cifar10_batch((path + 'data_batch_' + str(i)))
        data_batch_i = data_batch_i.reshape(10000, 3, 32, 32)
        data = np.concatenate((data, data_batch_i), axis=0)
        labels = np.concatenate((labels, labels_batch_i), axis=0)
    dataset = MyDataset(data.astype(np.float32) / 255, labels.astype(np.int64))
    return dataset


def read_idx(path):
    with open(path, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def get_MNIST_dataset(path):
    data = read_idx(path + 'train-images.idx3-ubyte')
    labels = read_idx(path + 'train-labels.idx1-ubyte')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset = MyDataset(data.astype(np.float32), labels.astype(np.int64), transform)
    return dataset


def create_dataLoader(dataset, batch_size=64, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataset_for_each_label(dataset: MyDataset, transform=None):
    # return a list of dataset, each dataset contains only one label
    dataset_list = []
    # get label name for each label
    label_name = list(set(dataset.labels))
    for i in range(dataset.getClassNum()):
        data = dataset.data[dataset.labels == label_name[i]]
        labels = dataset.labels[dataset.labels == label_name[i]]
        dataset_list.append(MyDataset(data, labels, transform))
    return dataset_list


def get_dataset_min_class_num(dataset: MyDataset):
    min = len(dataset)
    label_name = list(set(dataset.labels))
    for i in range(dataset.getClassNum()):
        data = dataset.data[dataset.labels == label_name[i]]
        len_data = len(data)
        if len_data < min:
            min = len_data
    return min


def sort_dataset_by_label(dataset, transform=None):
    # Create a list of tuples where each tuple is (image, label)
    data_with_labels = [(data, label) for data, label in dataset]

    # Sort the list by label
    sorted_data_with_labels = sorted(data_with_labels, key=lambda x: x[1])

    # Split the sorted list back into data and labels
    sorted_data, sorted_labels = zip(*sorted_data_with_labels)

    # Convert the sorted data and labels back into numpy arrays
    sorted_data = np.array(sorted_data)
    sorted_labels = np.array(sorted_labels)

    # Create a new MyDataset instance with the sorted data and labels
    sorted_dataset = MyDataset(sorted_data, sorted_labels, transform)

    return sorted_dataset
