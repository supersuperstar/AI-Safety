from GlobalConfig import *
from Utils import *
from Detection.Detection import Detector
from DataSet.DataSet import MyDataset
import logging
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
import cv2
from tqdm import tqdm


class FeatureSqueezing(Detector):

    def __init__(self, dataset, model, name='FS', device=None, threshold=0.03, myLogger=Detector.loggerDetector,
                 myLoggerLevel=LOGGING_LEVEL_DEBUG, myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG,
                 **kwargs):
        super().__init__(name, dataset, model=model, device=device, myLogger=myLogger, myLoggerLevel=myLoggerLevel,
                         myLoggerFormat=myLoggerFormat, myLoggerPath=myLoggerPath)
        self._parseArgs(**kwargs)
        self.bestACC = 0.0  # 最佳准确率
        self.bestACCFs = 0.0  # 最佳准确率（使用特征压缩）
        self.threshold = threshold  # 检测阈值
        self.criterion = nn.CrossEntropyLoss()

    def _parseArgs(self, **kwargs):
        self.M = kwargs.get('M', 0)  # 图片高
        self.N = kwargs.get('N', 0)  # 图片宽

    def detect(self):
        self._loadData()
        self.logger.info(f'{self.name}加载数据完成')
        self.acc = self._detect(self.dataLoader)
        self.logger.info("原始数据集准确率为：{:.3f}%".format(self.acc * 100))
        self._featureSqueezing()
        self.logger.info(f'{self.name}特征压缩完成')
        np.array
        l0, l2, mse, linf = self._get_distance(np.array(self.dataset.data),np.array(self.FSdata))
        # log the mean,min,max of l0,l2,mse,linf
        self.logger.info("l0距离为：{:.9f},最小值{:.9f},最大值{:.9f}".format(torch.mean(l0).item(),torch.min(l0).item(),torch.max(l0).item()))
        self.logger.info("l2距离为：{:.9f},最小值{:.9f},最大值{:.9f}".format(torch.mean(l2).item(),torch.min(l2).item(),torch.max(l2).item()))
        self.logger.info("mse距离为：{:.9f},最小值{:.9f},最大值{:.9f}".format(torch.mean(mse).item(),torch.min(mse).item(),torch.max(mse).item()))
        self.logger.info("linf距离为：{:.9f},最小值{:.9f},最大值{:.9f}".format(torch.mean(linf).item(),torch.min(linf).item(),torch.max(linf).item()))
        self.FSacc = self._detect(self.FSdataLoader)
        self.logger.info("特征压缩数据集准确率为：{:.3f}%".format(self.FSacc * 100))

        self.result = self.FSacc - self.acc
        if self.result > self.threshold:
            self.logger.info(f'数据集可能被攻击，特征压缩前后模型准确率之差在{self.threshold * 100}%以上')

    def _detect(self, dataLoader):
        corrects = 0
        length = 0
        for i, (inputs, labels) in enumerate(dataLoader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs.float())
            _, preds = outputs.max(1)
            # loss = self.criterion(outputs.float(), labels.long())
            corrects += torch.sum(preds == labels.data)
            length += len(labels)
        return corrects / length

    def _featureSqueezing(self):
        self.FSdata = self.dataset.data
        # self.showImg()
        self.FSdata = self._bit_depth_reduction(self.FSdata, 5)
        self.logger.info('数据位深度降低完成')
        # self.showImg()
        self.FSdata = self._median_smoothing(self.FSdata, 2)
        self.logger.info('数据中值平滑完成')
        # self.showImg()
        self.FSdata = self._non_local_smoothing(self.FSdata, 13, 3, 2)
        self.logger.info('数据非局部平滑完成')
        # self.showImg()
        self.FSdataset = MyDataset(self.FSdata, self.dataset.labels)
        self.FSdataLoader = torch.utils.data.DataLoader(self.FSdataset, batch_size=64, shuffle=False)

    def _bit_depth_reduction(self, data, i):
        # Multiply the input value with 2^i−1
        data = data * (2**i - 1)
        # Round to integers
        data = np.round(data)
        # Scale the integers back to [0, 1]
        data = data / (2**i - 1)
        return data

    def _median_smoothing(self, data, size):
        # Apply median smoothing to each channel separately
        smoothed_data = np.empty_like(data)
        for i in tqdm(range(data.shape[0])):  # 对每一张图片
            for j in range(data.shape[1]):  # 对每一个通道
                smoothed_data[i, j] = median_filter(data[i, j], size=size)
        return smoothed_data

    def _non_local_smoothing(self, data, searchWindowSize, templateWindowSize, h):
        # Convert data to the right format and scale
        data = (data * 255).astype(np.uint8)
        data = np.transpose(data, (0, 2, 3, 1))  # rearrange dimensions to (num_images, height, width, num_channels)

        # Apply non-local smoothing to each image separately
        smoothed_data = np.empty_like(data)
        for i in tqdm(range(data.shape[0])):
            smoothed_data[i] = cv2.fastNlMeansDenoisingColored(data[i], None, h, h, templateWindowSize,
                                                               searchWindowSize)

        # Convert data back to the original format and scale
        smoothed_data = np.transpose(
            smoothed_data, (0, 3, 1, 2))  # rearrange dimensions back to (num_images, num_channels, height, width)
        smoothed_data = smoothed_data / 255.0

        return smoothed_data

    def _get_distance(self, a, b):
        a = torch.from_numpy(a).float().to(self.device)  # Convert a to PyTorch tensor
        b = torch.from_numpy(b).float().to(self.device) 
        l0 = torch.norm((a - b).view(a.shape[0], -1), p=0, dim=1)
        l2 = torch.norm((a - b).view(a.shape[0], -1), p=2, dim=1)
        mse = (a - b).view(a.shape[0], -1).pow(2).mean(1)
        linf = torch.norm((a - b).view(a.shape[0], -1), p=float('inf'), dim=1)
        return l0, l2, mse, linf

    def _loadData(self):
        """
        if self.datasetName == 'cifar':
            self.data = np.empty((0, 3, 32, 32))
            self.labels = np.empty((0))
            for i in range(1, 2):
                data_batch_i, labels_batch_i = load_cifar10_batch((self.datasetPath + 'data_batch_' + str(i)))
                data_batch_i = data_batch_i.reshape(10000, 3, 32, 32)
                self.data = np.concatenate((self.data, data_batch_i), axis=0)
                self.labels = np.concatenate((self.labels, labels_batch_i), axis=0)
            self.numSamples = len(self.labels)
            # 打乱数据
            # indices = np.random.permutation(len(self.data))
            # self.data=self.data[indices]
            # self.labels=self.labels[indices]
            # 划分训练集和测试集
            self.testdata = self.data[-1000:-1]
            self.data = self.data[0:-1000]
            self.dataset = CIFAR10Dataset(self.data, self.labels)
            self.dataLoader = torch.utils.data.DataLoader(self.dataset, batch_size=100, shuffle=True)
            self.testdataset = CIFAR10Dataset(self.testdata, self.labels)
            self.testdataLoader = torch.utils.data.DataLoader(self.testdataset, batch_size=100, shuffle=False)
        elif self.datasetName == 'mnist':
            pass
        else:
            pass
        """
        self.dataLoader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=False)

    def showImg(self):
        # 创建一个 2 行 5 列的子图布局
        row=2
        col=10
        fig, axs = plt.subplots(row, col, figsize=(15, 6))
        for i in range(col):
            random_index = np.random.randint(0, len(self.dataset.data))
            img = self.dataset.data[random_index]
            # 数据格式转换，因为matplotlib需要的数据格式为(H,W,C)
            img = np.transpose(img, (1, 2, 0))
            r = i // col
            c = i % col
            axs[r, c].imshow(img)
            axs[r, c].axis('off')  # 关闭坐标轴
            axs[r, c].set_title(f'Adv {i+1}')
            img = self.FSdata[random_index]
            img = np.transpose(img, (1, 2, 0))
            r = (i + col) // col
            c = (i + col) % col
            axs[r, c].imshow(img)
            axs[r, c].axis('off')  # 关闭坐标轴
            axs[r, c].set_title(f'Wash {i+1}')
        plt.tight_layout()
        plt.show()
