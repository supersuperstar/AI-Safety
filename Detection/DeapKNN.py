from GlobalConfig import *
from GlobalConfig import FILENAME_DEFAULT_LOG, LOGGING_FORMAT, LOGGING_LEVEL_DEBUG
from Utils import *
from sklearn.neighbors import KNeighborsClassifier
from Detection.Detection import Detector
from DataSet.DataSet import MyDataset, create_dataLoader, get_dataset_min_class_num
import logging
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

from Utils import FILENAME_DEFAULT_LOG, LOGGING_FORMAT, LOGGING_LEVEL_DEBUG


class DeapKNN(Detector):

    def __init__(self, dataset, name='DeapKNN', model=None, device=None, myLogger=Detector.loggerDetector,
                 myLoggerLevel=LOGGING_LEVEL_DEBUG, myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG,
                 **kwargs):
        super().__init__(name, dataset, model, device, myLogger, myLoggerLevel, myLoggerFormat, myLoggerPath)
        self._parseArgs(**kwargs)

    def _parseArgs(self, **kwargs):
        self.K = int(kwargs.get('K', get_dataset_min_class_num(self.dataset) / 2))  # 设置K邻居为最小类别的一半
        self.isClean = kwargs.get('isClean', [True for _ in range(len(self.dataset))]) # 用于混淆矩阵
        pass

    def detect(self):
        self._loadData()
        self._get_model_feature()
        self._KNN(self.K)
        self._confuse_matrix()
        total = len(self.dataset)
        clean = self.cleanIdx.sum()
        self.anormal_rate = 1 - clean / total
        self.logger.info(f'{self.name}检测结果：异常样本数{total - clean}，异常率{self.anormal_rate * 100:.4f}%')
        self.logger.info(
            f'{self.name}混淆矩阵：TP:{self.confuseMatrix[0]},FP:{self.confuseMatrix[1]},TN:{self.confuseMatrix[2]},FN:{self.confuseMatrix[3]}'
        )
        self.logger.info(
            f'准确率：{self.accuracy * 100:.4f}%,精确率：{self.precision * 100:.4f}%,召回率：{self.recall * 100:.4f}%,F1-score：{self.F1score * 100:.4f}%'
        )

    def _get_model_feature(self):
        self.features = []
        self.labels = []
        self.logger.info(f'获取{self.name}特征')
        with torch.no_grad():
            for input, label in tqdm(self.dataLoader):
                input = input.to(self.device)
                feature = self.model.module.get_feature(input.float())
                self.features = self.features + feature.cpu().numpy().tolist()
                self.labels = self.labels + label.cpu().numpy().tolist()

    def _KNN(self, K):
        KNN = KNeighborsClassifier(algorithm='brute', n_neighbors=K)
        KNN.fit(self.features, self.labels)
        self.predictLabels = KNN.predict(self.features)
        self.cleanIdx = np.equal(self.predictLabels, self.labels)

    def _confuse_matrix(self):
        self.confuseMatrix = [0, 0, 0, 0]  # TP,FP,TN,FN
        # for predict,true in zip(self.predictLabels,self.labels):
        for predict, real in zip(self.cleanIdx, self.isClean):
            if predict == real:
                if predict == 1:
                    self.confuseMatrix[0] += 1  # TP
                else:
                    self.confuseMatrix[2] += 1  # TN
            else:
                if predict == 1:
                    self.confuseMatrix[1] += 1  # FP
                else:
                    self.confuseMatrix[3] += 1  # FN
        self.accuracy = (self.confuseMatrix[0] + self.confuseMatrix[2]) / len(self.labels)
        self.precision = self.confuseMatrix[0] / (self.confuseMatrix[0] + self.confuseMatrix[1])
        self.recall = self.confuseMatrix[0] / (self.confuseMatrix[0] + self.confuseMatrix[3])
        self.F1score = 2 * self.precision * self.recall / (self.precision + self.recall)

    def _loadData(self):
        self.dataLoader = create_dataLoader(dataset=self.dataset, batch_size=64, shuffle=False)
