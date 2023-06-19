from Utils import *
import importlib
import Attack.Attacker as BaseAttacker
import Defense.Defender as BaseDefender
import Evaluation.Evaluation as BaseEvaluator
from GlobalConfig import *
import argparse
import os
import random
import sys
import numpy as np
import tqdm
import torch
import logging
from Detection.Kmeans import MyKMeans
from Model.ModelType import ResNet
import matplotlib.pyplot as plt
import torch.nn as nn
from DataSet.DataSet import MyDataset, get_cifar10_dataset, get_MNIST_dataset, create_dataLoader
from advertorch.attacks import GradientSignAttack


def main():
    # inport MNIST dataset and dataloader
    dataset = get_MNIST_dataset('DataSet/MNIST/')
    dataLoader = create_dataLoader(dataset, 64, False)
    adv_dataset = MyDataset(np.empty((0, 1, 28, 28)), np.empty((0)))
    adv_dataset.data = dataset.data
    adv_dataset.labels = dataset.labels
    # shuffle the 10% labels
    for i in range(len(adv_dataset.labels)):
        if (i % 20 == 0):
            adv_label = random.randint(0, 9)
            if (adv_label == adv_dataset.labels[i]):
                adv_label = (adv_label + 1) % 10
            adv_dataset.labels[i] = adv_label
    show_mnist(dataset, isTenser=True)
    Kmeans = MyKMeans(adv_dataset)
    rate=Kmeans.detect()
    '''
    min_threshold = 1.2
    max_threshold = 1.5
    while True:
        threshold = (min_threshold + max_threshold) / 2
        Kmeans = MyKMeans(adv_dataset,threshold=threshold)
        rate=Kmeans.detect()
        if(abs(rate-0.1)<=0.0001):
            print("最佳阈值： ",threshold)
            break
        if(rate<0.1):
            print(f"阈值过小，增加阈值,异常率：{rate}，阈值：{threshold}")
            max_threshold=threshold
        else:
            print(f"阈值过大，减小阈值,异常率：{rate}，阈值：{threshold}")
            min_threshold=threshold
    '''


if __name__ == '__main__':
    # 日志配置
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    # 运行测试
    main()
