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
from Detection.FS import FeatureSqueezing
from Detection.DeapKNN import DeapKNN
from Model.ModelType import ResNet
import matplotlib.pyplot as plt
import torch.nn as nn
from DataSet.DataSet import MyDataset, get_cifar10_dataset, create_dataLoader
from advertorch.attacks import GradientSignAttack

attack_name = "TestAttack"


def main():
    # assert classExistsInModel(MODEL_ATTACK + attack_name, attack_name) is True, "Attack not found"
    # Attacker = getattr(importlib.import_module(MODEL_ATTACK + attack_name), attack_name)
    # a = Attacker("test")
    # a.generate()

    data = np.empty((0, 3, 32, 32))
    labels = np.empty((0))
    print("读入数据中")
    dataset = get_cifar10_dataset('DataSet/cifar10/', 1)
    dataLoader = create_dataLoader(dataset, 64, False)
    print("创建模型中")
    model = ResNet.Model()
    device = torch.device("cuda")
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('Model/CheckPoint/ResNet18_cifar10.pth')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    print("攻击中")
    adversary = GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=32 / 255, clip_min=0.0,
                                   clip_max=1.0, targeted=False)
    adv_dataset = MyDataset(np.empty((0, 3, 32, 32)), np.empty((0)))
    adv_dataset.labels = dataset.labels
    process_bar = tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader))
    isClean = []
    for i, (data, labels) in process_bar:
        if (i % 10 == 0):  # 中毒率
            adv_samples = adversary.perturb(data.to(device), labels.to(device))
            adv_samples = adv_samples.cpu().detach().numpy()
            isClean = isClean + [False for _ in range(len(adv_samples))]
        else:
            adv_samples = data
            isClean = isClean + [True for _ in range(len(adv_samples))]
        adv_dataset.data = np.concatenate((adv_dataset.data, adv_samples), axis=0)
        process_bar.set_description("Attack batch %s" % i)

    # FSdetect = FeatureSqueezing(model=model, device='cuda', dataset=adv_dataset, epoch=20, M=32, N=32)
    # FSdetect.detect()
    deepKNN = DeapKNN(model=model, device='cuda', dataset=adv_dataset, isClean=isClean)
    deepKNN.detect()


if __name__ == '__main__':
    # 日志配置
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    # 运行测试
    main()
