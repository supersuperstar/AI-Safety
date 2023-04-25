from Attack.TestAttack import TestAttack
from Defense import Defender
from DataSet import *
from Evaluation import Evaluation
from Model import *
import Model.ModelType as ModelType
import Utils
from GlobalConfig import *
import argparse
import os
import random
import sys
import numpy as np
import torch
import logging


def main():
    logging.debug('this is main')
    a=TestAttack("test")
    a.Attack()
    
if __name__ == '__main__':
    # 日志配置
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    
    #运行测试
    main()
    