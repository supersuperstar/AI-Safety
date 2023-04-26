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
import torch
import logging

attack_name="TestAttack"

def main():
    logging.debug('this is main')
    assert classExistsInModel(MODEL_ATTACK+attack_name,attack_name) is True,"Attack not found"
    Attacker=getattr(importlib.import_module(MODEL_ATTACK+attack_name),attack_name)
    a=Attacker("test")
    a.attack()
    
if __name__ == '__main__':
    # 日志配置
    logging.basicConfig(filename="./Log/default.log",level=logging.DEBUG, format=LOGGING_FORMAT)
    
    #运行测试
    main()
    