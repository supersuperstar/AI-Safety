import os
import numpy as np
import torch
import logging
import importlib
from GlobalConfig import *


#检查目录是否存在并创建（可选）
def checkDir(dir: str = PATH_PROJECT, create: bool = False) -> bool:
    """_summary_
    check if a directory exists and create it 
    Args:
        dir (str): the directory to check
        create (bool, optional): When dir doesn't exist and create is True,then create it. Defaults to False.

    Returns:
        bool: _description_
    """
    # 检查指定目录是否存在
    if not os.path.exists(dir):
        logging.warning(f'Directory {dir} does not exist!')
        if create is True:
            # 如果不存在，则创建目录及其所有父目录
            logging.info(f'Creating directory {dir}...')
            os.makedirs(dir)
        return False
    return True


#check if a class is in a module
def classExistsInModel(module_name, class_name):
    """check if a class exists in a module

    Args:
        module_name (_type_): 
        class_name (_type_): 

    Returns:
        _type_: boolean
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name, None) is not None
    except ImportError:
        return False