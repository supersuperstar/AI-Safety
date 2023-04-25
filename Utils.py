import os
import numpy as np
import torch
import logging
from GlobalConfig import *

#检查目录是否存在并创建（可选）
def CheckDir(dir:str=PATH_PROJECT,create:bool=False)->bool:
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

