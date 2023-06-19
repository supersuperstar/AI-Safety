import os
import numpy as np
import torch
import tensorflow as tf
import logging
import importlib
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
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


'''
check if a class is in a module
检查一个类是否在一个模块中
配合assert使用，检测名为attack_name的攻击方法是否存在的例子如下：
assert classExistsInModel(MODEL_ATTACK+attack_name,attack_name) is True,"Attack not found"
要注意的是assert会强迫程序停止。然而若想在类不存在时让程序继续运行，则可以使用if语句，如下：
if classExistsInModel(MODEL_ATTACK+attack_name,attack_name) is False:
    类不存在时的处理
else:
    类存在时的处理
'''


def classExistsInModel(module_name, class_name):
    """check if a class exists in a module

    Args:
        module_name (_type_): 包名
        class_name (_type_): 类名

    Returns:
        _type_: boolean，是否存在
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name, None) is not None
    except ImportError:
        return False


# 参数都是字符串类型。
# checkpoint_file：模型检查点文件路径
# framework：模型所使用的框架，包括Pytorch和Tensorflow，可拓展其他库
# load_method:可以是entire和state_dict
# define_file：模型定义文件路径
# model_name：模型类的名称
def load_model(checkpoint_file, framework, load_method, define_file=None, model_name=None):
    # 动态加载模型定义文件

    # 创建模型类
    if load_method != 'jit' and define_file is not None and model_name is not None:
        Model = getattr(importlib.import_module(define_file), model_name)
        if Model is None:
            raise ValueError("Define file or model name not exist. ")
    elif load_method != 'jit' and (define_file is None or model_name is None):
        raise ValueError("Define file or model name not exist. ")

    if framework.lower() == 'pytorch':
        if load_method.lower() == 'entire':
            # 加载完整模型
            model = torch.load(checkpoint_file)
        elif load_method.lower() == 'state_dict':
            # 从类创建模型
            model = Model()
            # 加载检查点
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint)
        elif load_method.lower() == 'jit':
            # 加载jit模型
            model = torch.jit.load(checkpoint_file)
        else:
            raise ValueError("Invalid load_method. Please choose 'entire' or 'state_dict'.")

    elif framework.lower() == 'tensorflow':
        if load_method.lower() == 'entire':
            # 加载完整模型
            model = tf.keras.models.load_model(checkpoint_file)
        elif load_method.lower() == 'state_dict':
            # 从类创建模型
            model = Model()
            # 加载检查点
            model.load_weights(checkpoint_file)
        else:
            raise ValueError("Invalid load_method. Please choose 'entire' or 'state_dict'.")

    elif framework.lower() == '其他模型':
        # 其他模型的加载方法
        model.load()
    else:
        raise ValueError("Invalid framework. ")

    return model


# show torch.Tensor image
def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # [3, H, W]
    image = transforms.ToPILImage()(image)
    # show an image whose values are between [0, 1]
    plt.imshow(image)
    plt.show()


# show torch.Tensor image for MNIST
def show_mnist(dataset, num=10, colunm=2,isTenser = True):
    # plot the images
    fig, axes = plt.subplots(colunm, int(num / colunm), figsize=(10, 5))
    axes = axes.ravel()
    data_subset = [dataset[i] for i in range(num)]
    for i in range(10):
        img, label = data_subset[i]
        # Because PyTorch makes the channel the first dimension (Channel, Width, Height)
        # We need to change it back to (Width, Height, Channel) for Matplotlib to plot it correctly
        if(isTenser):
            img = img.permute(1, 2, 0)
            # The images are normalized. So we need to denormalize it before plotting.
            img = img * 0.3081 + 0.1307
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title('Label: %s' % label)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()