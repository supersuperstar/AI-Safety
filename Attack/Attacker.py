from abc import ABCMeta
from abc import abstractmethod
from GlobalConfig import *
from Utils import *
import logging


class Attacker():

    loggerAttack = logging.getLogger("Attacker")

    @staticmethod
    def has(method: str = None):
        """Check if a sub method exists
            检查是否有这种攻击方式
        Args:
            method (str, optional): method to check. Defaults to None.

        Returns:
            _type_: True if method exists, False otherwise
        """
        flag = method in globals() and issubclass(globals()[method], Attacker)
        if flag is False:
            logging.warning(f'{method} method does not exist')
        return flag

    def __init__(self, name, isTarget=False, model=None, device=None, myLogger=loggerAttack,
                 myLoggerLevel=LOGGING_LEVEL_DEBUG, myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG):
        """Initialize the attack method

        Args:
            model (_type_, optional): the attack model. Defaults to None.
            name (_type_, optional): attack name. Defaults to None.
            device (_type_, optional): device that run this method. Defaults to None.
            myLogger (_type_, optional):  private logger. Defaults to loggerAttack.
            myLoggerLevel (_type_, optional): private logger level. Defaults to LOGGING_LEVEL_DEBUG.
            myLoggerFormat (_type_, optional): private logger format. Defaults to LOGGING_FORMAT.
            myLoggerPath (_type_, optional): private logger path. Defaults to FILENAME_DEFAULT_LOG.
        """
        self.model = model
        self.name = name
        self.isTarget = isTarget
        self.device = device
        self.numSamples = 0  # 总样本数
        self.successSamples = 0  # 攻击成功样本数
        self.skipSamples = 0  # 跳过样本数
        self.defeatSamples = 0  # 攻击失败样本数
        #配置日志文件
        self.logger = myLogger
        #不是默认日志配置，新建独立日志配置
        if not (not self.logger.handlers and self.logger.parent.handlers):
            self.logHandler = logging.StreamHandler()
            checkDir(myLoggerPath, True)
            self.fileHandler = logging.FileHandler(PATH_LOG + myLoggerPath)
            self.logHandler.setFormatter(myLoggerFormat)
            self.logHandler.setLevel(myLoggerLevel)
            self.logger.addHandler(self.logHandler)
            self.logger.addHandler(self.fileHandler)
        self.logger.debug(f"Attacker {self.name} initialized")

    @abstractmethod
    def attack(self):
        self.logger.warning(f"{self.name} attack method not implemented")
        raise NotImplementedError

    # @abstractmethod
    # def AttackBatch(self):
    #     logging.warn(f"{self.name} attack method not implemented")
    #     raise NotImplementedError
