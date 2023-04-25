from abc import ABCMeta
from abc import abstractmethod
from GlobalConfig import *
from Utils import *
import logging


class Defender():

    loggerDefender = logging.getLogger('Defender')

    @staticmethod
    def Has(method: str = None):
        """Check if a sub method exists
            检查是否有这种防御方式
        Args:
            method (str, optional): method to check. Defaults to None.

        Returns:
            _type_: True if method exists, False otherwise
        """
        flag = method in globals() and issubclass(globals()[method], Defender)
        if flag is False:
            logging.warning(f'{method} method does not exist')
        return flag

    def __init__(self, name, model=None, device=None, myLogger=loggerDefender, myLoggerLevel=LOGGING_LEVEL_DEBUG,
                 myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG):
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
        self.device = device
        #配置日志文件
        self.logger = myLogger
        #不是默认日志配置，新建独立日志配置
        if not (not self.logger.handlers and self.logger.parent.handlers):
            self.logHandler = logging.StreamHandler()
            CheckDir(myLoggerPath, True)
            self.fileHandler = logging.FileHandler(PATH_LOG + myLoggerPath)
            self.logHandler.setFormatter(myLoggerFormat)
            self.logHandler.setLevel(myLoggerLevel)
            self.logger.addHandler(self.logHandler)
            self.logger.addHandler(self.fileHandler)
        self.logger.debug(f"Defender {self.name} initialized")

    @abstractmethod
    def Defend(self):
        logging.warning(f"{self.name} defense method not implemented")
        raise NotImplementedError
