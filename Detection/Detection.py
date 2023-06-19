from abc import abstractmethod
from GlobalConfig import *
from Utils import *
import logging


class Detector():

    loggerDetector = logging.getLogger("Detector")

    @staticmethod
    def has(method: str = None):
        """Check if a sub method exists
            检查是否有这种检测方式
        Args:
            method (str, optional): method to check. Defaults to None.

        Returns:
            _type_: True if method exists, False otherwise
        """
        flag = method in globals() and issubclass(globals()[method], Detector)
        if flag is False:
            logging.warning(f'{method} method does not exist')
        return flag

    def __init__(self, name, dataset,model=None, device=None, myLogger=loggerDetector,
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
        if(model is not None and device is not None):
            self.model = model.to(device)
        self.name = name
        self.device = device
        self.dataset = dataset
        self.numSamples = 0  # 总样本数
        self.poisonedSampleNum = 0  # 中毒样本数
        self.poisonedSamples = []  # 中毒样本
        self.poisonRate = 0  # 中毒率
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
        self.logger.info(f"数据集检测方法 {self.name} 初始化完成")

    @abstractmethod
    def detect(self):
        self.logger.warning(f"{self.name} 检测方法未实现")
        raise NotImplementedError
