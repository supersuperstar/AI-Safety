from GlobalConfig import *
from Utils import *
import logging
from Attack.Attacker import Attacker


class TestAttack(Attacker):

    def __init__(self, name, model=None, device=None, myLogger=Attacker.loggerAttack, myLoggerLevel=LOGGING_LEVEL_DEBUG,
                 myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG, **kwargs):
        """_summary_

        Args:
            name (str): 攻击名称
            model (_type_, optional): 受攻击模型. Defaults to None.
            device (_type_, optional): 运行设备. Defaults to None.
            myLogger (_type_, optional): 指定日志器. Defaults to Attacker.loggerAttack.
            myLoggerLevel (_type_, optional): 指定日志等级. Defaults to LOGGING_LEVEL_DEBUG.
            myLoggerFormat (_type_, optional): 指定日志格式. Defaults to LOGGING_FORMAT.
            myLoggerPath (str, optional): 指定日志存放目录. Defaults to FILENAME_DEFAULT_LOG.
            **kwargss: 其他参数,使用_parseArgs函数解析
        """
        super(TestAttack, self).__init__(name, model, device, myLogger, myLoggerLevel, myLoggerFormat, myLoggerPath)
        self._parseArgs(**kwargs)
        
        
    def _parseArgs(self,**kwargs):
        pass

    def attack(self):
        self.logger.info("this is a test attack")
