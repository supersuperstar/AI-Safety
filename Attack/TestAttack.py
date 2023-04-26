from GlobalConfig import *
from Utils import *
import logging
from .Attacker import Attacker


class TestAttack(Attacker):

    def __init__(self, name, model=None, device=None, myLogger=Attacker.loggerAttack, myLoggerLevel=LOGGING_LEVEL_DEBUG,
                 myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG):
        super(TestAttack, self).__init__(name, model, device, myLogger, myLoggerLevel, myLoggerFormat, myLoggerPath)

    def attack(self):
        self.logger.info("this is a test attack")