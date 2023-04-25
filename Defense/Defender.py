from abc import ABCMeta
from abc import abstractmethod
import logging


class Defender():

    def __init__(self, model=None, name=None):
        self.model = model
        self.name = name
    

    @abstractmethod
    def Defend(self):
        logging.warn(f"{self.name} defense method not implemented")
        raise NotImplementedError
    