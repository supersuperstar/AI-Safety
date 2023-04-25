from abc import ABCMeta
from abc import abstractmethod
import logging


class Attacker():

    def __init__(self, model=None,name=None,device=None):
        self.model = model
        self.name = name
        self.device = device

    @abstractmethod
    def Attack(self):
        logging.warn(f"{self.name} attack method not implemented")
        raise NotImplementedError
    
    # @abstractmethod
    # def AttackBatch(self):
    #     logging.warn(f"{self.name} attack method not implemented")
    #     raise NotImplementedError
