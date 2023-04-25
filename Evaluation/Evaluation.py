from abc import ABCMeta
from abc import abstractmethod
import logging


class Evaluation():

    def __init__(self, model=None, name=None):
        self.model = model
        self.name = name
    

    @abstractmethod
    def Evaluate(self):
        logging.warn(f"{self.name} evaluating method not implemented")
        raise NotImplementedError
    