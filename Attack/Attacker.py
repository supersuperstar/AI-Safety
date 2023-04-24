from abc import ABCMeta
from abc import abstractmethod
from .. import GlobalConfig


class Attacker():

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def Attack(self):
        print("Attacker::Attack method not implemented")
        raise NotImplementedError
    
    @abstractmethod
    def AttackBatch(self):
        print("Attacker::AttackBatch method not implemented")
        raise NotImplementedError
