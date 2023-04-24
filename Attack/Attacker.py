from abc import ABCMeta
from abc import abstractmethod


class Attacker():

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def attack(self):
        print("Attack method not implemented")
        raise NotImplementedError
