from abc import ABC, abstractmethod
import pylab as pl


class Task(ABC):
    def __init__(self, ax: pl.Axes):
        self.ax = ax

    @abstractmethod
    def update(self):
        pass
