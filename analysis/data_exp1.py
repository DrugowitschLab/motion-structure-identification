from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from analysis.data import Data
from exp1 import Exp1
from config import ExperimentConfig as ExpConfig

from stimuli.motion_structure import MotionStructure


class DataExp1(Data):
    structures = Exp1.structures

    def __init__(self, pid):
        super().__init__(join('exp1', pid, f'{pid}_exp1.dat'))

    def plot_confusion_matrix(self, ax: Optional[plt.Axes] = None) -> np.ndarray:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.df['ground_truth'], self.df['choice'], Exp1.structures)
        if ax:
            from analysis.utils.confusion_matrix import plot_confusion_matrix
            ticklabels = list(map(lambda s: f'${s}$', Exp1.structures))
            plot_confusion_matrix(cm, ticklabels, ticklabels, ax)
            ax.set_xlabel('Choice')
            ax.set_ylabel('True Structure')
        return cm

    @staticmethod
    def permuted_structures(glo: float = ExpConfig.glo_H, λ_I: float = ExpConfig.λ_I) -> Dict[str, MotionStructure]:
        return {
            'I': MotionStructure(1, 2),
            'G': MotionStructure(1, λ_I),
            'C_012': MotionStructure(0, λ_I, permutation=[0, 1, 2]),
            'C_120': MotionStructure(0, λ_I, permutation=[1, 2, 0]),
            'C_201': MotionStructure(0, λ_I, permutation=[2, 0, 1]),
            'H_012': MotionStructure(glo, λ_I, permutation=[0, 1, 2]),
            'H_120': MotionStructure(glo, λ_I, permutation=[1, 2, 0]),
            'H_201': MotionStructure(glo, λ_I, permutation=[2, 0, 1]),
        }

    def match(self):
        matches = []
        seeds = np.array(self.extract('seed'))
        for seed in np.unique(seeds):
            matches.append(np.where(seeds == seed)[0])
        return np.array(matches)

    def score(self):
        # return self._score(self.df.loc[self.idx])
        return self._score(self.apply_kalman_filters())
