from typing import Dict, Tuple, Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, ConnectionPatch
from matplotlib.colors import to_rgb
from scipy.optimize import OptimizeResult
from os.path import join

from exp2 import Exp2
from analysis.data import Data
from config import ExperimentConfig as ExpConfig
from stimuli.motion_structure import MotionStructure
import analysis.models as models


class DataExp2(Data):
    structures = Exp2.structures
    x = np.arange(len(ExpConfig.glo_exp2))

    def __init__(self, pid):
        super().__init__(join('exp2', pid, f'{pid}_exp2.dat'))

    @staticmethod
    def permuted_structures(glo: float = ExpConfig.glo_H, λ_I: float = ExpConfig.λ_I) -> Dict[str, MotionStructure]:
        return {
            'C_012': MotionStructure(0, λ_I, permutation=[0, 1, 2]),
            'C_120': MotionStructure(0, λ_I, permutation=[1, 2, 0]),
            'C_201': MotionStructure(0, λ_I, permutation=[2, 0, 1]),
            'H_012': MotionStructure(glo, λ_I, permutation=[0, 1, 2]),
            'H_120': MotionStructure(glo, λ_I, permutation=[1, 2, 0]),
            'H_201': MotionStructure(glo, λ_I, permutation=[2, 0, 1]),
        }

    def load_model(self, Model: Type[models.Model], res: OptimizeResult) -> models.Model:
        if len(res.b) > 1:
            res.b = np.array([res.b[-1] - res.b[-2]])
        return super().load_model(Model, res)

    def plot_stacked_bar(self, ax: plt.Axes, width: float = 0.8, plot_legend: bool = True):
        p, labels = [], []
        bottom = np.zeros(len(ExpConfig.glo_exp2))

        def whiten_color(color):
            return [c + (.8 - c) / 2 for c in to_rgb(color)]
        colormap = [('C', 'high', whiten_color('darkgoldenrod')),
                    ('C', 'low', whiten_color('goldenrod')),
                    ('H', 'low', whiten_color('green')),
                    ('H', 'high', whiten_color('darkgreen'))]
        for choice, confidence, color in colormap:
            y = [len(self.df[(self.df['choice'] == choice) &
                             (self.df['confidence'] == confidence) &
                             (self.df['ground_truth'] == f'{s:.2f}')]) / (len(self.df) / len(ExpConfig.glo_exp2))
                 for s in ExpConfig.glo_exp2]
            p.append(ax.bar(self.x, y, width=width, bottom=bottom, color=color, alpha=0.85)[0])
            labels.append(f'{"C" if choice=="C" else "H"} {confidence}')
            bottom += y
        ax.set_xlim(-width / 2, self.x[-1] + width / 2)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel(r'$P$(choice=$C\,|\,\bf{X}$)')
        dx, dy = ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0))
        w, r = .9, .7 / 12
        aspect = dx / ((self.x[-1] + w) * dy - dx)
        for x in self.x:
            ax.add_patch(FancyBboxPatch((x - w/2, -aspect), w, w * aspect, boxstyle='Round,pad=0,rounding_size=0.05',
                                        fc='#E6E6E6', ec='#B3B3B3', lw=1, clip_on=False, mutation_aspect=aspect))
            nodes = []
            for dx in [-5 * r, 0, 5 * r]:
                nodes.append((x + dx, (-.9 + r) * aspect))
            nodes.append((x - 2.5 * r, -.72 * aspect))
            if x > 0:
                nodes.append((x, (-.2 - .52 * ExpConfig.glo_exp2[x]) * aspect))
            for node in nodes:
                ax.add_artist(Ellipse(node, 2 * r, 2 * r * aspect, fc='k', clip_on=False))
            ax.add_patch(ConnectionPatch(nodes[0], nodes[3], 'data', 'data', clip_on=False))
            ax.plot([nodes[0][0], nodes[3][0]], [nodes[0][1], nodes[3][1]], color='k', clip_on=False)
            ax.plot([nodes[1][0], nodes[3][0]], [nodes[1][1], nodes[3][1]], color='k', clip_on=False)
            if x == 0:
                ax.plot([nodes[2][0], nodes[2][0]], [nodes[2][1], -.2 * aspect], color='k', clip_on=False)
                ax.plot([nodes[3][0], nodes[3][0]], [nodes[3][1], -.2 * aspect], color='k', clip_on=False)
            else:
                ax.plot([nodes[2][0], nodes[4][0]], [nodes[2][1], nodes[4][1]], color='k', clip_on=False)
                ax.plot([nodes[3][0], nodes[4][0]], [nodes[3][1], nodes[4][1]], color='k', clip_on=False)
                ax.plot([nodes[4][0], nodes[4][0]], [nodes[4][1], -.2 * aspect], color='k', clip_on=False)
        ax.text(-width / 2, -0.15 * aspect, '$C$', ha='left', va='top')
        ax.text(self.x[-1] - width / 2, -0.15 * aspect, '$H$', ha='left', va='top')
        if plot_legend:
            ax.add_artist(plt.legend(p[::-1], labels[::-1], loc='lower left'))
        else:
            return p[::-1], labels[::-1]

    def plot_line_human(self) -> Tuple[np.ndarray, np.ndarray]:
        accuracy = (self.df['choice'] == 'C') * 1.0
        y_human = [accuracy[self.df['ground_truth'] == f'{s:.2f}'].mean() for s in ExpConfig.glo_exp2]
        n_human = [(self.df['ground_truth'] == f'{s:.2f}').sum() for s in ExpConfig.glo_exp2]
        err = [np.sqrt(p * (1 - p) / n) for p, n in zip(y_human, n_human)]
        return np.array(y_human), np.array(err)

    def plot_line_model(self, prediction: pd.DataFrame) -> np.ndarray:
        y = [prediction['C'][self.df['ground_truth'] == f'{s:.2f}'].mean() for s in ExpConfig.glo_exp2]
        return np.array(y)

    def score(self):
        from scipy.special import logsumexp
        df = self.apply_kalman_filters()
        df['C'] = logsumexp(df[['C_012', 'C_120', 'C_201']], axis=1)
        df['H'] = logsumexp(df[['H_012', 'H_120', 'H_201']], axis=1)
        df.loc[df['C'] > df['H'], 'ground_truth'] = 'C'
        df.loc[df['C'] < df['H'], 'ground_truth'] = 'H'
        return self._score(df)


if __name__ == '__main__':
    _, ax = plt.subplots()
    data = DataExp2('3216')
    data.plot_stacked_bar(ax)
    plt.show()
