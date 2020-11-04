from typing import Dict, List, Type
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_exp1 import DataExp1
import analysis.models as models
from analysis.utils.groupBMC import GroupBMC


Models = [models.ChoiceModel4Param, models.BiasFreeChoiceModel,
          models.LapseFreeChoiceModel, models.NonBayesianChoiceModel4Param]
L: Dict[Type[models.Model], List[float]] = {Model: [] for Model in Models}
for pid in DataExp1.pids:
    data = DataExp1(pid)
    for Model in Models:
        df = data.cross_validate(Model)
        df['choice'] = data.df['choice']
        L[Model].append(np.log(df.apply(lambda row: row[row['choice']], axis=1)).sum())


def plot_4A(ax: plt.Axes):
    x = np.arange(1, len(DataExp1.pids) + 1)
    ax.hlines(0, 0, len(DataExp1.pids) + 1, label='Full model', colors=models.ChoiceModel4Param.color)
    baseline = np.array(L[models.ChoiceModel4Param])
    for Model in Models[1:]:
        ax.scatter(x, np.array(L[Model]) - baseline, marker=Model.marker, color=Model.color, label=Model.name)
    ax.set_xlabel('Participant')
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylabel(r'$\mathcal{L}$(model) $-$ $\mathcal{L}$(full model)')
    ax.set_ylim(-90, 10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=2, handlelength=1, handletextpad=0.5)


def plot_4B(ax: plt.Axes):
    df = pd.DataFrame(L).transpose()
    res = GroupBMC(df.to_numpy()).get_result()
    ax.bar(np.arange(4), res.protected_exceedance_probability, color='white', edgecolor='k')
    ax.set_ylim(0, 1)
    ax.set_ylabel('PXP')
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([Model.name for Model in Models], rotation=22.5)


if __name__ == '__main__':
    from analysis.utils import svg_layout
    from analysis.utils.svg_layout import Panel, PanelLabel, Figure
    import plotting.matplotlib_config

    svg_layout.working_dir = 'figs'
    panels = [
        Panel('4A.svg', 00.4, 0.0, 6.7, 4.8, plot_4A),
        Panel('4B.svg', 07.3, 0.0, 4.8, 4.8, plot_4B),
    ]
    labels = PanelLabel.generate_labels(len(panels), -0.3, 0.5, style={'font-size': '18', 'font-family': 'sans-serif'})
    Figure(12, 5.2, panels, labels).plot('Fig 4.svg')

