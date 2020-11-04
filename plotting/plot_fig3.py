import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

from config import ExperimentConfig as ExpConfig
from analysis.data import pool
from analysis.data_exp1 import DataExp1
from analysis.data_exp2 import DataExp2
import analysis.models as models
from plotting.colors import *


def plot_3B(ax: plt.Axes):
    data = pool(DataExp2)
    data.plot_stacked_bar(ax)
    n = len(ExpConfig.glo_exp2)
    y_human, y1, y2 = np.zeros(n), np.zeros(n), np.zeros(n)
    for pid in DataExp2.pids:
        data = DataExp2(pid)
        y_human += data.plot_line_human()[0]
        m1 = data.load_model(models.ChoiceModel4Param, DataExp1(pid).build_model(models.ChoiceModel4Param).fit())
        y1 += data.plot_line_model(m1.predict(m1.fit()))
        y2 += data.plot_line_model(data.cross_validate(models.ChoiceModel4Param))
    y_human, y1, y2 = y_human / len(DataExp2.pids), y1 / len(DataExp2.pids), y2 / len(DataExp2.pids)
    err = [np.sqrt(p * (1 - p) / len(DataExp2.pids) / 20) for p in y_human]
    ax.errorbar(DataExp2.x, y_human, err, label='Human $\pm$ sem', color=colors['decision_human'],
                capsize=5, capthick=1, lw=1, ms=3, fmt='o', zorder=3)
    ax.plot(DataExp2.x, y1, 'o--', label='Transfer model', color=colors['decision_transfer'], lw=1, ms=3, zorder=2)
    ax.plot(DataExp2.x, y2, 'o-', label='Fitted model', color=colors['decision_model'], lw=1, ms=3, zorder=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right',
              handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.35)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def plot_3C(ax: plt.Axes, bin_edges_human: np.ndarray =
            np.array([-1000, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 1000]),
            bin_edges_model: np.ndarray = np.array([-1000, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 1000])):
    Δ, p_human, p1, p2 = [], [], [], []
    for pid in DataExp2.pids:
        data = DataExp2(pid)
        model = data.build_model(models.BayesianIdealObserver)
        df = model.predict(model.fit())
        Δ += list(np.log(df['C']) - np.log(df['H']))
        model = data.load_model(models.ChoiceModel4Param, DataExp1(pid).build_model(models.ChoiceModel4Param).fit())
        p1 += list(model.predict(model.fit())['C'])
        p2 += list(data.cross_validate(models.ChoiceModel4Param)['C'])
        p_human += list((data.df['choice'] == 'C') * 1.0)
    df = pd.DataFrame({'Δ': Δ, 'p_human': p_human, 'p1': p1, 'p2': p2})
    x_human, y_human, yerr_human, x_model, y1, yerr1, y2, yerr2 = [], [], [], [], [], [], [], []
    df['bin'] = pd.cut(df['Δ'], bin_edges_human, labels=False)
    for i in range(len(bin_edges_human) - 1):
        _df = df[df['bin'] == i]
        x_human.append(_df['Δ'].mean())
        y_human.append(_df['p_human'].mean())
        yerr_human.append(_df['p_human'].sem())
    df['bin'] = pd.cut(df['Δ'], bin_edges_model, labels=False)
    for i in range(len(bin_edges_model) - 1):
        _df = df[df['bin'] == i]
        x_model.append(_df['Δ'].mean())
        y1.append(_df['p1'].mean())
        yerr1.append(_df['p1'].sem())
        y2.append(_df['p2'].mean())
        yerr2.append(_df['p2'].sem())
    ax.errorbar(x_human, y_human, yerr_human, label='Human $\pm$ sem', color=colors['decision_human'],
                fmt='.', capsize=2, ms=2, capthick=0.5, zorder=1)
    ax.plot(x_model, y1, '--', color=colors['decision_transfer'], label='Transfer model', ms=1, zorder=0)
    ax.plot(x_model, y2, '-', color=colors['decision_model'], label='Fitted model', ms=1, zorder=0)
    ax.set_xlabel(r'logit( $P_\mathregular{ideal}(S=C\,|\,\bf{X}$) )')
    ax.set_ylabel(r'$P$(choice=$C\,|\,\bf{X}$)')
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right',
              handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def plot_3D(ax: plt.Axes,
            # bin_edges = np.array([-100, -3, -1.5, 1.5, 3, 7, 100])):
            bin_edges = np.array([-1000, -4.5, -1.5, 1.5, 4.5, 10, 1000])):
    x, y = [], []
    for pid in DataExp2.pids:
        data = DataExp2(pid)
        model = data.build_model(models.ChoiceModel4Param)
        L = model.L + np.repeat([0] + list(model.fit().b), model.multiplicity) + model.L_uniform
        x += list(logsumexp(L, b=model.is_chosen, axis=1) - logsumexp(L, b=1 - model.is_chosen, axis=1))
        y += list((model.df['confidence'] == 'high').astype(float))
    df = pd.DataFrame({'x': x, 'y': y})
    x, y, yerr = [], [], []
    df['bin'] = pd.cut(df['x'], bin_edges, labels=False)
    for i in range(len(bin_edges) - 1):
        _df = df[df['bin'] == i]
        print(len(_df))
        if len(_df) == 0:
            continue
        x.append(_df['x'].mean())
        y.append(_df['y'].mean())
        yerr.append(_df['y'].sem())
    ax.errorbar(x, y, yerr, label='Human $\pm$ sem', c='darkgreen', fmt='.-', capsize=2, ms=2, capthick=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Low', 'High'])
    ax.set_ylabel('Avg. reported\nconfidence', labelpad=-16)
    # ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel(r'logit$(\,P(S\,|\bf{X}$$)\,)$')
    ax.legend(loc='lower right', handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def plot_3E(ax: plt.Axes):
    sns.set_palette(sns.color_palette(['white']))
    L = []
    for pid in DataExp2.pids:
        data = DataExp2(pid)
        m1 = data.load_model(models.ChoiceModel4Param, DataExp1(pid).build_model(models.ChoiceModel4Param).fit())
        df = data.cross_validate(models.ChoiceModel4Param)
        df['choice'] = data.df['choice']
        L.append(m1.fit().log_likelihood - np.log(df.apply(lambda row: row[row['choice']], axis=1)).sum())
    sns.boxplot(data=L, fliersize=0, ax=ax, linewidth=0.5, width=0.2)
    sns.scatterplot(x=np.linspace(-0.09, 0.09, 12), y=L,
                    fc='white', ec=sns_edge_color('white'),
                    ax=ax, s=5, linewidth=0.5, zorder=11, clip_on=False, legend=False)
    # sns.stripplot(data=L, jitter=True, ax=ax, size=2.5, linewidth=0.5, zorder=10, clip_on=False)
    ax.axhline(0, 0, 1, color='k', zorder=1)
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylabel(r'$\mathcal{L}$(transfer) - $\mathcal{L}$(fitted)')
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()


if __name__ == '__main__':
    from analysis.utils import svg_layout
    from analysis.utils.svg_layout import Panel, PanelLabel, Figure
    import plotting.matplotlib_config

    svg_layout.working_dir = 'figs'
    panels = [
        Panel('3A.svg', 00.3, 0.3, 9.0, 2.6),
        Panel('3B.svg', 00.3, 3.3, 9.0, 8.0, plot_3B),
        Panel('3C.svg', 09.7, 0.3, 9.0, 6.0, plot_3C),
        Panel('3D.svg', 09.7, 6.5, 5.4, 4.6, plot_3D),
        Panel('3E.svg', 15.5, 6.5, 3.2, 4.6, plot_3E),
    ]
    labels = PanelLabel.generate_labels(len(panels), -0.2, 0.2, style={'font-size': '18', 'font-family': 'sans-serif'})
    Figure(19, 11, panels, labels).plot('Fig 3.svg')
