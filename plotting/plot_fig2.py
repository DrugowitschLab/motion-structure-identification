from typing import Type
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

from exp1 import Exp1
from analysis.data_exp1 import DataExp1
import analysis.models as models
from analysis.utils.confusion_matrix import plot_confusion_matrix
from plotting.colors import *

fp = FontProperties(fname='Font Awesome 5 Free-Solid-900.otf')


def plot_2B(ax: plt.Axes):
    cm = np.zeros((len(Exp1.structures), len(Exp1.structures)))
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        cm += data.plot_confusion_matrix()
    cm /= len(DataExp1.pids)
    ticklabels = list(map(lambda s: f'${s}$', Exp1.structures))
    plot_confusion_matrix(cm, ticklabels, ticklabels, ax)
    ax.set_title('Human avg.')
    ax.set_xlabel('Choice')
    ax.set_ylabel('True Structure')
    plt.tight_layout()


def consistency(Model: Type[models.Model]):
    df_model = pd.DataFrame({s: [] for s in Exp1.structures})
    df_human = pd.DataFrame({s: [] for s in Exp1.structures})
    df_conf = pd.DataFrame({s: [] for s in Exp1.structures})
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        # model = data.build_model(Model)

        idx = data.match()
        choice = data.df['choice'].to_numpy()[idx]
        human_consistency = np.zeros(idx.shape[0] * idx.shape[1])
        human_consistency[idx[:, 0]] = human_consistency[idx[:, 1]] = choice[:, 0] == choice[:, 1]
        confidence = data.df['confidence'].to_numpy()[idx]
        confidence_consistency = np.zeros(idx.shape[0] * idx.shape[1])
        confidence_consistency[idx[:, 0]] = confidence_consistency[idx[:, 1]] = confidence[:, 0] == confidence[:, 1]
        df = data.cross_validate(Model)
        model_consistency = (df ** 2).sum(axis=1)

        df['human_consistency'] = human_consistency
        df['model_consistency'] = model_consistency
        df['confidence_consistency'] = confidence_consistency
        df['ground_truth'] = data.df['ground_truth']
        df_human = df_human.append(df.groupby('ground_truth')['human_consistency'].mean(), ignore_index=True)
        df_model = df_model.append(df.groupby('ground_truth')['model_consistency'].mean(), ignore_index=True)
        df_conf = df_conf.append(df.groupby('ground_truth')['confidence_consistency'].mean(), ignore_index=True)
    return {'human': df_human, 'model': df_model, 'confidence': df_conf}


def plot_2C(ax: plt.Axes):
    sns.set_palette(sns.color_palette([colors['consistency_human']]))
    human_consistency = []
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        choice = np.array(data.extract('choice'))[data.match()]
        human_consistency.append((choice[:, 0] == choice[:, 1]).sum() / len(choice))
    sns.boxplot(data=human_consistency, fliersize=0, ax=ax, linewidth=0.5, width=0.2)
    sns.scatterplot(x=np.linspace(-0.09, 0.09, 12), y=human_consistency,
                    fc=colors['consistency_human'], ec=sns_edge_color(colors['consistency_human']),
                    ax=ax, s=5, linewidth=0.5, zorder=11, clip_on=False, legend=False)
    # m = np.mean(human_consistency)
    # print(m, np.sqrt(m * (1 - m) / 1200))
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Choice consistency\nacross trial-repetitions')
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()


def plot_2E(ax: plt.Axes, bin_edges_human: np.ndarray =
            np.array([-1000, -90, -60, -30, -4, -2, 0, 2, 4, 30, 60, 1000]),
            bin_edges_model: np.ndarray = np.linspace(-160, 100, 40)):
    Δ, p_human, p_model = [], [], []
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        model = data.build_model(models.BayesianIdealObserver)
        df = model.predict(model.fit())
        δ = np.stack([np.log(df[s]) - np.log(np.sum(df.loc[:, df.columns != s], axis=1)) for s in data.structures])
        Δ += list(δ.T.flatten())
        p_model += list(data.cross_validate(models.ChoiceModel4Param).to_numpy().flatten())
        p_human += list(np.array([data.df['choice'] == s for s in Exp1.structures]).T.flatten())
    df = pd.DataFrame({'Δ': Δ, 'p_human': p_human, 'p_model': p_model})
    x_human, y_human, yerr_human, x_model, y_model, yerr_model = [], [], [], [], [], []
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
        y_model.append(_df['p_model'].mean())
        yerr_model.append(_df['p_model'].sem())
    ax.errorbar(x_human, y_human, yerr_human, label='Human ± sem', color=colors['decision_human'],
                fmt='.', capsize=2, ms=2, capthick=0.5, zorder=1)
    ax.plot(x_model, y_model, color=colors['decision_model'], label='Model', ms=1, zorder=0)
    ax.set_xlabel(r'logit( $P_\mathregular{ideal}$($S\,|\,\bf{X}$) )')
    ax.set_ylabel(r'$P($choice=$S\,|\,\bf{X}$)')
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',
              handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def plot_2F(ax: plt.Axes):
    cm = np.zeros((len(Exp1.structures), len(Exp1.structures)))
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        model = data.build_model(models.ChoiceModel4Param)
        pred = data.cross_validate(models.ChoiceModel4Param)
        cm += model.plot_confusion_matrix(pred)
    cm /= len(DataExp1.pids)
    ticklabels = list(map(lambda s: f'${s}$', Exp1.structures))
    plot_confusion_matrix(cm, ticklabels, ticklabels, ax)
    ax.set_title('Model avg.')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Structure')
    plt.tight_layout()


def plot_2G(ax: plt.Axes):
    # data from analysis.analyze_shuffling.ipynb
    import pickle
    with open('../data/shuffled.dat', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    plt.scatter(x, y, fc='gray', ec='k', linewidths=0.5, s=5, alpha=0.5)
    xy_min, xy_max = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], ':', color='gray')
    ax.axis('equal')
    ax.set_xlabel(r'$\mathcal{L}$(human choices)')
    ax.set_ylabel(r'$\mathcal{L}$(shuffled choices)')
    plt.yticks([-200, -150], rotation=90, va='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def plot_2H(ax: plt.Axes, pids=(3, 10)):
    pos = ax.get_position()
    ax.set_position((pos.x0, pos.y0 + 0.08, pos.width, pos.height))
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_xlabel('Human choice/model prediction', labelpad=10)
    ax.set_yticks([])
    ax.set_ylabel('True Structure', labelpad=16)
    fig = plt.gcf()
    gs = gridspec.GridSpecFromSubplotSpec(len(pids), 2, subplot_spec=ax, wspace=0.1, hspace=0.1)
    mpl.rcParams['font.size'] -= 2
    mpl.rcParams['xtick.labelsize'] -= 2
    mpl.rcParams['ytick.labelsize'] -= 2
    for j in range(len(pids)):
        ax = plt.Subplot(fig, gs[0, j])
        data = DataExp1(DataExp1.pids[pids[j] - 1])
        data.plot_confusion_matrix(ax)
        ax.set_title(f'\uf007$\#${pids[j]}', size=7, fontproperties=fp)
        ax.set_xticks([])
        ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel('Human')
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
            # ax.yaxis.set_label_coords(-0.4, 0.6 - i * 0.2)
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, gs[1, j])
        model = data.build_model(models.ChoiceModel4Param)
        model.plot_confusion_matrix(data.cross_validate(models.ChoiceModel4Param), ax)
        ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel('Model')
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        fig.add_subplot(ax)
    mpl.rcParams['font.size'] += 2
    mpl.rcParams['xtick.labelsize'] += 2
    mpl.rcParams['ytick.labelsize'] += 2


# def plot_2I(ax: plt.Axes, bin_edges: np.ndarray = np.linspace(0, 1, 11)):
#     # pos = ax.get_position()
#     # ax.set_position((pos.x0 + 0.1, pos.y0, pos.width, pos.height))
#     x, y = [], []
#     for pid in DataExp1.pids:
#         data = DataExp1(pid)
#         model = data.build_model(models.HumanObserver4Param)
#         df = model.predict(model.fit())
#         df['choice'] = model.df['choice']
#         df['confidence'] = (model.df['confidence'] == 'high').astype(float)
#         df['p_model'] = df.apply(lambda row: row[row['choice']], axis=1)
#         x += df['p_model'].tolist()
#         y += df['confidence'].tolist()
#     df = pd.DataFrame({'p_model': x, 'confidence': y})
#     x, y, yerr = [], [], []
#     df['bin'] = pd.cut(df['p_model'], bin_edges, labels=False)
#     for i in range(len(bin_edges) - 1):
#         _df = df[df['bin'] == i]
#         if len(_df) == 0:
#             continue
#         x.append((bin_edges[i] + bin_edges[i + 1]) / 2)
#         y.append(_df['confidence'].mean())
#         yerr.append(_df['confidence'].sem())
#     ax.errorbar(x, y, yerr, label='Human $\pm$ sem', c='darkgreen', fmt='.-', capsize=2, ms=2, capthick=0.5)
#     model = LinearRegression().fit(np.array(x).reshape((-1, 1)), np.array(y))
#     x = np.linspace(0, 1, 100)
#     y = model.coef_ * x + model.intercept_
#     print(model.coef_)
#     print(model.intercept_)
#     ax.plot(x, y, ls='--', c='gray')
#     ax.set_xticks(bin_edges[::2])
#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(['Low(0)', 'High(1)'])
#     ax.set_ylabel('Avg. reported\nconfidence', labelpad=-20)
#     ax.set_xticks([0, 0.5, 1])
#     ax.set_xlabel(r'$P($choice=$S|\bf{X})$')
#     ax.legend(loc='lower right', handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.tight_layout()


def plot_2I(ax: plt.Axes,
            bin_edges: np.ndarray = np.array([-1000, -40, -20, -10, -4.5, -1.5, 1.5, 4.5, 10, 20, 40, 1000])):
    x, y = [], []
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        model = data.build_model(models.ChoiceModel4Param)
        res = model.fit()
        L = model.L + np.repeat([0] + list(res.b), model.multiplicity) + model.L_uniform
        x += list(logsumexp(L, b=model.is_chosen, axis=1) - logsumexp(L, b=1 - model.is_chosen, axis=1))
        y += list((model.df['confidence'] == 'high').astype(float))
    df = pd.DataFrame({'x': x, 'y': y})
    df['bin'] = pd.cut(df['x'], bin_edges, labels=False)
    x, y, yerr = [], [], []
    for i in range(len(bin_edges) - 1):
        _df = df[df['bin'] == i]
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
    ax.set_xlabel(r'logit$(\,P(S\,|\,\bf{X}$$)\,)$')
    ax.legend(loc='lower right', handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


if __name__ == '__main__':
    from analysis.utils import svg_layout
    from analysis.utils.svg_layout import Panel, PanelLabel, Figure
    import plotting.matplotlib_config

    svg_layout.working_dir = 'figs'
    panels = [
        Panel('2A.svg', 00.1, 0.3, 13.0, 02.6),
        Panel('2B.svg', 00.1, 3.1, 04.2, 04.2, plot_2B),
        Panel('2C.svg', 04.4, 3.1, 03.6, 04.2, plot_2C),
        Panel('2D.svg', 08.8, 3.1, 04.2, 03.0),
        Panel('2E.svg', 13.4, 0.3, 05.6, 06.2, plot_2E),
        Panel('2F.svg', 00.1, 7.7, 04.2, 04.2, plot_2F),
        Panel('2G.svg', 04.4, 7.7, 04.2, 04.2, plot_2G),
        Panel('2H.svg', 08.8, 6.5, 05.6, 05.6, plot_2H),
        Panel('2I.svg', 14.0, 6.5, 05.0, 05.6, plot_2I),
    ]
    labels = PanelLabel.generate_labels(len(panels), 0, 0.3, style={'font-size': '18', 'font-family': 'sans-serif'})
    Figure(19, 12.0, panels, labels).plot('Fig 2.svg')
    # panels[6].preview()
