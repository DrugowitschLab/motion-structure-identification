from typing import List
import pickle
from os.path import exists
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

from exp1 import Exp1
from analysis.data_exp1 import DataExp1
from analysis.data_exp2 import DataExp2
import analysis.models as models
from plotting.colors import *


def exp1all(ax: plt.Axes):
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_xlabel('Human choice/model prediction', labelpad=10)
    ax.set_yticks([])
    ax.set_ylabel('True structure', labelpad=20)
    fig = plt.gcf()
    n_col = len(DataExp1.pids) // 2
    outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax, wspace=0.1, hspace=0.2)
    half1 = gridspec.GridSpecFromSubplotSpec(2, n_col, subplot_spec=outer[0, 0], wspace=0.1, hspace=0.1)
    half2 = gridspec.GridSpecFromSubplotSpec(2, n_col, subplot_spec=outer[1, 0], wspace=0.1, hspace=0.1)
    mpl.rcParams['font.size'] -= 2
    mpl.rcParams['xtick.labelsize'] -= 2
    mpl.rcParams['ytick.labelsize'] -= 2
    for i in range(len(DataExp1.pids)):
        half = half1 if i < n_col else half2
        ax = plt.Subplot(fig, half[0, i % n_col])
        data = DataExp1(DataExp1.pids[i])
        data.plot_confusion_matrix(ax)
        ax.set_title(f'#{i + 1}')
        ax.set_xlabel('')
        ax.set_xticks([])
        if i % n_col == 0:
            ax.set_ylabel('Human')
        else:
            ax.set_ylabel('')
            ax.set_yticks([])
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, half[1, i % n_col])
        model = data.build_model(models.ChoiceModel4Param)
        model.plot_confusion_matrix(data.cross_validate(models.ChoiceModel4Param), ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if i % n_col == 0:
            ax.set_ylabel('Model')
        else:
            ax.set_ylabel('')
            ax.set_yticks([])
        fig.add_subplot(ax)
    mpl.rcParams['font.size'] += 2
    mpl.rcParams['xtick.labelsize'] += 2
    mpl.rcParams['ytick.labelsize'] += 2


def proximity(ax: plt.Axes):
    proximity_all, accuracy_all = [], []
    for pid in DataExp1.pids:
        proximity, accuracy = [], []
        data = DataExp1(pid).data
        for trial in data:
            if trial['ground_truth'] == 'C':
                x = np.array(trial['φ'])[:, :3]
                dx = np.abs(x[:, 0] - x[:, 1])
                proximity.append(np.minimum(dx, 2 * np.pi - dx).mean())
                accuracy.append(trial['choice'] == 'C')
        proximity_all += proximity
        accuracy_all += accuracy
        df = pd.DataFrame({'proximity': proximity, 'accuracy': accuracy})
        df['accuracy'] = df['accuracy'].astype(float)
        print(pearsonr(df['proximity'], df['accuracy']))
    proximity_all, accuracy_all = np.array(proximity_all), np.array(accuracy_all)
    ax.boxplot([proximity_all[~accuracy_all], proximity_all[accuracy_all]])
    ax.set_xticklabels(['Non-$C$', '$C$'])
    ax.set_xlabel('Human choice')
    ax.set_ylabel('Avg. angular distance b/w clustered dots')


def cross_evaluation(ax: plt.Axes, file='../data/cross_evaluation.dat'):
    if exists(file):
        with open(file, 'rb') as f:
            cm = pickle.load(f)
    else:
        cm = np.zeros((12, 12))
        for i in range(len(DataExp1.pids)):
            res = DataExp1(DataExp1.pids[i]).build_model(models.ChoiceModel4Param).fit()
            for j in range(len(DataExp1.pids)):
                cm[i][j] = DataExp1(DataExp1.pids[j]).load_model(models.ChoiceModel4Param, res).fit().log_likelihood
        cm = np.exp(cm - np.max(cm, axis=0))
        with open(file, 'wb+') as f:
            pickle.dump(cm, f)
    im = ax.imshow(cm)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(DataExp1.pids)))
    ax.set_yticks(np.arange(len(DataExp1.pids)))
    ax.set_xticklabels(map(str, np.arange(1, len(DataExp1.pids) + 1)), size=6)
    ax.set_yticklabels(map(str, np.arange(1, len(DataExp1.pids) + 1)), size=6)
    ax.set_xlabel('Participant fitted')
    ax.set_ylabel('Participant predicted')
    plt.tight_layout()


def binned_plot(xdata: List[float], ydata: List[float], q: int, ax: plt.Axes):
    df = pd.DataFrame({'x': xdata, 'y': ydata})
    df['bin'] = pd.qcut(xdata, q, labels=False, duplicates='drop')
    x, y, yerr = [], [], []
    for i in range(q):
        _df = df[df['bin'] == i]
        x.append(_df['x'].mean())
        y.append(_df['y'].mean())
        yerr.append(_df['y'].sem())
    ax.errorbar(x, y, yerr, label='Human $\pm$ sem', c='darkgreen', fmt='.-', capsize=2, ms=2, capthick=0.5)


def consistency(ax: plt.Axes):
    from plotting.plot_fig2 import consistency
    sns.set_palette(sns.color_palette([colors['consistency_human'], colors['consistency_model']]))
    dfs = consistency(models.ChoiceModel4Param)
    df = pd.DataFrame(
        [[dfs[a][s][p], s, a] for a in ['human', 'model'] for s in Exp1.structures for p in range(len(DataExp1.pids))],
        columns=['Consistency', 'True structure', 'Agent']
    )
    sns.boxplot(data=df, x='True structure', y='Consistency', hue='Agent', fliersize=0, ax=ax, linewidth=0.5)
    # sns.stripplot(data=df, x='True structure', y='Consistency', hue='Agent', jitter=True, ax=ax, dodge=True,
    #               size=2.5, linewidth=0.5, zorder=11, clip_on=False)

    ax.set_xticklabels(map(lambda s: f'${s}$', Exp1.structures))
    ax.set_ylim(0, 1)
    dx, dy, y = 0.2, 0.05, 0.3
    jitter = 0.8
    for i in range(len(Exp1.structures)):
        x1 = np.linspace(i - (1 + jitter) * dx, i - (1 - jitter) * dx, 12)
        y1 = dfs['human'][Exp1.structures[i]]
        sns.scatterplot(x=x1, y=y1,
                        fc=colors['consistency_human'], ec=sns_edge_color(colors['consistency_human']),
                        ax=ax, s=10, linewidth=0.5, zorder=11, clip_on=False, legend=False)
        x2 = np.linspace(i + (1 - jitter) * dx, i + (1 + jitter) * dx, 12)
        y2 = dfs['model'][Exp1.structures[i]]
        sns.scatterplot(x=x2, y=y2,
                        fc=colors['consistency_model'], ec=sns_edge_color(colors['consistency_model']),
                        ax=ax, s=10, linewidth=0.5, zorder=11, clip_on=False, legend=False)
        for _x1, _y1, _x2, _y2 in zip(x1, y1, x2, y2):
            ax.plot([_x1, _x2], [_y1, _y2], 'k-', lw=0.1, zorder=10)
        ax.plot([i - dx, i - dx, i + dx, i + dx], [y, y - dy, y - dy, y], 'k')
        r = spearmanr(dfs['human'][Exp1.structures[i]], dfs['model'][Exp1.structures[i]])[0]
        ax.text(i, y - 2 * dy, f'ρ={r:.2f}', ha='center', va='top', fontsize=6)
    plt.legend([Line2D([0], [0], marker='o', mec='k', mfc=colors['consistency_human'], ms=2.5, mew=.5, ls='None'),
                Line2D([0], [0], marker='o', mec='k', mfc=colors['consistency_model'], ms=2.5, mew=.5, ls='None')],
               ['Human', 'Model'], loc='upper right')
    ax.set_ylabel('Choice consistency across trial-repetitions')
    plt.tight_layout()


def confidence_consistency(ax: plt.Axes):
    from plotting.plot_fig2 import consistency
    sns.set_palette(sns.color_palette([colors['consistency_human'], colors['consistency_model']]))
    dfs = consistency(models.ChoiceModel4Param)
    df = pd.DataFrame(
        [[dfs[a][s][p], s, a] for a in ['human', 'confidence'] for s in Exp1.structures for p in range(len(DataExp1.pids))],
        columns=['Consistency', 'True structure', 'Agent']
    )
    sns.boxplot(data=df, x='True structure', y='Consistency', hue='Agent', fliersize=0, ax=ax, linewidth=0.5)
    # sns.stripplot(data=df, x='True structure', y='Consistency', hue='Agent', jitter=True, ax=ax, dodge=True,
    #               size=2.5, linewidth=0.5, zorder=11, clip_on=False)

    ax.set_ylim(0, 1)
    dx, dy, y = 0.2, 0.05, 0.3
    jitter = 0.8
    for i in range(len(Exp1.structures)):
        x1 = np.linspace(i - (1 + jitter) * dx, i - (1 - jitter) * dx, 12)
        y1 = dfs['human'][Exp1.structures[i]]
        sns.scatterplot(x=x1, y=y1,
                        fc=colors['consistency_human'], ec=sns_edge_color(colors['consistency_human']),
                        ax=ax, s=10, linewidth=0.5, zorder=11, clip_on=False, legend=False)
        x2 = np.linspace(i + (1 - jitter) * dx, i + (1 + jitter) * dx, 12)
        y2 = dfs['confidence'][Exp1.structures[i]]
        sns.scatterplot(x=x2, y=y2,
                        fc=colors['consistency_model'], ec=sns_edge_color(colors['consistency_model']),
                        ax=ax, s=10, linewidth=0.5, zorder=11, clip_on=False, legend=False)
        for _x1, _y1, _x2, _y2 in zip(x1, y1, x2, y2):
            ax.plot([_x1, _x2], [_y1, _y2], 'k-', lw=0.1, zorder=10)
        ax.plot([i - dx, i - dx, i + dx, i + dx], [y, y - dy, y - dy, y], 'k')
        r = spearmanr(dfs['human'][Exp1.structures[i]], dfs['confidence'][Exp1.structures[i]])[0]
        ax.text(i, y - 2 * dy, f'ρ={r:.2f}', ha='center', va='top', fontsize=6)
    plt.legend([Line2D([0], [0], marker='o', mec='k', mfc=colors['consistency_human'], ms=2.5, mew=.5, ls='None'),
                Line2D([0], [0], marker='o', mec='k', mfc=colors['consistency_model'], ms=2.5, mew=.5, ls='None')],
               ['Choice', 'Confidence'], loc='upper right')
    ax.set_ylabel('Consistency across trial-repetitions')
    plt.tight_layout()


def exp2all():
    n_col = len(DataExp2.pids) // 3
    _, axes = plt.subplots(3, n_col, figsize=(7.5, 5))
    h, l = None, None
    for i in range(len(DataExp2.pids)):
        pid = DataExp2.pids[i]
        ax = axes[i // n_col, i % n_col]
        ax.set_title(f'#{i + 1}')
        data = DataExp2(pid)
        mpl.rcParams['font.size'] -= 2
        h, l = data.plot_stacked_bar(ax, plot_legend=False)
        mpl.rcParams['font.size'] += 2
        y_human, err = data.plot_line_human()
        ax.errorbar(DataExp2.x, y_human, err, label='Human $\pm$ sem', color=colors['decision_human'],
                    capsize=3, capthick=1, lw=1, ms=2, fmt='o', zorder=3)
        m1 = data.load_model(models.ChoiceModel4Param, DataExp1(pid).build_model(models.ChoiceModel4Param).fit())
        y1 = data.plot_line_model(m1.predict(m1.fit()))
        ax.plot(DataExp2.x, y1, 'o--', label='Transfer model', color=colors['decision_transfer'], lw=1, ms=2, zorder=2)
        y2 = data.plot_line_model(data.cross_validate(models.ChoiceModel4Param))
        ax.plot(DataExp2.x, y2, 'o-', label='Fitted model', color=colors['decision_model'], lw=1, ms=2, zorder=2)
        handles, labels = ax.get_legend_handles_labels()
        h += handles[::-1]
        l += labels[::-1]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(' ', labelpad=18)
    plt.gcf().legend(h, l, loc='lower center', ncol=7,
                     handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.35)})
    plt.tight_layout()


def bayesian_confidence(ax: plt.Axes,
                        bin_edges: np.ndarray = np.array([-1000, -40, -20, -10, -4.5, -1.5, 1.5, 4.5, 10, 20, 40, 1000])):
    x, y = [], []
    for pid in DataExp1.pids:
        data = DataExp1(pid)
        model = data.build_model(models.ChoiceModel4Param)
        L = model.L + model.L_uniform
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
    ax.set_xlabel(r'logit$(\,P_{\mathregular{ideal}}(S\,|\,\bf{X}$$)\,)$')
    ax.legend(loc='lower right', handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.25)})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()


def model_against_chance(ax: plt.Axes):
    from plotting.plot_fig4 import Models, L
    from config import ExperimentConfig
    baseline = - np.log(len(DataExp1.structures)) * ExperimentConfig.n_trials_exp1
    x = np.arange(1, len(DataExp1.pids) + 1)
    ax.text(0.5, 0, 'Random\nchoice', ha='left', va='center', fontsize=8)
    ax.hlines(0, 0, len(DataExp1.pids) + 1, colors='k')
    for Model in Models:
        ax.scatter(x, np.array(L[Model]) - np.array(baseline), marker=Model.marker, color=Model.color, label=Model.name)
    ax.set_xlabel('Participant')
    ax.set_xticks(np.arange(1, len(DataExp1.pids) + 1))
    ax.set_xticklabels(map(str, np.arange(1, len(DataExp1.pids) + 1)), size=6)
    ax.set_xlim(0.5, len(DataExp1.pids) + 0.5)
    ax.set_ylabel(r'$\mathcal{L}$(model) $-$ $\mathcal{L}$(chance)')
    ax.set_ylim(-50, 170)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.02), frameon=False, ncol=2, handlelength=1, handletextpad=0.5)
    plt.tight_layout()


def correlation(ax: plt.Axes, file='../data/correlation.dat'):
    if exists(file):
        with open(file, 'rb') as f:
            df = pickle.load(f)
    else:
        ρ01, ρ02, ρ12, g = [], [], [], []
        for pid in DataExp1.pids:
            print(pid)
            data = DataExp1(pid)
            for i in data.idx:
                g.append(data.data[i]['ground_truth'])
                V = data.empirical_velocity()[i][:, data.data[i]['permutation']]
                ρ01.append(pearsonr(V[:, 0], V[:, 1])[0])
                ρ02.append(pearsonr(V[:, 0], V[:, 2])[0])
                ρ12.append(pearsonr(V[:, 1], V[:, 2])[0])
        df = pd.DataFrame({'ρ01': ρ01, 'ρ02': ρ02, 'ρ12': ρ12, 'g': g})
        with open(file, 'wb+') as f:
            pickle.dump(df, f)

    ρ_g, ρ_c = Exp1.presets['H'].Σ[0, 1:3] / 4
    R = {'I': ([0], [0], [0]),
         'G': ([ρ_g], [ρ_g], [ρ_g]),
         'C': ([ρ_g, 0, 0], [0, ρ_g, 0], [0, 0, ρ_g]),
         'H': ([ρ_g, ρ_c, ρ_c], [ρ_c, ρ_g, ρ_c], [ρ_c, ρ_c, ρ_g])}
    for s, c, m in zip(Exp1.structures, ['b', 'g', 'y', 'r'], ['x', 'o', '^', '+']):
        _df = df[df['g'] == s]
        alpha = 0.3 if s == 'G' else 0.1
        ax.scatter(_df['ρ01'], _df['ρ02'], _df['ρ12'], color=c, marker=m, alpha=alpha, s=16, label=s, linewidths=0.8)
        ax.scatter(R[s][0], R[s][1], R[s][2], color=c, marker=m, alpha=0.8, s=128, linewidths=0.8)

    ax.view_init(30, -45)
    ax.set_xlabel(r'$ρ_\mathregular{blue, green}$', labelpad=-4)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel(r'$ρ_\mathregular{blue, red}$', labelpad=-4)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zlabel(r'$ρ_\mathregular{green, red}$', labelpad=-4)
    ax.set_zlim(-1, 1)
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    legend = ax.legend(loc='lower right', handlelength=1, handletextpad=0)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.tight_layout()


if __name__ == '__main__':
    from analysis.utils.svg_layout import px_per_unit
    import plotting.matplotlib_config

    def cm2in(inch):
        return inch * px_per_unit['cm'] / px_per_unit['in']

    _, ax = plt.subplots(figsize=(cm2in(9), cm2in(6)))
    proximity(ax)
    plt.savefig('./figs/proximity.pdf')

    _, ax = plt.subplots(figsize=(cm2in(14), cm2in(10)))
    exp1all(ax)
    plt.savefig('./figs/exp1all.pdf')

    _, ax = plt.subplots(figsize=(cm2in(6), cm2in(5)))
    cross_evaluation(ax)
    plt.savefig('./figs/cross_evaluation.pdf')

    _, _ = plt.subplots(figsize=(cm2in(14), cm2in(10)))
    exp2all()
    plt.savefig('./figs/exp2all.pdf')

    _, ax = plt.subplots(figsize=(cm2in(14), cm2in(10)))
    consistency(ax)
    plt.savefig('./figs/consistency.pdf')

    # _, ax = plt.subplots(figsize=(cm2in(14), cm2in(10)))
    # confidence_consistency(ax)
    # plt.savefig('./figs/confidence_consistency.png', transparent=True)

    _, ax = plt.subplots(figsize=(cm2in(5.0), cm2in(5.6)))
    bayesian_confidence(ax)
    plt.savefig('./figs/confidence.pdf')

    _, ax = plt.subplots(figsize=(cm2in(6.7), cm2in(4.8)))
    model_against_chance(ax)
    plt.savefig('./figs/model_against_chance.pdf')

    fig = plt.figure(figsize=(cm2in(14), cm2in(12)))
    ax = fig.add_subplot(111, projection='3d')
    correlation(ax)
    plt.savefig('./figs/correlation.pdf')
