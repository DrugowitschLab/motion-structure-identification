import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import logsumexp
from scipy.stats import sem
from sklearn.model_selection import train_test_split
import numpy as np

from exp1 import Exp1
from analysis._data_exp1 import DataExp1
from analysis._data_exp2 import DataExp2, DataMetaExp2
from response.visual import draw_structure
from analysis.utils.confusion_matrix import plot_confusion_matrix
from analysis.analyze_exp1 import fit_meta_human


def plot_structures(self):
    for structure in Exp1.structures:
        _, ax = plt.subplots(figsize=(0.27, 0.3))
        draw_structure(ax, (0, 0, 0.27, 0.3), structure)
        ax.axis('off')
        ax.axis([0, 0.27, 0, 0.3])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'/home/sichao/Documents/Motion/code/sichao/stimuli/{structure}.svg', transparent=True)
        plt.show()


def plot_confusion_matrix_human(pid):
    data = DataExp1(pid)
    _, ax = plt.subplots(figsize=(0.714, 0.714))
    data.plot_confusion_matrix(ax)
    ax.axis('off')
    ax.set_title('')
    ax.axis([-0.5, 3.5, 3.5, -0.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'../data/{pid}/{pid}_exp1.svg', transparent=True)


def plot_confusion_matrix_model(pid, train_size=0.5, reps=10):
    data = DataExp1(pid)
    _, ax = plt.subplots(figsize=(0.714, 0.714))
    cms = []
    for _ in range(reps):
        data.idx = np.arange(data.n_trials)
        idx_train, idx_test = train_test_split(data.idx, train_size=0.5)
        data.idx = idx_train
        res = data.build_model(data.n_params_4, data.params_4).fit()
        data.idx = idx_test
        cm = data.load_model(res).plot_confusion_matrix(res)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cms.append(cm)
    cm = np.array(cms).mean(axis=0)
    plot_confusion_matrix(labels=Exp1.structures, cm=cm, ax=ax)
    ax.axis('off')
    ax.set_title('')
    ax.axis([-0.5, 3.5, 3.5, -0.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'../data/{pid}/{pid}_exp1_4_param_split.svg', transparent=True)
    # plt.show()


def plot_ax(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    lw = 0.5
    ohg = 0.1
    # manual arrowhead width and length
    xhw = 1. / 40. * (ymax - ymin)
    xhl = 1. / 40. * (xmax - xmin)
    # compute matching arrowhead length and width
    yhw = xhw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = xhl / (xmax - xmin) * (ymax - ymin) * width / height
    # draw x and y axis
    ax.arrow(xmin, ymin, xmax - xmin, 0., fc='k', lw=lw,
             head_width=xhw, head_length=xhl, overhang=ohg,
             length_includes_head=True, clip_on=False)
    ax.arrow(xmin, ymin, 0., ymax - ymin, fc='k',  lw=lw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False)
    plt.tick_params(direction='in', width=0.4, length=2, pad=1)


def plot_1e(pids, σ_obs=DataExp1.params_4['σ_obs']):
    _, ax = plt.subplots(figsize=(1.5, 1.05))
    Δ, p_human, p_model = [], [], []
    for pid in pids:
        data = DataExp1(pid)
        df = data.apply_Kalman_filters(σ_obs)
        df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
        df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
        model = data.build_model(data.n_params_4, data.params_4)
        p = model.predict(model.fit().x)
        for s in Exp1.structures:
            ns = list(Exp1.structures)
            ns.remove(s)
            Δ.append(df[s] - logsumexp(df[np.array(ns)], axis=1))
            p_human.append((df['choice'] == s) * 1)
            p_model.append(p[s])
    Δ, p_human, p_model = np.concatenate(Δ), np.concatenate(p_human), np.concatenate(p_model)
    order = np.argsort(Δ)
    Δ, p_human, p_model = Δ[order], p_human[order], p_model[order]
    x_min, x_max = -60, 60

    slice = (x_min <= Δ) & (Δ < x_max)
    Δ, p_human, p_model = Δ[slice], p_human[slice], p_model[slice]
    # ax.scatter(Δ, np.random.normal(p_human, 0.02), marker='o', s=1, label='human')
    # ax.scatter(Δ, np.random.normal(p_model, 0.02), marker='o', s=1, label='model')

    n = len(Δ)
    x, y_human, y_model, err = [], [], [], []
    bin_width = 15
    for i in np.arange(x_min, x_max, bin_width):
        print(i)
        x.append(i + bin_width / 2)
        idx = (i <= Δ) & (Δ < i + bin_width)
        y_human.append(p_human[idx].mean())
        y_model.append(p_model[idx].mean())
        err.append(sem(p_human[idx]))
        # err.append(np.sqrt(y_human[-1] * (1 - y_human[-1]) / np.sum(idx)))
    y_human = np.array(y_human)
    ax.errorbar(x, y_human, err, label='Humans ± sem', color='k', fmt='.', capsize=2, ms=1.5, capthick=0.5)
    # x = np.arange(x_min, x_max, bin_size_model)
    ax.plot(x, y_model, '.-', color='gray', label='Model', ms=1, zorder=5)

    ax.legend(loc='upper left', handlelength=1, handletextpad=0.3)
    ax.set_ylim((0, 1.08))
    ax.set_xlim(x_min, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(['.0', '.2', '.4', '.6', '.8', '1.0'])
    ax.set_xticks([x_min + bin_width, 0, x_max - bin_width])
    plot_ax(ax)
    # plt.show()
    plt.savefig(f'../pub/cosyne_2020/1e2.svg', transparent=True)


def plot_1f():
    _, ax = plt.subplots(figsize=(0.9, 0.9))
    x = [-135.0776, -176.2761, -156.0936, -202.9394, -166.2647, -183.4829,
         -119.6768, -208.3617, -153.8785, -199.2785, -155.3106, -178.9703]
    y = [-167.3934, -200.3342, -188.2256, -212.4345, -208.5521, -213.9509,
         -160.0937, -222.5519, -193.9772, -217.2727, -175.9234, -198.3846]
    plt.scatter(x, y, color='k', s=2, linewidths=0)
    xy_min, xy_max = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], ':', color='gray')
    ax.set_xlim((xy_min, xy_max))
    ax.set_ylim((xy_min, xy_max))
    ax.axis('equal')
    # ax.axis('off')
    plot_ax(ax)
    plt.yticks([-200, -150], rotation=90, va='center')
    # plt.show()
    plt.savefig(f'../pub/cosyne_2020/1f.svg', transparent=True)


def plot_2a(pids):
    _, ax = plt.subplots(figsize=(1.9, 1.28))
    DataMetaExp2(pids).plot_stacked_bar(ax)
    # res_meta = fit_meta_human(pids)
    y_human, y1, y2, y3 = [], [], [], []
    for pid in pids:
        data1 = DataExp1(pid)
        data2 = DataExp2(pid)
        y_human.append(data2.plot_line_human())
        res1 = data1.build_model(data1.n_params_4, data1.params_4).fit()
        y1.append(data2.plot_line_model(data2.load_model(res1)))
        res2 = data2.build_model(data2.n_params_4, data2.params_4).fit()
        y2.append(data2.plot_line_model(data2.load_model(res2)))
        # res3 = data2.build_model(data2.n_params_4, data2.params_4, exp1=True).fit()
        # y3.append(data2.plot_line_model(data2.load_model(res3)))
        # y_model.append(data2.plot_line_model(data2.load_model(res_meta)))
    y_human, y1, y2, y3 = np.array(y_human).mean(axis=0), np.array(y1), np.array(y2), np.array(y3)
    ax.errorbar(DataExp2.x, y_human, np.sqrt(y_human * (1 - y_human) / 240), fmt='k.', ms=2, label='Humans ± sem', zorder=10,
                capsize=4, capthick=1, linewidth=1)
    ax.plot(DataExp2.x, y2.mean(axis=0), '.-', color='gray', ms=4, label='Models (Exp 2)', zorder=7, linewidth=1)
    ax.plot(DataExp2.x, y1.mean(axis=0), '.-', color='blue', ms=4, label='Models (Exp 1)', zorder=6, linewidth=1)
    # ax.plot(DataExp2.x, y3.mean(axis=0), '.-', color='brown', ms=4, label='4-param Exp 1 (C, H)', zorder=5, linewidth=1)
    ax.legend(loc='upper right', handlelength=1, handletextpad=0.3)
    ax.set_ylim((0, 1.08))
    ax.set_xlabel('')
    ax.set_ylabel('p(choice = C)')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(['.0', '.2', '.4', '.6', '.8', '1.0'])
    plot_ax(ax)
    # plt.show()
    plt.savefig(f'../pub/cosyne_2020/2a.svg', transparent=True)


def plot_2b(pids, σ_obs=DataExp1.params_4['σ_obs']):
    _, ax = plt.subplots(figsize=(1.9, 1.1))
    data_meta = DataMetaExp2(pids)
    res_meta = fit_meta_human(pids)
    model = data_meta.load_model(res_meta)
    df = model.df
    df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
    df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
    Δ = df['CLU'] - df['SDH']
    p_human = (df['choice'] == 'CLU') * 1.
    # p_model = model.predict(model.fit().x)['CLU']
    p1, p2, p3 = [], [], []
    for i in range(len(pids)):
        data1 = DataExp1(pids[i])
        data2 = DataExp2(pids[i])
        res1 = data1.build_model(data1.n_params_4, data1.params_4).fit()
        model = data_meta.datas[i].load_model(res1)
        p1.append(model.predict(model.fit())['CLU'])
        res2 = data2.build_model(data2.n_params_4, data2.params_4).fit()
        model = data_meta.datas[i].load_model(res2)
        p2.append(model.predict(model.fit())['CLU'])
        res3 = data2.build_model(data2.n_params_4, data2.params_4, exp1=True).fit()
        # model = data_meta.datas[i].load_model(res3)
        # p3.append(model.predict(model.fit())['CLU'])
    order = np.argsort(Δ)
    Δ, p_human = Δ.to_numpy()[order], p_human.to_numpy()[order]
    # p_model = p_model.to_numpy()[order]
    p1 = np.concatenate(p1)[order]
    p2 = np.concatenate(p2)[order]
    # p3 = np.concatenate(p3)[order]
    x_min, x_max = -7, 7

    # slice = (x_min <= Δ) & (Δ < x_max)
    # Δ, p_human, p_model = Δ[slice], p_human[slice], p_model[slice]
    # ax.scatter(Δ, np.random.normal(p_human, 0.02), marker='o', s=1, label='human')
    # ax.scatter(Δ, np.random.normal(p_model, 0.02), marker='o', s=1, label='model')

    n = len(Δ)
    x, y_human, y1, y2, y3, err = [], [], [], [], [], []

    # n_bin = 8
    # bin_size = n // n_bin
    # for i in np.arange(0, n_bin * bin_size, bin_size):
    #     x_bin = Δ[i: i + bin_size]
    #     x.append((x_bin.min() + x_bin.max()) / 2)
    #     y_human.append(p_human[i: i + bin_size].mean())
    #     y_model.append(p_model[i: i + bin_size].mean())
    #     err.append(np.sqrt(y_human[-1] * (1 - y_human[-1]) / bin_size))

    bin_width = 2
    for i in np.arange(x_min, x_max, bin_width):
        print(i)
        x.append(i + bin_width / 2)
        idx = (i <= Δ) & (Δ < i + bin_width)
        y_human.append(p_human[idx].mean())
        y1.append(p1[idx].mean())
        y2.append(p2[idx].mean())
        # y3.append(p3[idx].mean())
        err.append(sem(p_human[idx]))
        # err.append(np.sqrt(y_human[-1] * (1 - y_human[-1]) / np.sum(idx)))

    y_human = np.array(y_human)
    ax.errorbar(x, y_human, err, label='Humans ± sem', color='k', fmt='.', capsize=2, ms=1.5, capthick=0.5, zorder=10)
    ax.plot(x, y2, '.-', color='gray', label='Models (Exp 2)', ms=2, zorder=6)
    ax.plot(x, y1, '.-', color='blue', label='Models (Exp 1)', ms=2, zorder=5)
    # ax.plot(x, y3, '.-', color='brown', label='4-param Exp 1 (C, H)', ms=2, zorder=4)

    ax.legend(loc='lower right', handlelength=1, handletextpad=0.3)
    ax.set_ylim((0, 1.08))
    ax.set_xlim(x_min, x_max + 0.5)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(['.0', '.2', '.4', '.6', '.8', '1.0'])
    plot_ax(ax)
    # plt.show()
    plt.savefig(f'../pub/cosyne_2020/2b.svg', transparent=True)


if __name__ == '__main__':

    pids = ['0999', '3085', '3216', '4066', '5681', '7288', '7765', '8393', '9126', '9196', '9403', '9863']
    mpl.rc('font', **{'family': 'sans-serif', 'size': 9, 'weight': 'normal',
                      'sans-serif': ['Arial', 'LiberationSans-Regular', 'FreeSans']})
    mpl.rc('lines', **{'linewidth': 1})
    plot_1e(pids, σ_obs=1.1)
    # plot_1f()
    # pids.remove('4066')
    # plot_2a(pids)
    # plot_2b(pids)
    # pid = '3216'
    # plot_confusion_matrix_human(pid)
    # plot_confusion_matrix_model(pid, reps=20)
