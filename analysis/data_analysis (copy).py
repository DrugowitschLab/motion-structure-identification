from utils.data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from analysis.model import StructureDecisionModel
from scipy.special import logsumexp
import matplotlib._color_data as mcd


def count(dat_file, key='ground_truth'):
    return Counter(load_data(dat_file, key))


def test_speed(dat_file, structures, n=3):
    data = load_data(dat_file)
    v3, v5, ground_truth = [], [], []
    for trial in data:
        L = structures[trial['ground_truth']].L
        for φ in trial['phi']:
            v5.append(np.abs(φ[n:]))
            v3.append(np.abs(L @ φ[n:]))
            ground_truth.append(trial['ground_truth'])
    v3, v5, ground_truth = np.array(v3), np.array(v5), np.array(ground_truth)
    df = pd.DataFrame({
        'ground_truth': ground_truth,
        **{f'v_dot{i}': v3[:, i] for i in range(n)},
        'v_glo': v5[:, 0],
        'v_clu': v5[:, 1],
        **{f'v_ind{i}': v5[:, 2 + i] for i in range(n)},
    })
    return df.groupby('ground_truth').agg(['mean', 'std'])


def plot_exp1_confusion_matrix(dat_file, mode='marginal', normalize=True, png_file='', select=None, ax=None):
    ax = ax or plt.gca()
    from analysis.confusion_matrix import plot_confusion_matrix, plot_conditional_confusion_matrix
    df = pd.DataFrame({
        'ground_truth': load_data(dat_file, 'ground_truth'),
        'choice': load_data(dat_file, 'choice'),
        'confidence': load_data(dat_file, 'confidence'),
    })
    if select is not None:
        df = df.loc[match_seed(dat_file)[select]]
    if png_file is True:
        png_file = dat_file[:-4] + '.png'
    if mode == 'marginal':
        plot_confusion_matrix(df['ground_truth'], df['choice'], normalize, file_path=png_file, ax=ax)
    elif mode == 'both':
        plot_conditional_confusion_matrix(df,
                                          ('ground_truth', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('choice', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('confidence', ['low', 'high']),
                                          normalize, file_path=png_file, ax=ax)
    else:
        plot_confusion_matrix(df[df['confidence'] == mode]['ground_truth'],
                              df[df['confidence'] == mode]['choice'],
                              normalize, file_path=png_file, ax=ax)


def apply_Kalman_filters(dat_file, structures, σ_R=0., σ_x=0., repeats=1, csv_file=True, suffix=''):
    from analysis.kalman_filter import apply_Kalman_filter as f
    from os.path import exists
    if csv_file is True:
        csv_file = dat_file[:-4] + f'_σ={σ_R:.2f}{suffix}.csv'
    if csv_file is not False and exists(csv_file):
        return pd.read_csv(csv_file, dtype={**{s: float for s in structures}})
    keys = ['ground_truth', 'choice', 'confidence']
    skip = 3
    df = pd.DataFrame(
        data=([f(np.random.normal(trial['φ'][skip:], σ_x), trial['t'][skip:], structures[s], σ_R) for s in structures] +
              [trial[k] for k in keys] for _ in range(repeats) for trial in load_data(dat_file)),
        columns=[s for s in structures] + keys,
    )
    if csv_file is not False:
        df.to_csv(csv_file, index=False)
    return df


def apply_Kalman_filters_on_exp1(dat_file, glo=3/4, λ_I=1/4, σ_R=0., σ_x=0, repeats=1):
    from stimuli.motion_structure import MotionStructure
    return apply_Kalman_filters(dat_file, {
        'IND': MotionStructure(1, 2),
        'GLO': MotionStructure(1, λ_I),
        'CLU_012': MotionStructure(0, λ_I, permutation=[0, 1, 2]),
        'CLU_120': MotionStructure(0, λ_I, permutation=[1, 2, 0]),
        'CLU_201': MotionStructure(0, λ_I, permutation=[2, 0, 1]),
        'SDH_012': MotionStructure(glo, λ_I, permutation=[0, 1, 2]),
        'SDH_120': MotionStructure(glo, λ_I, permutation=[1, 2, 0]),
        'SDH_201': MotionStructure(glo, λ_I, permutation=[2, 0, 1]),
    }, σ_R, σ_x, repeats, suffix=f'_glo={glo}_λ_I={λ_I}')


def apply_Kalman_filters_on_exp2(dat_file, glo=3/4, λ_I=1/4, σ_R=0., σ_x=0, repeats=1):
    from stimuli.motion_structure import MotionStructure
    return apply_Kalman_filters(dat_file, {
        'CLU_012': MotionStructure(0, λ_I, permutation=[0, 1, 2]),
        'CLU_120': MotionStructure(0, λ_I, permutation=[1, 2, 0]),
        'CLU_201': MotionStructure(0, λ_I, permutation=[2, 0, 1]),
        'SDH_012': MotionStructure(glo, λ_I, permutation=[0, 1, 2]),
        'SDH_120': MotionStructure(glo, λ_I, permutation=[1, 2, 0]),
        'SDH_201': MotionStructure(glo, λ_I, permutation=[2, 0, 1]),
    }, σ_R, σ_x, repeats, suffix=f'_glo={glo}_λ_I={λ_I}')


def param_extractor_exp1(𝜃):
    from scipy.stats import expon
    _α = 𝜃[0]
    α = np.linspace(0, 1, 50)
    pα = expon.pdf(α, scale=_α)
    pα = pα / pα.sum()
    A = [(α[i], pα[i]) for i in range(len(α))]
    # A = [(𝜃[0], 1)]
    β = 𝜃[1]
    b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    return A, β, b


def bounds_exp1():
    return [(1e-3, 1.), (0., None), (None, None), (None, None), (None, None)]


def fit_model_to_exp1(dat_file, 𝜃_0, glo=3/4, λ_I=1/4, σ_R=0, σ_x=0, repeats=1, verbose=1, normalize=False,
                      param_extractor=param_extractor_exp1, bounds=bounds_exp1, ax=None):
    col_structures = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['IND', 'GLO', 'CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    model = StructureDecisionModel(
        df=apply_Kalman_filters_on_exp1(dat_file, glo, λ_I, σ_R, σ_x, repeats).loc[match_seed(dat_file)[0]],
        col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=param_extractor, bounds=bounds(), normalize=normalize
    )
    res = model.fit(np.array(𝜃_0), method='SLSQP', loss='choice', verbose=verbose)
    𝜃 = res.x
    if ax:
        ax.set_title(f'σ={σ_R:.1f}  α={𝜃[0]:.6f}  β={𝜃[1]:.6f}\nb_GLO={𝜃[2]:.6f}  b_CLU={𝜃[3]:.6f}  b_SDH={𝜃[4]:.6f}')
        model.predict_confusion_matrix(res.x, normalize=True, ax=ax)
    return res


def param_extractor_exp2(𝜃):
    from scipy.stats import expon
    _α = 𝜃[0]
    α = np.linspace(0, 1, 50)
    pα = expon.pdf(α, scale=_α)
    pα = pα / pα.sum()
    A = [(α[i], pα[i]) for i in range(len(α))]
    # A = [(𝜃[0], 1)]
    β = 𝜃[1]
    b = np.array([0] * 3 + [𝜃[-1]] * 3)
    return A, β, b


def bounds_exp2():
    return [(1e-3, 1.), (0., None), (None, None)]


def fit_model_to_exp2(dat_file, 𝜃_0, glo=3/4, λ_I=1/4, σ_R=0, σ_x=0, repeats=1, verbose=1, normalize=False, plot=False,
                      param_extractor=param_extractor_exp2, bounds=bounds_exp2):
    col_structures = ['CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    model = StructureDecisionModel(
        df=apply_Kalman_filters_on_exp2(dat_file, glo, λ_I, σ_R, σ_x, repeats),
        col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=param_extractor, bounds=bounds(), normalize=normalize,
    )
    res = model.fit(np.array(𝜃_0), method='SLSQP', loss='choice', verbose=verbose)
    # model.predict_confusion_matrix(res.x, plot=plot, normalize=True)
    return res


def fit_σ_R(dat_file, Σ, 𝜃_0):
    for σ in Σ:
        res = fit_model_to_exp1(dat_file, σ_R=σ, 𝜃_0=𝜃_0, verbose=0)
        print(f'σ={σ:.2f} loss={res.fun} x={res.x}')
        print()


def plot_exp2_stacked_bar(dat_file, png_file='', ax=None):
    ax = ax or plt.gca()
    df = pd.DataFrame({
        'ground_truth': load_data(dat_file, 'ground_truth')[:100],
        'choice': load_data(dat_file, 'choice')[:100],
        'confidence': load_data(dat_file, 'confidence')[:100],
    })
    df['accuracy'] = (df['choice'] == 'CLU') * 1.0
    print(len(df))

    from config import ExperimentConfig
    x = ExperimentConfig.glo_exp2

    count_y = lambda choice, confidence: \
        [len(df[(df['choice'] == choice) &
                (df['confidence'] == confidence) &
                (df['ground_truth'] == f'{s:.2f}')]) / 20
         for s in x]
    p = []
    bottom = np.zeros(len(x))
    responses = [('CLU', 'high', 'darkgoldenrod'), ('CLU', 'low', 'goldenrod'),
                 ('SDH', 'low', 'green'), ('SDH', 'high', 'darkgreen')]
    for choice, confidence, color in responses:
        y = count_y(choice, confidence)
        p.append(ax.bar(x, y, width=0.14, bottom=bottom, color=mcd.CSS4_COLORS[color])[0])
        bottom += y
    ax.add_artist(plt.legend(p, ('CLU high', 'CLU low', 'SDH low', 'SDH high'), loc='lower left'))
    ax.set_ylabel('% CLU choices')
    ax.set_xlabel('λ_GLO^2/(λ_CLU^2+λ_GLO^2)')
    ax.set_xticks(x)
    if png_file is True:
        png_file = dat_file[:-4] + '.png'
    if png_file != '':
        plt.savefig(png_file)


def plot_exp2_prediction(dat_file, 𝜃, σ_R=0., ax=None):
    from config import ExperimentConfig
    glo = ExperimentConfig.glo_exp2
    ax = ax or plt.gca()

    plot_exp2_stacked_bar(dat_file, ax=ax)
    df = apply_Kalman_filters_on_exp2(dat_file, σ_R=0)
    df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
    df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
    df.loc[df['CLU'] > df['SDH'], 'argmax'] = 'CLU'
    df.loc[df['CLU'] < df['SDH'], 'argmax'] = 'SDH'
    accuracy = (df['choice'] == 'CLU') * 1.0
    y = [accuracy[df['ground_truth'] == g].mean() for g in glo]
    err = [np.sqrt(p * (1 - p) / 20) for p in y]
    ax.errorbar(glo, y, err, label='Human Choice', color='k', linewidth=3, marker='o', zorder=3)

    accuracy = (df['argmax'] == 'CLU') * 1.0
    ax.plot(glo, [accuracy[df['ground_truth'] == g].mean() for g in glo],
            label='Likelihood Argmax', color=mcd.CSS4_COLORS['navy'], linewidth=3, marker='o', zorder=4)

    df = apply_Kalman_filters_on_exp2(dat_file, σ_R=σ_R)
    col_structures = ['CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    model = StructureDecisionModel(
        df=df,
        col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=param_extractor_exp2, bounds=bounds_exp2(), normalize=False
    )
    p = model.predict(𝜃)
    df['p_CLU'] = np.sum(p[:, [i for i in range(len(structures)) if structures[i] == 'CLU']], axis=1)
    df['p_SDH'] = np.sum(p[:, [i for i in range(len(structures)) if structures[i] == 'SDH']], axis=1)
    ax.plot(glo, [df['p_CLU'][float(df['ground_truth']) == g].mean() for g in glo],
            label=f'Model Fit to Exp1', color=mcd.CSS4_COLORS['maroon'], linewidth=3, marker='o', zorder=4)

    ax.legend(loc='upper right')
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('λ_GLO^2/(λ_CLU^2+λ_GLO^2)')
    ax.set_ylabel('% CLU choices')
    return y


colors = ['b', 'g', 'y', 'r']


def plot_accuracy_confidence(dat_files, ax=None):
    ax = ax or plt.gca()
    for dat_file, color, jitter, zorder in zip(dat_files, colors, np.linspace(-0.05, 0.05, 4), np.arange(1, 5)):
        name = dat_file.split('/')[-1].split('_')[0]
        df = pd.DataFrame({
            'confidence': load_data(dat_file, 'confidence'),
            'accuracy': (np.array(load_data(dat_file, 'ground_truth')) == np.array(load_data(dat_file, 'choice'))) * 100
        })
        accuracy = df.groupby('confidence').agg(['mean', 'std']).reindex(['low', 'high'])['accuracy']
        plt.errorbar(np.array([0, 1]) + jitter, accuracy['mean'], accuracy['std'],
                     marker='o', color=color, linewidth=3, elinewidth=2, label=name, zorder=zorder, capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])
    ax.set_xlabel('confidence')
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((-20, 140))
    ax.set_ylabel('% correct choices')
    plt.legend(loc='lower right')


def plot_rt_confidence(dat_files, ax=None):
    ax = ax or plt.gca()
    for dat_file, color, jitter, zorder in zip(dat_files, colors, np.linspace(-0.15, 0.15, 4), np.arange(1, 5)):
        name = dat_file.split('/')[-1].split('_')[0]
        df = pd.DataFrame({
            'confidence': load_data(dat_file, 'confidence'),
            'rt': load_data(dat_file, 'rt'),
        })
        rt = df.groupby('confidence').median().reindex(['low', 'high'])['rt']
        plt.plot(np.array([0, 1]) + jitter, rt,
                 marker='o', color=color, linewidth=3, label=name, zorder=zorder)
        bp = plt.boxplot([df[df['confidence'] == c]['rt'] for c in ['low', 'high']],
                         positions=np.array([0, 1]) + jitter, widths=0.05, sym='.', zorder=10)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color, linewidth=2)
        plt.setp(bp['fliers'], markeredgecolor=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])
    ax.set_xlabel('confidence')
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((0, 7))
    ax.set_ylabel('decision time/s')
    plt.legend(loc='upper right')


def plot_rt_accuracy(dat_files, ax=None):
    from matplotlib.patches import Patch
    ax = ax or plt.gca()
    markers = [('IND', 'd'), ('GLO', 'o'), ('CLU', '>'), ('SDH', 'X')]
    patch_list = [plt.scatter([], [], marker=marker, color='k', s=128, label=s) for s, marker in markers]
    x, y = [], []
    for dat_file, color, jitter, zorder in zip(dat_files, colors, np.linspace(-0.05, 0.05, 4), np.arange(1, 5)):
        name = dat_file.split('/')[-1].split('_')[0]
        patch_list.append(Patch(color=color, label=name))
        df = pd.DataFrame({
            'rt': load_data(dat_file, 'rt'),
            'ground_truth': load_data(dat_file, 'ground_truth'),
            'accuracy': (np.array(load_data(dat_file, 'ground_truth')) == np.array(load_data(dat_file, 'choice'))) * 100
        })
        for s, marker in markers:
            sub = df[df['ground_truth'] == s]
            x.append(sub['accuracy'].mean())
            y.append(sub['rt'].median())
            plt.scatter(x[-1], y[-1], color=color, marker=marker, s=128)
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), '--', linewidth=2, zorder=-1)
    ax.set_xlabel('% correct choices')
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 3))
    ax.set_ylabel('median decision time/s')
    plt.legend(handles=patch_list, loc='lower left')


def score(ground_truth, choice, confidence):
    confidence = ((confidence) == 'high') * 2 + 1
    return (((choice == ground_truth) * 1. - 0.5) * confidence + 0.5).sum()


def score_exp1(dat_file):
    confidence = np.array(load_data(dat_file, 'confidence'))
    choice = np.array(load_data(dat_file, 'choice'))
    ground_truth = np.array(load_data(dat_file, 'ground_truth'))
    return score(ground_truth, choice, confidence)


def score_exp2(dat_file):
    df = apply_Kalman_filters_on_exp2(dat_file)
    df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
    df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
    df.loc[df['CLU'] > df['SDH'], 'ground_truth'] = 'CLU'
    df.loc[df['CLU'] < df['SDH'], 'ground_truth'] = 'SDH'
    return score(df['ground_truth'], df['choice'], df['confidence'])


def match_seed(dat_file):
    lookup = []
    idx = [[], []]
    seeds = load_data(dat_file, 'seed')
    for i, seed in enumerate(seeds):
        if seed in lookup:
            idx[1].append(i)
        else:
            idx[0].append(i)
            lookup.append(seed)
    return idx


def fullfile(id, exp=1):
    return f'/home/sichao/Documents/Motion/code/sichao/data/exp{exp}/{id}/{id}.dat'


def config_display():
    np.set_printoptions(linewidth=200)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def param_extractor2(𝜃):
    from scipy.stats import expon
    α = np.linspace(0, 1, 50)
    pα = expon.pdf(α, scale=0.05)
    pα = pα / pα.sum()
    A = [(α[i], pα[i]) for i in range(len(α))]
    β = 𝜃[1]
    b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    return A, β, b


class Data:
    def __init__(self, id):
        self.f1 = fullfile(id, 1)



if __name__ == '__main__':
    # config_display()
    # # plot_exp1_confusion_matrix('../data/exp1/pilot2_20190808155837.dat')
    # # plot_exp2_stacked_bar('../data/exp2/johannes_0805/johannes_0805.dat')
    # # plt.show()
    # # from time import time
    # # t_s = time()
    # # print(apply_Kalman_filters_on_exp1('../data/exp1/sichao_0806/sichao_0806.dat'))
    # # print(time() - t_s)
    # # loss = fit_model_to_exp1_lapse('../data/exp1/johannes_0729/johannes_0729.dat', σ_R=1,
    # #                                𝜃_0=np.array([0.1, 0, 0, 0]), verbose=0)
    # # print(loss)
    # name = 'johannes_0805'
    # σ_R = 0.5
    # 𝜃 = fit_model_to_exp1(f'../data/exp1/{name}/{name}.dat', np.array([[0.1, 0.1, 0, 0, 0]]), σ_R=σ_R).x
    # print(𝜃)
    # print(plot_exp2_prediction(f'../data/exp2/{name}/{name}.dat', np.array([𝜃[0], 𝜃[1], 𝜃[-1] - 𝜃[-2]]), σ_R=σ_R))

    # dat_files = [
    #     '../data/exp2/johannes_0805/johannes_0805.dat',
    #     '../data/exp2/sichao_0806/sichao_0806.dat',
    #     '../data/exp2/pilot1_0806/pilot1_0806.dat',
    #     '../data/exp2/pilot2_0808/pilot2_0808.dat',
    # ]
    # # plot_rt_confidence(dat_files)
    # # plt.show()
    # for dat_file in dat_files:
    #     print(score_exp2(dat_file))
    # print(score_exp1('/home/sichao/Documents/Motion/code/sichao/data/exp1/5681_20190815102227.dat'))
    # print(score_exp2('/home/sichao/Documents/Motion/code/sichao/data/exp2/5681_20190815110829.dat'))
    # plot_exp2_stacked_bar('/home/sichao/Documents/Motion/code/sichao/data/exp2/9403_20190815205417.dat')
    # # plot_exp1_confusion_matrix('/home/sichao/Documents/Motion/code/sichao/data/exp1/9403_20190815202702.dat')

    # plot_exp1_confusion_matrix('/home/sichao/Documents/Motion/code/sichao/data/exp1/7288_20190823091156.dat')
    # plot_exp2_stacked_bar('/home/sichao/Documents/Motion/code/sichao/data/exp2/7288_20190823094355.dat')
    # data = load_data('/home/sichao/Documents/Motion/code/sichao/data/exp2/sichao_20190822183912.dat')
    # print(len(data))
    # print(score_exp1('/home/sichao/Documents/Motion/code/sichao/data/exp1/9126_20190823130951.dat'))
    # print(score_exp2('/home/sichao/Documents/Motion/code/sichao/data/exp2/9126_20190823133402.dat'))
    # seed_last = 0
    # c = 0
    # for trial in data:
    #     print(trial['seed'], trial['ground_truth'], trial['choice'])
    #     if trial['seed'] != seed_last:
    #         c += 1
    #     seed_last = trial['seed']
    # print(c)

    # # fit_model_to_exp1('/home/sichao/Documents/Motion/code/sichao/data/exp1/3216_0815/3216_0815.dat',
    # #                   np.array([0.1, 0.1, 0, 0, 0]), σ_R=1, ax=plt.gca())
    # name = '7288'
    # # fit_σ_R(f'../data/exp1/{name}/{name}.dat', np.arange(0, 2, 0.1), np.array([[0.1, 0.1, 0, 0, 0]]))
    # σ_R = 0.5
    # 𝜃 = fit_model_to_exp1(f'../data/exp1/{name}/{name}.dat', np.array([[0.1, 1, 0, 10, 0]]), σ_R=σ_R, param_extractor=param_extractor2, bounds=lambda: None, ax=plt.gca()).x
    # print(𝜃)
    # print(plot_exp2_prediction(f'../data/exp2/{name}/{name}.dat', np.array([𝜃[0], 𝜃[1], 𝜃[-1] - 𝜃[-2]]), σ_R=σ_R, ax=plt.gca()))
    # plot_exp1_confusion_matrix(fullfile('7288', 1))
    # plot_exp1_confusion_matrix('/home/sichao/Documents/Motion/code/sichao/data/exp1/9126_20190823130951.dat')
    # fit_model_to_exp1('/home/sichao/Documents/Motion/code/sichao/data/exp1/4066/4066.dat', np.array([0.1, 0.01, 0, 0, 0]), σ_R=1,  ax=plt.gca(), param_extractor=param_extractor2, bounds=lambda :None)
    # plot_exp2_stacked_bar('/home/sichao/Documents/Motion/code/sichao/data/exp2/4066_20190820120357.dat')
    plt.show()
