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


def plot_exp1_confusion_matrix(dat_file, mode='marginal', normalize=True, png_file=True, select=None, ax=None):
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
        png_file = dat_file[:-4] + ('.png' if select is None else f'_half{select + 1}.png')
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


def plot_exp2_confusion_matrix(dat_file, normalize=True, png_file=True, ax=None):
    ax = ax or plt.gca()
    from analysis.confusion_matrix import plot_confusion_matrix
    df = apply_Kalman_filters_on_exp2(dat_file)
    df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
    df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
    df.loc[df['CLU'] > df['SDH'], 'ground_truth'] = 'CLU'
    df.loc[df['CLU'] < df['SDH'], 'ground_truth'] = 'SDH'
    if png_file is True:
        png_file = dat_file[:-4] + '.png'
    plot_confusion_matrix(df['ground_truth'], df['choice'], normalize,
                          file_path=png_file, ax=ax, categories=('CLU', 'SDH'))


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


def get_model_exp1_5param(df, normalize=False):
    col_structures = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['IND', 'GLO', 'CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    return StructureDecisionModel(
        df=df, col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=lambda 𝜃: (𝜃[0], 𝜃[1], np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)),
        bounds=[(1e-3, 1.), (0., None), (None, None), (None, None), (None, None)],
        normalize=normalize
    )


def get_model_exp1_4param(df, α=1e-3, normalize=False):
    col_structures = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['IND', 'GLO', 'CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    return StructureDecisionModel(
        df=df, col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=lambda 𝜃: (α, 𝜃[0], np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)),
        bounds=[(0., None), (None, None), (None, None), (None, None)],
        normalize=normalize
    )


def get_model_exp2_3param(df, normalize=False):
    col_structures = ['CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    return StructureDecisionModel(
        df=df, col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=lambda 𝜃: (𝜃[0], 𝜃[1], np.array([0] * 3 + [𝜃[-1]] * 3)),
        bounds=[(1e-3, 1.), (0., None), (None, None)],
        normalize=normalize
    )


def get_model_exp2_2param(df, α=1e-3, normalize=False):
    col_structures = ['CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
    structures = ['CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    return StructureDecisionModel(
        df=df, col_target='choice', col_structures=col_structures, structures=structures,
        param_extractor=lambda 𝜃: (α, 𝜃[0], np.array([0] * 3 + [𝜃[-1]] * 3)),
        bounds=[(0., None), (None, None)],
        normalize=normalize
    )


def fit_model_to_exp1(dat_file, 𝜃_0=None, α=True,
                      glo=3/4, λ_I=1/4, σ_R=0, σ_x=0, repeats=1, verbose=1, normalize=False, ax=None, png_file=False):
    df = apply_Kalman_filters_on_exp1(dat_file, glo, λ_I, σ_R, σ_x, repeats)  # .loc[match_seed(dat_file)[0]]
    if α is True:
        if 𝜃_0 is None:
            𝜃_0 = np.array([0.1, 0.1, 0, 0, 0])
        model = get_model_exp1_5param(df, normalize)
        res = model.fit(np.array(𝜃_0), method='SLSQP', loss='choice', verbose=verbose)
    else:
        if 𝜃_0 is None:
            𝜃_0 = np.array([0.1, 0, 0, 0])
        model = get_model_exp1_4param(df, α, normalize)
        res = model.fit(np.array(𝜃_0), method='SLSQP', loss='choice', verbose=verbose)
    𝜃 = res.x
    if png_file is True:
        png_file = dat_file[:-4] + f'_σ={σ_R:.2f}_α={𝜃[0] if α is True else α:.2f}.png'
        fig, ax = plt.subplots()
    if ax:
        if α is True:
            ax.set_title(f'σ={σ_R:.1f}   α={𝜃[0]:.6f}   β={𝜃[1]:.6f}\nb_GLO={𝜃[2]:.6f}   b_CLU={𝜃[3]:.6f}   b_SDH={𝜃[4]:.6f}')
        else:
            ax.set_title(f'σ={σ_R:.1f}   α={α:.6f}   β={𝜃[0]:.6f}\nb_GLO={𝜃[1]:.6f}   b_CLU={𝜃[2]:.6f}   b_SDH={𝜃[3]:.6f}')
        model.predict_confusion_matrix(res.x, normalize=True, ax=ax)
    if png_file is not False:
        print(png_file)
        plt.savefig(png_file)
        plt.close()
    return res


def fit_model_to_exp2(dat_file, 𝜃_0=np.array([0.1, 0.1, 0]),
                      glo=3/4, λ_I=1/4, σ_R=0, σ_x=0, repeats=1, verbose=1, normalize=False):
    df = apply_Kalman_filters_on_exp2(dat_file, glo, λ_I, σ_R, σ_x, repeats)
    model = get_model_exp2_3param(df, normalize)
    res = model.fit(np.array(𝜃_0), method='SLSQP', loss='choice', verbose=verbose)
    # model.predict_confusion_matrix(res.x, plot=plot, normalize=True)
    return res


def fit_σ_R(dat_file, Σ, 𝜃_0):
    m = (0, np.Inf)
    for σ in Σ:
        res = fit_model_to_exp1(dat_file, α=0.1, σ_R=σ, 𝜃_0=𝜃_0, verbose=0)
        print(f'σ={σ:.2f} loss={res.fun} x={res.x}')
        if res.fun < m[1]:
            m = (σ, res.fun)
    return m[0]


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


def plot_exp2_stacked_bar_agg(dat_files, ax=None):
    ax = ax or plt.gca()
    ys = [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)]
    from config import ExperimentConfig
    x = ExperimentConfig.glo_exp2
    responses = [('CLU', 'high', 'darkgoldenrod'), ('CLU', 'low', 'goldenrod'),
                 ('SDH', 'low', 'green'), ('SDH', 'high', 'darkgreen')]
    for dat_file in dat_files:
        df = pd.DataFrame({
            'ground_truth': load_data(dat_file, 'ground_truth')[:100],
            'choice': load_data(dat_file, 'choice')[:100],
            'confidence': load_data(dat_file, 'confidence')[:100],
        })
        df['accuracy'] = (df['choice'] == 'CLU') * 1.0
        print(len(df))

        count_y = lambda choice, confidence: \
            [len(df[(df['choice'] == choice) &
                    (df['confidence'] == confidence) &
                    (df['ground_truth'] == f'{s:.2f}')]) / 20
             for s in x]
        for i, resp in enumerate(responses):
            # print(df[df['choice'] == resp[0]])
            ys[i] += count_y(resp[0], resp[1])
    print(ys)
    p = []

    bottom = np.zeros(len(x))
    for i in range(4):
        p.append(ax.bar(x, ys[i] / len(dat_files), width=0.14, bottom=bottom, color=responses[i][2]))
        bottom += ys[i] / len(dat_files)
    ax.add_artist(plt.legend(p, ('CLU high', 'CLU low', 'SDH low', 'SDH high'), loc='lower left'))
    ax.set_ylabel('% CLU choices')
    ax.set_xlabel('λ_GLO^2/(λ_CLU^2+λ_GLO^2)')
    ax.set_xticks(x)


def plot_exp2_prediction(dat_file, 𝜃, σ_R=0., ax=None, png_file=True):
    from config import ExperimentConfig
    glo = ExperimentConfig.glo_exp2
    ax = ax or plt.gca()

    plot_exp2_stacked_bar(dat_file, ax=ax)
    df = apply_Kalman_filters_on_exp2(dat_file, σ_R=0)
    df['ground_truth'] = df['ground_truth'].astype(float)
    df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
    df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
    df.loc[df['CLU'] > df['SDH'], 'argmax'] = 'CLU'
    df.loc[df['CLU'] < df['SDH'], 'argmax'] = 'SDH'
    accuracy = (df['choice'] == 'CLU') * 1.0
    y_human = [accuracy[df['ground_truth'] == g].mean() for g in glo]
    err = [np.sqrt(p * (1 - p) / 20) for p in y_human]
    ax.errorbar(glo, y_human, err, label='Human Choice', color='k', linewidth=3, marker='o', zorder=3)

    accuracy = (df['argmax'] == 'CLU') * 1.0
    y_argmax = [accuracy[df['ground_truth'] == g].mean() for g in glo]
    ax.plot(glo, y_argmax, label='Likelihood Argmax', color=mcd.CSS4_COLORS['navy'], linewidth=3, marker='o', zorder=4)

    df = apply_Kalman_filters_on_exp2(dat_file, σ_R=σ_R)
    structures = ['CLU', 'CLU', 'CLU', 'SDH', 'SDH', 'SDH']
    model = get_model_exp2_3param(df)
    p = model.predict(𝜃)
    df['p_CLU'] = np.sum(p[:, [i for i in range(len(structures)) if structures[i] == 'CLU']], axis=1)
    df['p_SDH'] = np.sum(p[:, [i for i in range(len(structures)) if structures[i] == 'SDH']], axis=1)
    y_fit = [df['p_CLU'][df['ground_truth'].astype(float) == g].mean() for g in glo]
    ax.plot(glo, y_fit, label=f'Model Fit to Exp1', color=mcd.CSS4_COLORS['maroon'], linewidth=3, marker='o', zorder=4)

    ax.legend(loc='upper right')
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('λ_GLO^2/(λ_CLU^2+λ_GLO^2)')
    ax.set_ylabel('% CLU choices')

    if png_file is True:
        png_file = dat_file[:-4] + f'_σ={σ_R:.2f}_α={𝜃[0]:.2f}.png'
    if png_file is not False:
        plt.savefig(png_file)
        # plt.close()
    return y_human, y_argmax, y_fit


def plot_accuracy_confidence(dat_files, ax=None):
    ax = ax or plt.gca()
    n = len(dat_files)
    for dat_file, jitter, zorder in zip(dat_files, np.linspace(-0.01 * n, 0.01 * n, n), np.arange(1, n + 1)):
        name = dat_file.split('/')[-1].split('_')[0]
        df = pd.DataFrame({
            'confidence': load_data(dat_file, 'confidence'),
            'accuracy': (np.array(load_data(dat_file, 'ground_truth')) == np.array(load_data(dat_file, 'choice'))) * 100
        })
        accuracy = df.groupby('confidence').agg(['mean', 'sem']).reindex(['low', 'high'])['accuracy']
        plt.errorbar(np.array([0, 1]) + jitter, accuracy['mean'], accuracy['sem'],
                     marker='o', linewidth=2, elinewidth=2, label=name, zorder=zorder, capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])
    ax.set_xlabel('confidence')
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((0, 100))
    ax.set_ylabel('% correct choices')
    # plt.legend(loc='lower right')


def plot_rt_confidence(dat_files, ax=None):
    ax = ax or plt.gca()
    n = len(dat_files)
    colors = []
    for dat_file, jitter, zorder in zip(dat_files, np.linspace(-0.02 * n, 0.02 * n, n), np.arange(1, n + 1)):
        name = dat_file.split('/')[-1].split('_')[0]
        df = pd.DataFrame({
            'confidence': load_data(dat_file, 'confidence'),
            'rt': load_data(dat_file, 'rt'),
        })
        rt = df.groupby('confidence').median().reindex(['low', 'high'])['rt']
        p = plt.plot(np.array([0, 1]) + jitter, rt, marker='o', linewidth=1, label=name, zorder=zorder)
        bp = plt.boxplot([df[df['confidence'] == c]['rt'] for c in ['low', 'high']],
                         positions=np.array([0, 1]) + jitter, widths=0.02, sym='.', zorder=10)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=p[0].get_color(), linewidth=1)
        plt.setp(bp['fliers'], markeredgecolor=p[0].get_color())
        colors.append(p[0].get_color())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])
    ax.set_xlabel('confidence')
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((0, 8))
    ax.set_ylabel('decision time/s')
    # plt.legend(loc='upper right')
    return colors


def plot_rt_accuracy(dat_files, colors, ax=None):
    from matplotlib.patches import Patch
    ax = ax or plt.gca()
    markers = [('IND', 'd'), ('GLO', 'o'), ('CLU', '>'), ('SDH', 'X')]
    patch_list = [plt.scatter([], [], marker=marker, color='k', s=16, label=s) for s, marker in markers]
    x, y = [], []
    n = len(dat_files)
    for dat_file, jitter, zorder in zip(dat_files, np.linspace(-0.01 * n, 0.01 * n, n), np.arange(1, n + 1)):
        name = dat_file.split('/')[-1].split('_')[0]
        df = pd.DataFrame({
            'rt': load_data(dat_file, 'rt'),
            'ground_truth': load_data(dat_file, 'ground_truth'),
            'accuracy': (np.array(load_data(dat_file, 'ground_truth')) == np.array(load_data(dat_file, 'choice'))) * 100
        })
        # color = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        color = colors[zorder - 1]
        patch_list.append(Patch(color=color, label=name))
        for s, marker in markers:
            sub = df[df['ground_truth'] == s]
            x.append(sub['accuracy'].mean())
            y.append(sub['rt'].median())
            p = plt.scatter(x[-1], y[-1], color=color, marker=marker, s=16)
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]
    print(np.corrcoef(x, y))
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
    return f'/home/sichao/Documents/Motion/code/sichao/data/{id}/{id}_exp{exp}.dat'


def plot_exp2_prediction_agg(ids, σ_R=1., ax=None):
    Y_human, Y_argmax, Y_fit = [], [], []
    for id in ids:
        α = 0.1
        res = fit_model_to_exp1(fullfile(id, 1), α=α, σ_R=σ_R, ax=None, png_file=False, verbose=1)
        y_human, y_argmax, y_fit = plot_exp2_prediction(fullfile(id, 2), np.array([0.1, res.x[-4], res.x[-1] - res.x[-2]]), σ_R=σ_R, png_file=False)
        Y_human.append(y_human)
        Y_argmax.append(y_argmax)
        Y_fit.append(y_fit)
    plt.close()
    ax = ax or plt.gca()
    Y_human = np.array(Y_human)
    Y_argmax = np.array(Y_argmax)
    Y_fit = np.array(Y_fit)
    from config import ExperimentConfig
    glo = ExperimentConfig.glo_exp2
    plot_exp2_stacked_bar_agg([fullfile(id, 2) for id in ids], ax=ax)
    from scipy.stats import sem
    ax.errorbar(glo, Y_human.mean(axis=0), sem(Y_human, axis=0), label='Human Choice', color='k', linewidth=3, marker='o', zorder=3)
    ax.plot(glo, Y_argmax.mean(axis=0), label='Likelihood Argmax', color=mcd.CSS4_COLORS['navy'], linewidth=3, marker='o', zorder=4)
    ax.plot(glo, Y_fit.mean(axis=0), label=f'Model Fit to Exp1', color=mcd.CSS4_COLORS['maroon'], linewidth=3, marker='o', zorder=4)
    ax.legend(loc='upper right')
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('λ_GLO^2/(λ_CLU^2+λ_GLO^2)')
    ax.set_ylabel('% CLU choices')
    plt.show()


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


from scipy.special import logsumexp, expit
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit


def logistic(x, β, b):
    return expit(β * (x + b))


def plot_psychometric_curve(dat_files):
    df_list = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for dat_file in dat_files:
        df = apply_Kalman_filters_on_exp2(dat_file)
        df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
        df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
        df_list.append(df)
    df = pd.concat(df_list)
    x = df['dL'] = df['CLU'] - df['SDH']
    y = df['choice'] = (df['choice'] == 'CLU') * 1.
    order = np.argsort(x)
    x = x.to_numpy()[order]
    y = y.to_numpy()[order]
    groups = df.groupby('confidence')
    colors = {'high': mcd.CSS4_COLORS['green'], 'low': mcd.CSS4_COLORS['orange'],
              'all': mcd.CSS4_COLORS['deepskyblue']}
    jitter = 0.02
    for c, group in groups:
        ax.scatter(group['dL'], np.random.normal(group['choice'] + jitter, 0.02), marker='o', s=1, label=c, color=colors[c])
        jitter = -jitter
        clf = LogisticRegression()
        xi, yi = group['dL'], group['choice']
        order = np.argsort(xi)
        xi = xi.to_numpy()[order]
        yi = yi.to_numpy()[order]
        popt, pcov = curve_fit(logistic, xi, yi, p0=[0.1, 1])
        print(popt)
        ax.plot(x, logistic(x, *popt), label=c, color=colors[c], linewidth=1)
        ax.axvline(-popt[1], ls='--', linewidth=1, zorder=-1, color=colors[c])
    popt, pcov = curve_fit(logistic, x, y)
    print(popt)
    print()
    c = 'all'
    ax.plot(x, logistic(x, *popt), linewidth=1, zorder=-1, label=c, color=colors[c])
    ax.axvline(-popt[1], ls='--', linewidth=1, zorder=-1, color=colors[c])
    ax.legend(loc='right')
    # ax.set_xlim((-200, 500))
    ax.set_xlabel('lnL(CLU)-lnL(SDH)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['SDH', 'CLU'])
    ax.set_ylabel('choice')
    plt.tight_layout()
    plt.show()


def plot_psychometric_curve2(dat_files):
    df_list = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for dat_file in dat_files:
        df = apply_Kalman_filters_on_exp2(dat_file)
        df['CLU'] = logsumexp(df[['CLU_012', 'CLU_120', 'CLU_201']], axis=1)
        df['SDH'] = logsumexp(df[['SDH_012', 'SDH_120', 'SDH_201']], axis=1)
        df_list.append(df)
    df = pd.concat(df_list)
    x = np.abs(df['CLU'] - df['SDH']) ** 1
    y = df['confidence'] = (df['confidence'] == 'high') * 1.
    order = np.argsort(x)
    x = x.to_numpy()[order]
    y = y.to_numpy()[order]
    popt, pcov = curve_fit(logistic, x, y)
    import statsmodels.api as sm
    res = sm.Logit(y, x).fit()
    print(res.summary())
    print(popt)
    print()
    ax.scatter(x, np.random.normal(y, 0.02), marker='o', s=0.5, color='y')
    ax.plot(x, logistic(x, *popt), linewidth=2, zorder=-1, color='g')
    # ax.set_xlim((0, 250000))
    ax.set_xlabel('|lnL(CLU)-lnL(SDH)|')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['low', 'high'])
    ax.set_ylabel('confidence')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # config_display()
    σ_R_fit = {'sichao': 0.5, 'johannes': 2.0, '0999': 0.0, '3085': 0.6, '3216': 1.2, '4066': 0.5,
               '5681': 0.4, '7288': 2.2, '8393': 0.3, '9126': 1.3, '9196': 0.0, '9403': 1.7, '9863': 0.1}
    # plot_exp2_prediction_agg([id for id in σ_R_fit])
    plot_psychometric_curve2([fullfile(id, 2) for id in σ_R_fit])
    # colors = plot_rt_confidence([fullfile(id) for id in σ_R_fit])
    # fig, ax = plt.subplots(figsize=(5, 6))
    # plot_rt_accuracy([fullfile(id) for id in σ_R_fit], colors, ax=ax)
    # plot_exp2_stacked_bar_agg([fullfile(id, 2) for id in σ_R_fit])
    # id = 'sichao'
    # print(plot_exp2_prediction(fullfile(id, 2), np.array([0.1, 0.1, 0, 0, 0]), σ_R=σ_R_fit[id]))
    # plt.tight_layout()
    # plt.show()



    # σ_R_fit = {'sichao': 0.5, 'johannes': 2.0, '3085': 0.6, '3216': 1.2, '4066': 0.5,
    #            '5681': 0.4, '7288': 2.2, '7765': 0.6, '8393': 0.3, '9126': 1.3, '9196': 0.0, '9403': 1.7, '9863': 0.1}
    # id = '0999'
    # # # print(id, fit_σ_R(fullfile(id, 1), np.arange(0, 2.1, 0.1), np.array([0.1, 0, 0, 0])))
    # # plot_exp2_stacked_bar(fullfile(id, 2))
    # # plt.show()
    # # plot_exp1_confusion_matrix(fullfile(id))
    # α = 0.1
    # σ_R = 1
    # res = fit_model_to_exp1(fullfile(id, 1), α=α, σ_R=σ_R, ax=None, png_file=True, verbose=1)
    # plt.show()
    # print(plot_exp2_prediction(fullfile(id, 2), np.array([0.1, res.x[-4], res.x[-1] - res.x[-2]]), σ_R=σ_R, ax=plt.gca(), png_file=False))  # res.x[0]
    # plt.show()

    # fit_σ_R(fullfile('9863'), np.arange(0, 1.1, 0.1), np.array([0.1, 0, 0, 0]))

    # print(np.log(200) * 5 + 2 * res.fun)
    # for α in np.arange(0.01, 0.2, 0.01):
    #     res = fit_model_to_exp1(fullfile(id, 1), α=α, σ_R=2, ax=None, png_file=False, verbose=0)
    #     print(α, np.log(200) * 4 + 2 * res.fun)

    # plot_exp1_confusion_matrix(fullfile(id, 1))
    # plt.show()

    # σ_R = [0.5, 2.0, 0.6, 1.2, 0.5, 0.4, 2.0, 1.3, 0.0, 1.7]
    # for id in ['sichao', 'johannes', 3085, 3216, 4066, 5681, 7288, 9126, 9196, 9403, ]:
    #     print(id)
    #     print(id, fit_σ_R(fullfile(id, 1), np.arange(0, 2.1, 0.1), np.array([0.1, 0, 0, 0])))

    # df = apply_Kalman_filters_on_exp1(fullfile(id, 1), σ_R=2)
    # model = get_model_exp1_5param(df)
    # model.predict_confusion_matrix(np.array([0, 0.5, 2, 3, 3]), normalize=True, ax=plt.gca())
    # plt.show()
