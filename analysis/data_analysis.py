from utils.data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count(dat_file, key='ground_truth'):
    from collections import Counter
    return Counter(load_data(dat_file, key))


def test_speed(dat_file, structures, n=3):
    data = load_data(dat_file)
    v3, v5, ground_truth = [], [], []
    for trial in data:
        L = structures[trial['answer']].L
        for φ in trial['phi']:
            v5.append(np.abs(φ[n:]))
            v3.append(np.abs(L @ φ[n:]))
            ground_truth.append(trial['answer'])
    v3, v5, ground_truth = np.array(v3), np.array(v5), np.array(ground_truth)
    df = pd.DataFrame({
        'ground_truth': ground_truth,
        **{f'v_dot{i}': v3[:, i] for i in range(n)},
        'v_glo': v5[:, 0],
        'v_clu': v5[:, 1],
        **{f'v_ind{i}': v5[:, 2 + i] for i in range(n)},
    })
    return df.groupby('ground_truth').agg(['mean', 'std'])


def plot_exp1_confusion_matrix(dat_file, mode='marginal', normalize=True, png_file=''):
    from analysis.confusion_matrix import plot_confusion_matrix, plot_conditional_confusion_matrix
    df = pd.DataFrame({
        'ground_truth': load_data(dat_file, 'ground_truth'),
        'choice': load_data(dat_file, 'choice'),
        'confidence': load_data(dat_file, 'confidence'),
    })
    if png_file is True:
        png_file = dat_file[:-4] + '.png'
    if mode == 'marginal':
        plot_confusion_matrix(df['ground_truth'], df['choice'], normalize, file_path=png_file)
    elif mode == 'both':
        plot_conditional_confusion_matrix(df,
                                          ('ground_truth', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('choice', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('confidence', ['low', 'high']),
                                          normalize, file_path=png_file)
    else:
        plot_confusion_matrix(df[df['confidence'] == mode]['ground_truth'],
                              df[df['confidence'] == mode]['choice'],
                              normalize, file_path=png_file)


def plot_exp2_stacked_bar(dat_file, png_file=''):
    df = pd.DataFrame({
        'ground_truth': load_data(dat_file, 'ground_truth'),
        'choice': load_data(dat_file, 'choice'),
        'confidence': load_data(dat_file, 'confidence'),
    })
    df['accuracy'] = (df['choice'] == 'CLU') * 1.0

    from exp2 import Exp2
    structures = list(Exp2.presets.keys())
    x = structures

    count_y = lambda choice, confidence: \
        [len(df[(df['choice'] == choice) &
                (df['confidence'] == confidence) &
                (df['ground_truth'] == s)])
         for s in structures]
    p = []
    bottom = np.zeros(len(x))
    responses = [('CLU', 'high'), ('CLU', 'low'), ('SDH', 'low'), ('SDH', 'high')]
    for choice, confidence in responses:
        y = count_y(choice, confidence)
        p.append(plt.bar(x, y, bottom=bottom)[0])
        bottom += y
    plt.legend(p, ('CLU hi', 'CLU lo', 'SDH lo', 'SDH hi'), loc='lower left')
    plt.yticks(np.arange(0, bottom[0] + 1, bottom[0] // 10))
    plt.ylabel('# choices')
    plt.xlabel('glo')
    if png_file is True:
        png_file = dat_file[:-4] + '.png'
    if png_file != '':
        plt.savefig(png_file)
    plt.show()


def apply_Kalman_filters(dat_file, σ_R):
    from os.path import exists
    csv_file = dat_file[:-4] + f'_σ={σ_R}.csv'
    if exists()


def config_display():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


if __name__ == '__main__':
    plot_exp1_confusion_matrix('../data/exp1/pilot1_0806/pilot1_0806.dat')

    # from glob import glob
    # for dat in glob('../data/*.dat'):
    #     try:
    #         print(dat)
    #         print(count(dat, 'answer'))
    #     except:
    #         pass
