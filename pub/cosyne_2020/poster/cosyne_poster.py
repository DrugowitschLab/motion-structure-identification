import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import numpy as np

from exp1 import Exp1
from analysis._data_exp1 import DataExp1, DataMetaExp1
from analysis._data_exp2 import DataExp2
from analysis.utils.confusion_matrix import plot_confusion_matrix


def plot_R1(pids):
    data = DataMetaExp1(pids)
    _, ax = plt.subplots(figsize=(3.336, 3.336))
    data.plot_confusion_matrix(ax)
    ax.axis('off')
    ax.set_title('')
    ax.axis([-0.5, 3.5, 3.5, -0.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'C:\\Users\\yangs\\OneDrive\\Documents\\COSYNE 20\\plots\\R1.svg', transparent=True)
    # plt.show()


def plot_R2A(pids):
    data = DataMetaExp1(pids)
    model = data.build_model(data.n_params_4, data.params_4)
    res = model.fit()
    _, ax = plt.subplots(figsize=(3.336, 3.336))
    model.plot_confusion_matrix(res, ax)
    ax.axis('off')
    ax.set_title('')
    ax.axis([-0.5, 3.5, 3.5, -0.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'C:\\Users\\yangs\\OneDrive\\Documents\\COSYNE 20\\plots\\R2A.svg', transparent=True)
    # plt.show()


def plot_R2A_CV(pids, train_size=0.5, reps=1):
    cms = []
    for pid in pids:
        data = DataExp1(pid)
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
    _, ax = plt.subplots(figsize=(3.336, 3.336))
    plot_confusion_matrix(labels=Exp1.structures, cm=cm, ax=ax)
    ax.axis('off')
    ax.set_title('')
    ax.axis([-0.5, 3.5, 3.5, -0.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'C:\\Users\\yangs\\OneDrive\\Documents\\COSYNE 20\\plots\\R2A_CV.svg', transparent=True)
    # plt.show()


def confidence(pids):
    from scipy.stats import pearsonr
    data = DataMetaExp1(pids)
    model = data.build_model({'σ_obs': 0, 'π': 0, 'β': 0, 'b': 0}, {'σ_obs': 1.1, 'π': 0, 'β': 1, 'b': [0, 0, 0]})
    # ll = model.ll + model.ll_multiplicity
    # ll = ll - logsumexp(ll, axis=1, keepdims=True)
    # print(ll)
    # df = pd.DataFrame({s: logsumexp(ll[:, np.array(model.columns) == s], axis=1) for s in model.structures})
    # print(df)
    df = model.predict(model.fit())
    # x = df.apply(lambda row: row[model.df['choice'][row.name]], axis=1)
    x = df.max(axis=1)
    # print(data.df)
    # x = df['CLU']
    print(x)
    y = model.df['confidence'] == 'high'
    pearsonr(x, y)
    print(pearsonr(x, y))


def score(pids):
    scores = np.zeros((len(pids), 2))
    for i in range(len(pids)):
        scores[i, 0] = DataExp1(pids[i]).score().sum() / 200
        scores[i, 1] = DataExp2(pids[i]).score().sum() / 100
    print(scores)
    print(scores[:, 0].mean())
    print(scores[:, 0].std())
    print(scores[:, 1].mean())
    print(scores[:, 1].std())
    from scipy.stats import ttest_rel
    print(ttest_rel(scores[:, 0], scores[:, 1]))


def learning(pids):
    score1, score2 = np.array([]), np.array([])
    for pid in pids:
        data = DataExp1(pid)
        idx = data.match()
        data.idx = idx[:, 0]
        score1 = np.append(score1, data.score())
        data.idx = idx[:, 1]
        score2 = np.append(score2, data.score())
    print((score2 - score1).mean())
    print((score2 - score1).std())
    from scipy.stats import ttest_rel
    print(ttest_rel(score1, score2))


if __name__ == '__main__':
    pids = ['0999', '3085', '3216', '4066', '5681', '7288', '7765', '8393', '9126', '9196', '9403', '9863']
    mpl.rc('font', **{'family': 'sans-serif', 'size': 28, 'weight': 'normal',
                      'sans-serif': ['Arial', 'LiberationSans-Regular', 'FreeSans']})
    mpl.rc('lines', **{'linewidth': 0.5})
    # plot_R2A_CV(pids, reps=10)
    # confidence(pids)
    # score(pids)
    learning(pids)
