import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product


def draw_matrix(cm, xticks, yticks, normalize=False, cmap=None, file_path='', ax=None):
    ax = ax or plt.gca()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = 0.7
        vmin, vmax = 0, 1
    else:
        thresh = cm.mean() * 1.5
        vmin, vmax = cm.min(), cm.max()
    im = ax.imshow(cm, interpolation='nearest', vmin=vmin, vmax=vmax,
                   cmap=plt.get_cmap('Blues') if cmap is None else cmap)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)  # , rotation=45)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]:.4f}' if normalize else f'{cm[i, j]:.4f}',
                 ha='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    if file_path == '' and ax is None:
        plt.show()
    else:
        plt.savefig(file_path)


def plot_confusion_matrix(y_true, y_pred, normalize=False, file_path='', ax=None,
                          categories=('IND', 'GLO', 'CLU', 'SDH')):
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    print(cm)
    accuracy_rate = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy_rate
    print(f'accuracy={accuracy_rate:0.4f}; misclass={misclass:0.4f}')

    ax.set_ylabel('Ground truth')
    ax.set_xlabel('Choice')
    draw_matrix(cm, categories, categories, normalize, file_path=file_path, ax=ax)


def plot_conditional_confusion_matrix(df, x, y1, y2, normalize=False, file_path="", ax=None):
    xticks = x[1]
    yticks = [f'{i}({j})' for i in y1[1] for j in y2[1]]
    cm = np.zeros((len(yticks), len(xticks)), dtype=int)
    for i, j in product(range(cm.shape[1]), range(cm.shape[0])):
        j1, j2 = j // len(y2[1]), j % len(y2[1])
        print(i, j1, j2)
        cm[j][i] = len(df[(df[x[0]] == x[1][i]) & (df[y1[0]] == y1[1][j1]) & (df[y2[0]] == y2[1][j2])])
    print(cm)

    ax.set_ylabel('Ground truth')
    ax.set_xlabel('Choice')
    draw_matrix(cm, xticks, yticks, normalize, file_path=file_path, ax=ax)
