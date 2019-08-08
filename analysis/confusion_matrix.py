from utils.data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product
from sys import argv


labels = ['IND', 'GLO', 'CLU', 'SDH']


def draw_matrix(cm, xticks, yticks, normalize=False, cmap=None, title='Confusion Matrix', file_path=''):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = 0.7
        vmin, vmax = 0, 1
    else:
        thresh = cm.mean() * 1.5
        vmin, vmax = cm.min(), cm.max()
    plt.imshow(cm, interpolation='nearest', vmin=vmin, vmax=vmax,
               cmap=plt.get_cmap('Blues') if cmap is None else cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(xticks)), xticks)  # , rotation=45)
    plt.yticks(np.arange(len(yticks)), yticks)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:0.4f}' if normalize else f'{cm[i, j]}',
                 ha='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    if file_path == '':
        plt.show()
    else:
        plt.savefig(file_path)


def plot_confusion_matrix(y_true, y_pred, normalize=False, file_path=""):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    accuracy_rate = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy_rate
    print(f'accuracy={accuracy_rate:0.4f}; misclass={misclass:0.4f}')

    plt.figure(figsize=(8, 6))
    plt.ylabel('Ground truth')
    plt.xlabel('Choice')
    draw_matrix(cm, labels, labels, normalize, file_path=file_path)


def plot_conditional_confusion_matrix(df, x, y1, y2, normalize=False, file_path=""):
    xticks = x[1]
    yticks = [f'{i}({j})' for i in y1[1] for j in y2[1]]
    cm = np.zeros((len(yticks), len(xticks)), dtype=int)
    for i, j in product(range(cm.shape[1]), range(cm.shape[0])):
        j1, j2 = j // len(y2[1]), j % len(y2[1])
        print(i, j1, j2)
        cm[j][i] = len(df[(df[x[0]] == x[1][i]) & (df[y1[0]] == y1[1][j1]) & (df[y2[0]] == y2[1][j2])])
    print(cm)

    plt.figure(figsize=(6, 8))
    plt.ylabel('Choice')
    plt.xlabel('Ground truth')
    draw_matrix(cm, xticks, yticks, normalize, file_path=file_path)
