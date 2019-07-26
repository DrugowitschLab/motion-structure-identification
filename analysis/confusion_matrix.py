from utils.data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from itertools import product
from sys import argv


labels = ['IND', 'GLO', 'CLU', 'SDH']


def draw_matrix(cm, xticks, yticks, normalize=False, cmap=None, title='Confusion Matrix', file_path=""):
    if len(argv) > 2:
        f"/home/sichao/Pictures/data/{argv[1]}_{argv[2]}"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.mean()

    plt.imshow(cm, interpolation='nearest', vmin=0, vmax=1, cmap=plt.get_cmap('Blues') if cmap is None else cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.yticks(np.arange(len(yticks)), yticks)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm[i, j]:0.4f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if file_path == "":
        plt.show()
    else:
        plt.savefig(file_path)


def plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    accuracy_rate = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy_rate
    print(f'accuracy={accuracy_rate:0.4f}; misclass={misclass:0.4f}')

    plt.figure(figsize=(8, 6))
    plt.ylabel('Ground truth')
    plt.xlabel('Choice')
    draw_matrix(cm, labels, labels, normalize)


def plot_conditional_confusion_matrix(df, x, y1, y2, normalize=False):
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
    draw_matrix(cm, xticks, yticks, normalize)


def plot_cm(file='sichao/pilot_sichao.dat', mode='marginal', nm=True):
    data = load_data(f'../data/{file}')
    answer, choice, confidence = [], [], []
    for record in data:
        answer.append(record['answer'])
        choice.append(record['choice'])
        confidence.append(record['confidence'])
    df = pd.DataFrame({'answer': answer, 'choice': choice, 'confidence': confidence})
    if mode == 'marginal':
        plot_confusion_matrix(answer, choice, nm)
    elif mode == 'both':
        plot_conditional_confusion_matrix(df,
                                          ('answer', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('choice', ['IND', 'GLO', 'CLU', 'SDH']),
                                          ('confidence', ['low', 'high']), nm)
    else:
        plot_confusion_matrix(df[df['confidence'] == argv[2]]['answer'], df[df['confidence'] == argv[2]]['choice'], nm)


if __name__ == '__main__':
    if len(argv) > 2:
        file = argv[1]
        mode = argv[2]
        nm = True if argv[3] == 't' else False
    else:
        file = 'sichao/pilot_sichao.dat'
        mode = 'marginal'
        nm = True
    plot_cm(file, mode, nm)
