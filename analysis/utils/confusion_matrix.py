from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from itertools import product


def plot_confusion_matrix(cm: np.ndarray, xticks: List[str], yticks: List[str], ax: plt.Axes,
                          cmap: Colormap = plt.get_cmap('Blues'), normalize: bool = True, colorbar: bool = False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.amin(cm), np.amax(cm)
    threshold = 0.6 * vmax + 0.4 * vmin
    im = ax.imshow(cm, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]:.2f}'[1 - int(cm[i, j]):], ha='center', va='center',
                color='white' if cm[i, j] > threshold else 'black')
    if colorbar:
        plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_frame_on(False)
    ax.axis('equal')
    return accuracy


if __name__ == '__main__':
    cm = np.random.rand(4, 4)
    print(cm)
    _, ax = plt.subplots()
    labels = ['1', '2', '3', '4']
    plot_confusion_matrix(cm, labels, labels, ax, normalize=False, colorbar=True)
    plt.show()
