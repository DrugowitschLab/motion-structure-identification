from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import pylab as pl
import numpy as np


class DecisionRegions:
    def __init__(self, ax, X, y):
        self.ax = ax
        self.X = X
        self.y = y
        # Initializing Classifiers
        clfs = {
            'LR': LogisticRegression(solver='lbfgs'),
            'RF': RandomForestClassifier(n_estimators=100),
            'NB': GaussianNB(),
            'SVM': SVC(gamma='auto')
        }
        self.clf = clfs['SVM']

    def fit(self, xi, yi):
        self.X.append(xi)
        self.y.append(yi)
        X = np.array(self.X)
        y = np.array(self.y)
        self.clf.fit(X, y)
        pl.cla()
        plot_decision_regions(X, y, self.clf, ax=self.ax)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 2)
        pl.draw()
