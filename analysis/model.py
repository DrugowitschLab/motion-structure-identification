import pandas as pd
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from collections import Counter


class StructureDecisionModel:
    def __init__(self, df, col_target, col_structures, structures,
                 param_extractor, bounds, normalize=True):
        self.df = df
        self.L = df[col_structures].to_numpy()
        self.mask = np.column_stack([df[col_target] == s for s in structures])
        if normalize:
            counter = Counter(df['ground_truth'])
            self.normalizer = np.array([1 / counter[s] for s in df['ground_truth']]).reshape((len(df), 1))
        else:
            self.normalizer = np.ones((len(df), 1))
        counter = Counter(structures)
        self.multiplicity = np.array([counter[s] for s in structures])
        self.structures = structures
        self.unique_structures = list(counter.keys())
        # self.cm = confusion_matrix(df['ground_truth'], df[col_target], self.unique_structures)
        self.prior = 1 / len(self.unique_structures) / self.multiplicity
        self.param_extractor = param_extractor
        self.bounds = bounds
        self.verbose = 1

    def loss_choice(self, 𝜃):
        loss = -np.dot(np.log((self.predict(𝜃) * self.mask).sum(axis=1)), self.normalizer)
        if self.verbose > 1:
            print(f'loss={loss}')
        return loss

    def loss_cm(self, 𝜃, normalize=False):
        target_cm = self.cm / self.cm.sum(axis=1).reshape(4, 1) if normalize else self.cm
        predicted_cm = self.predict_confusion_matrix(𝜃, normalize=normalize)
        loss = (np.abs(target_cm - predicted_cm).sum()) ** 1
        if self.verbose > 1:
            print(f'loss={loss}')
        return loss

    def fit(self, 𝜃_0, method='SLSQP', loss='choice', verbose=1):
        self.verbose = verbose
        if loss == 'choice':
            loss = self.loss_choice
        elif loss == 'cm':
            loss = self.loss_cm
        res = minimize(loss, 𝜃_0, bounds=self.bounds, method=method,
                 options={'maxiter': 1000, 'disp': verbose > 0})
        if verbose > 0:
            print(res)
        return res

    def predict(self, 𝜃):
        α, β, b = self.param_extractor(𝜃)
        if self.verbose >= 1:
            # print(f'𝜃={𝜃}')
            print(f'α={α}, β={β}, b={b}')
        u = β * (self.L + b) - np.log(self.multiplicity)
        p = α * self.prior + (1 - α) * np.exp(u - logsumexp(u, axis=1, keepdims=True))
        return p

    def predict_confusion_matrix(self, 𝜃, normalize=False, ax=None):
        p = self.predict(𝜃)
        df = pd.DataFrame({s: np.sum([p[:, i] for i in range(len(self.structures)) if self.structures[i] == s], axis=0)
                           for s in self.unique_structures})
        df['ground_truth'] = self.df['ground_truth']
        cm = df.groupby('ground_truth')[self.unique_structures].sum().reindex(self.unique_structures).to_numpy()
        if normalize:
            cm /= cm.sum(axis=1).reshape(4, 1)
        if ax:
            from analysis.confusion_matrix import draw_matrix
            import matplotlib.pyplot as plt
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Ground truth')
            draw_matrix(cm, self.unique_structures, self.unique_structures, normalize=normalize, ax=ax)
        return cm


if __name__ == '__main__':
    pass
