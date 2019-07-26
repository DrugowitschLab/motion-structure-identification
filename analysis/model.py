import pandas as pd
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from analysis.confusion_matrix import draw_matrix
import matplotlib.pyplot as plt


structures = ['IND', 'GLO', 'CLU', 'SDH']
s = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
masks = {
    'IND': [True, False, False, False, False, False, False, False],
    'GLO': [False, True, False, False, False, False, False, False],
    'CLU': [False, False, True, True, True, False, False, False],
    'SDH': [False, False, False, False, False, True, True, True],
}
mult = np.array([1/4, 1/4, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])


from utils.data import load_data
from sklearn.metrics import confusion_matrix
cm_data = load_data(f'../data/sichao/pilot_sichao.dat')
cm_ground_truth, cm_sichao_choice = [], []
for record in cm_data:
    cm_ground_truth.append(record['answer'])
    cm_sichao_choice.append(record['choice'])
cm_sichao = confusion_matrix(cm_ground_truth, cm_sichao_choice, structures)
cm_sichao = cm_sichao / cm_sichao.sum(axis=1).reshape(4, 1)


def extract_params(𝜃):
    α = 𝜃[0] if len(𝜃) == 5 else 0
    β = 𝜃[-4]
    # β = np.array([𝜃[0], 𝜃[1]] + [𝜃[2]] * 3 + [𝜃[3]] * 3)
    b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    # b = np.zeros(8)
    return α, β, b


def loss_fun(𝜃, df, count, disp=False):
    α, β, b = extract_params(𝜃)
    u = β * (df.iloc[:, :8] + b)
    normalizer = u.apply(logsumexp, axis=1)
    # ℓ = u.apply(lambda row: logsumexp(row[masks[df['target'][row.name]]]), axis=1) - normalizer
    # loss = -ℓ.sum()
    u = np.exp(u.subtract(normalizer, axis=0))
    u = (α * mult + (1 - α) * u)
    loss = u.apply(lambda row: sum(row[masks[df['target'][row.name]]]), axis=1)
    loss = -np.log(loss).sum()
    if disp:
        print(α, β, b)
        print(loss)
    return loss


def cm_fun(𝜃, df, count, disp=False):
    cm = predict(𝜃, df, 'target').groupby('ground_truth')[structures].mean().reindex(structures).to_numpy()
    loss = (np.exp(np.abs(cm - cm_sichao))).sum()
    loss += 5 * (np.exp(np.abs(cm - cm_sichao)[3, :])).sum()
    # loss += 2 * (np.abs(cm[3, :] - cm_sichao[3, :])).sum()
    print(loss)
    return loss


# def loss_jac(𝜃, df, count):
#     β, b = extract_params(𝜃)
#     u = df.iloc[:, :4].copy()
#     u.iloc[:, 1:] = u.iloc[:, 1:] + 𝜃[1:]
#     βu = u * 𝜃[0]
#     normalizer = βu.apply(logsumexp, axis=1)
#     βu = np.exp(βu.subtract(normalizer, axis=0))
#     dβ = ((βu * u.iloc[:, :4]).sum(axis=1) - u.apply(lambda row: row[df['target'][row.name]], axis=1)).sum()
#     db = 𝜃[0] * (βu.sum() - count)
#     grad = np.array([dβ, db[1], db[2], db[3]])
#     return grad


def fit(df, target, x0=np.array([0.01, 0.1, 0, 0, 0]),
        bounds=[(0.01, 0.99), (0.00001, None)] + [(None, None)] * 3, disp=True):
    df['target'] = df[target]
    df = df[s + ['target']]
    count = df['target'].value_counts()[structures].to_numpy()
    count = np.array([count[0], count[1], count[2], count[2], count[2], count[3], count[3], count[3]])
    method = 'SLSQP'
    x = minimize(loss_fun, x0, (df, count), bounds=bounds, method=method,  # jac=loss_jac,
                 options={'maxiter': 1000, 'disp': disp})
    return x


def predict(𝜃, df, ground_truth='ground_truth'):
    α, β, b = extract_params(𝜃)
    u = β * (df.iloc[:, :8] + b)
    normalizer = u.apply(logsumexp, axis=1)
    u = np.exp(u.subtract(normalizer, axis=0))
    u = (α * mult + (1 - α) * u)
    u['ground_truth'] = df[ground_truth]
    u['CLU'] = u[['CLU_012', 'CLU_120', 'CLU_201']].sum(axis=1)
    u['SDH'] = u[['SDH_012', 'SDH_120', 'SDH_201']].sum(axis=1)
    return u


def plot_prediction(df=None, file=None, target='ground_truth', lapse=True, x0=None, 𝜃=None):
    if file:
        df = pd.read_csv(f'../data/{file}', dtype={l: float for l in s})
        df = df[s + ['ground_truth', 'Johannes_choice', 'Sichao_choice']]
    if x0 is None:
        x0 = [0.02, 0.1, 0, 0, 0] if lapse else [0.01, 0, 0, 0]
    bounds = ([(0.001, 0.999), (0.000001, None)] + [(None, None)] * 3) if lapse else None
    if 𝜃 is None:
        𝜃 = fit(df, target, np.array(x0), bounds).x
    print(𝜃)
    u = predict(𝜃, df)
    cm = u.groupby('ground_truth')[structures].mean().reindex(structures)
    print(cm)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    draw_matrix(cm.to_numpy(), structures, structures, True)


if __name__ == '__main__':
    pd.options.display.float_format = '{:,.6f}'.format
    plot_prediction(file='pilot_1_0.80.csv', target='Sichao_choice', lapse=True, x0=np.array([0.01, 0.1, 0, 0, 0]))
    # df = pd.read_csv(f'../data/pilot_0.csv', dtype={l: float for l in s})
    # df = df[s + ['ground_truth', 'Johannes_choice', 'Sichao_choice']]
    # df['target'] = df['Johannes_choice']
    # print(loss_fun(np.array([0.01, 0.1, 0, 0, 0]), df, None))
