import pandas as pd
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from analysis.confusion_matrix import draw_matrix
import matplotlib.pyplot as plt


structures = ['IND', 'GLO', 'CLU', 'SDH']
s = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
mult = np.array([1/4, 1/4, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])


# from utils.data import load_data
# from sklearn.metrics import confusion_matrix
# cm_data = load_data(f'../data/sichao/pilot_sichao.dat')
# cm_ground_truth, cm_sichao_choice = [], []
# for record in cm_data:
#     cm_ground_truth.append(record['answer'])
#     cm_sichao_choice.append(record['choice'])
# cm_sichao = confusion_matrix(cm_ground_truth, cm_sichao_choice, structures)
# cm_sichao = cm_sichao / cm_sichao.sum(axis=1).reshape(4, 1)
#
# def cm_fun(𝜃, df, disp=False):
#     cm = predict(𝜃, df, 'target').groupby('ground_truth')[structures].mean().reindex(structures).to_numpy()
#     loss = (np.exp(np.abs(cm - cm_sichao))).sum()
#     loss += 5 * (np.exp(np.abs(cm - cm_sichao)[3, :])).sum()
#     # loss += 2 * (np.abs(cm[3, :] - cm_sichao[3, :])).sum()
#     print(loss)
#     return loss


def extract_params(𝜃):
    α = 0.05
    β = 1
    b = np.zeros(8)
    if len(𝜃) == 1:
        β = 𝜃[0]
    elif len(𝜃) == 2:
        α = 𝜃[0]
        β = 𝜃[1]
    elif len(𝜃) == 3:
        b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    elif len(𝜃) == 4:
        β = 𝜃[0]
        b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    elif len(𝜃) == 5:
        α = 𝜃[0]
        β = 𝜃[1]
        b = np.array([0, 𝜃[-3]] + [𝜃[-2]] * 3 + [𝜃[-1]] * 3)
    return α, β, b


def loss_fun(𝜃, df, mask, disp):
    α, β, b = extract_params(𝜃)
    u = β * (df.iloc[:, :8].to_numpy() + b) - np.log(1 / 4 / mult)
    p = (α * mult + (1 - α) * np.exp(u - logsumexp(u, axis=1, keepdims=True)))
    loss = -np.log((p * mask).sum(axis=1)).sum()
    if disp:
        print(f'α={α}, β={β}, b={b}')
        print(f'loss={loss}')
    return loss


# def loss_jac(𝜃, df, mask, disp):
#     α, β, b = extract_params(𝜃)
#     u = (df.iloc[:, :8] + b).to_numpy() - 1 / 4 / mult
#     ℓ = β * u
#     normalizer = logsumexp(ℓ, axis=1, keepdims=True)
#     p = np.exp(ℓ - normalizer)
#     denominator = ((α * mult + (1 - α) * p) * mask).sum(axis=1, keepdims=True)
#     dα = -(((mult - p) * mask).sum(axis=1, keepdims=True) / denominator).sum()
#     A = (1 - α) * p / denominator
#     dβ = -(A * (u - (u * p).sum(axis=1, keepdims=True)) * mask).sum()
#     db = -(A * β * (1 - p) * mask).sum(axis=0)
#     if disp:
#         print(dα, dβ, db)
#     return np.array([dα, dβ, db[1], db[2:5].sum(), db[5:8].sum()])


def fit(df, mask, target, x0=np.array([0.1, 1, 0, 0, 0]), method='SLSQP', disp=False):
    eps = 1e-3
    if len(x0) == 2:
        bounds = [(eps, 1.), (0., None)]
    elif len(x0) == 4:
        bounds = [(eps, 1.), (None, None), (None, None), (None, None)]
    elif len(x0) == 5:
        bounds = [(eps, 1.), (0., None), (None, None), (None, None), (None, None)]
    else:
        bounds = None
    df['target'] = df[target]
    df = df[s + ['target']]
    # count = df['target'].value_counts()[structures].to_numpy()
    # count = np.array([count[0], count[1], count[2], count[2], count[2], count[3], count[3], count[3]])
    x = minimize(loss_fun, x0, (df, mask, disp), bounds=bounds, method=method,  # jac=loss_jac,
                 options={'maxiter': 1000, 'disp': disp})
    return x


def predict(𝜃, df, ground_truth='ground_truth'):
    α, β, b = extract_params(𝜃)
    u = β * (df.iloc[:, :8] + b) - np.log(1 / 4 / mult)
    normalizer = u.apply(logsumexp, axis=1)
    u = np.exp(u.subtract(normalizer, axis=0))
    u = (α * mult + (1 - α) * u)
    u['ground_truth'] = df[ground_truth]
    u['CLU'] = u[['CLU_012', 'CLU_120', 'CLU_201']].sum(axis=1)
    u['SDH'] = u[['SDH_012', 'SDH_120', 'SDH_201']].sum(axis=1)
    return u


def plot_prediction(df=None, file=None, target='ground_truth', x0=None, 𝜃=None, method='SLSQP'):
    if file:
        df = pd.read_csv(f'../data/{file}', dtype={l: float for l in s})
        df = df[s + ['ground_truth', target]]
    mask = np.column_stack([df[target] == structure for structure in ['IND', 'GLO'] + ['CLU'] * 3 + ['SDH'] * 3])

    # print(loss_jac(x0, df, mask, False))
    # print(loss_fun(x0, df, mask, False))
    # input()

    if 𝜃 is None:
        res = fit(df, mask, target, np.array(x0), method, disp=True)
        print(res)
        𝜃 = res.x
    u = predict(𝜃, df)
    cm = u.groupby('ground_truth')[structures].mean().reindex(structures)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    draw_matrix(cm.to_numpy(), structures, structures, True)

    # pd.options.display.float_format = '{:,.6f}'.format
    # print(pd.concat([u[['GLO', 'CLU', 'SDH', 'ground_truth']], df['Sichao_choice']], axis=1))


if __name__ == '__main__':
    plot_prediction(file='exp1/sichao_0806/sichao_0806_σ=0.00.csv', target='choice', method='SLSQP', x0=np.array([0.1, 0, 0, 0]))

    # plot_prediction(file='pilot_johannes_glo=0.77_v=1.5_3.csv', target='Johannes_choice', lapse=True, method='SLSQP', x0=np.array([0.01, 0.1]))

    # df = pd.read_csv(f'../data/pilot_0.csv', dtype={l: float for l in, (None, None) s})
    # df = df[s + ['ground_truth', 'Johannes_choice', 'Sichao_choice']]
    # df['target'] = df['Johannes_choice']
    # print(loss_fun(np.array([0.01, 0.1, 0, 0, 0]), df, None))

    # TNC, 0.1, 0.01
