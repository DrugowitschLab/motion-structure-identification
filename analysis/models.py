from abc import ABC
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize, OptimizeResult
from collections import Counter
import matplotlib.pyplot as plt

ε = 1e-5
π_L_shared: float = 0.14
σ_obs_shared: float = 1.1
r_shared: float = 1.0


class Model(ABC):
    structures: List[str]
    permuted_structures: np.ndarray
    df: pd.DataFrame
    L: np.ndarray
    is_chosen: np.ndarray
    p_uniform: np.ndarray
    L_uniform: np.ndarray
    verbose: bool

    π_L: float = 0.
    β: Union[float, np.ndarray] = 1.
    b: np.ndarray

    name: str = ''
    marker: str = ''
    color: str = ''

    def __init__(self, get_df: Callable[..., pd.DataFrame], verbose: bool = False):
        self.df, self.verbose = get_df(), verbose
        self.L = self.df.iloc[:, :-3].to_numpy()
        self.get_L = lambda *args: get_df(*args).iloc[:, :-3].to_numpy()
        self.permuted_structures = np.array([s.split('_')[0] for s in self.df.columns[:-3]])
        counter = Counter(self.permuted_structures)
        self.structures, self.multiplicity = list(counter.keys()), list(counter.values())
        self.is_chosen = np.column_stack([self.df['choice'] == s for s in self.permuted_structures])
        self.p_uniform = 1 / len(counter) / np.repeat(self.multiplicity, self.multiplicity)
        self.L_uniform = np.log(1 / np.repeat(self.multiplicity, self.multiplicity))
        self.b = np.zeros(len(self.structures) - 1)

    def fit(self, method: str = 'SLSQP') -> OptimizeResult:
        θ_0 = self._vectorize_free_params()
        if len(θ_0) > 0:
            res = minimize(self._loss, θ_0, bounds=self._bounds(), method=method, options={'disp': self.verbose})
        else:
            res = OptimizeResult({'x': θ_0, 'success': True, 'fun': self._loss(θ_0)})
        self._build_res(res)
        self._devectorize_free_params(θ_0)
        return res

    def predict(self, res: OptimizeResult) -> pd.DataFrame:
        p = self.predict_permuted(res.x)    # use fitted parameters to generate predictions
        df = pd.DataFrame({s: np.sum(p[:, np.array(self.permuted_structures) == s], axis=1) for s in self.structures})
        return df

    def predict_permuted(self, θ: np.ndarray) -> np.ndarray:
        self._devectorize_free_params(θ)
        b = np.repeat([0] + list(self.b), self.multiplicity)
        β = self.β if isinstance(self.β, float) else np.repeat(self.β, self.multiplicity)
        if self.verbose:
            print(f"π={self.π_L}, β={β}, b={b}")
        L: np.ndarray = β * (self.L + b + self.L_uniform)
        p: np.ndarray = self.π_L * self.p_uniform + (1 - self.π_L) * np.exp(L - logsumexp(L, axis=1, keepdims=True))
        return p

    def plot_confusion_matrix(self, prediction: pd.DataFrame, ax: Optional[plt.Axes] = None) -> np.ndarray:
        from analysis.utils.confusion_matrix import plot_confusion_matrix
        prediction['ground_truth'] = self.df['ground_truth']
        cm = prediction.groupby('ground_truth')[self.structures].sum().reindex(self.structures).to_numpy()
        if ax:
            plot_confusion_matrix(cm, self.structures, self.structures, ax)
            ax.set_xlabel('Prediction')
            ax.set_ylabel('True Structure')
        return cm

    def load_res(self, res: OptimizeResult):
        self.π_L = res.π_L
        self.β = res.β
        self.b = res.b

    def _build_res(self, res: OptimizeResult):
        res.log_likelihood = -res.fun
        res.predictive_power = (self.predict_permuted(res.x) * self.is_chosen).sum() / len(self.df)
        res.aic = res.log_likelihood - self._n_free_params()
        res.bic = res.log_likelihood - self._n_free_params() * np.log(len(self.df)) / 2
        res.π_L = self.π_L
        res.β = self.β
        res.b = self.b

    def _loss(self, θ: np.ndarray) -> float:
        loss = -sum(np.log(np.maximum((self.predict_permuted(θ) * self.is_chosen).sum(axis=1), ε)))
        if self.verbose:
            print(f'loss={loss}')
        return loss

    def _n_free_params(self) -> int:
        return len(self._bounds())

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return []

    def _vectorize_free_params(self) -> np.ndarray:
        return np.array([])

    def _devectorize_free_params(self, θ: np.ndarray):
        pass


class BayesianIdealObserver(Model):
    """ Bayesian ideal observer.
        Free parameters: None.
        Fixed parameters: None. """
    σ_obs = σ_obs_shared    # Inject observation noise
    get_L: Callable[[float], np.ndarray]
    name: str = 'Ideal observer'

    def __init__(self, get_df: Callable[[float], pd.DataFrame], verbose: bool = False):
        super().__init__(lambda: get_df(self.σ_obs), verbose)
        self.get_L = lambda *args: get_df(*args).iloc[:, :-3].to_numpy()

    def load_res(self, res: OptimizeResult):
        super().load_res(res)
        if self.σ_obs != res.σ_obs:
            self.σ_obs = res.σ_obs
            self.L = self.get_L(self.σ_obs)

    def _build_res(self, res: OptimizeResult):
        super()._build_res(res)
        res.σ_obs = self.σ_obs


class BiasFreeChoiceModel(BayesianIdealObserver):
    """ Suboptimal Bayesian observer with observation noise, lapse, and fixed softmax choice.
        Free parameters: β.
        Fixed parameters: σ_obs, π_L. """
    π_L = π_L_shared            # Inject lapse probability
    β = 0.1                    # Initial guess of β
    name: str = 'Bias-free'
    marker: str = 's'
    color: str = 'y'

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)]

    def _vectorize_free_params(self) -> np.ndarray:
        return np.array([self.β])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β = θ[0]


class ChoiceModel4Param(BiasFreeChoiceModel):
    """ Suboptimal Bayesian observer with observation noise, lapse, biases, and fixed softmax choice.
        Free parameters: β, b.
        Fixed parameters: σ_obs, π_L. """
    name: str = 'Full model'
    marker: str = 'o'
    color: str = 'b'

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([[self.β], self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β, self.b = θ[0], θ[1:]


class LapseFreeChoiceModel(ChoiceModel4Param):
    """ Suboptimal Bayesian observer with observation noise, biases, and fixed softmax choice.
        Free parameters: β, b.
        Fixed parameters: σ_obs. """
    π_L = 0.
    name: str = 'Lapse-free'
    marker: str = '^'
    color: str = 'g'


class ChoiceModel5Param(ChoiceModel4Param):
    """ Suboptimal Bayesian observer with observation noise, lapse, biases, and fixed softmax choice.
        Free parameters: π_L, β, b.
        Fixed parameters: σ_obs. """

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] * 2 + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([[self.π_L, self.β], self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.π_L, self.β, self.b = θ[0], θ[1], θ[2:]


class ChoiceModel6Param(ChoiceModel5Param):
    """ Suboptimal Bayesian observer with observation noise, lapse, biases, and fixed softmax choice.
        Free parameters: σ_obs, π_L, β, b.
        Fixed parameters: None. """
    name: str = '6-param model'

    def __init__(self, get_df: Callable[..., pd.DataFrame], verbose: bool = False,
                 Σ: np.ndarray = np.arange(0.0, 2.1, 0.1)):
        super().__init__(get_df, verbose)
        self.Σ: np.ndarray = Σ

    def predict(self, res: OptimizeResult):
        self.L = self.get_L(res.σ_obs)
        return super().predict(res)

    def fit(self, method: str = 'SLSQP'):
        self.σ_obs, res = 0, OptimizeResult({'fun': np.Inf})
        for _σ_obs in self.Σ:
            self.L = self.get_L(_σ_obs)
            _res = super().fit(method)
            if self.verbose:
                print(f"σ={_σ_obs:.2f} loss={_res.fun} x={_res.x}")
            if _res.fun < res.fun:
                self.σ_obs, res = _σ_obs, _res
        self._build_res(res)
        return res

    def _n_free_params(self) -> int:
        return len(self._bounds()) + 1


class ChoiceModel7Param(ChoiceModel4Param):
    def __init__(self, get_df: Callable[..., pd.DataFrame], verbose: bool = False):
        super().__init__(get_df, verbose)
        self.β = np.ones(len(self.structures)) * self.β

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] * (len(self.structures)) + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([self.β.tolist(), self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β, self.b = θ[:len(self.structures)], θ[len(self.structures):]


class NonBayesianChoiceModel(Model):
    """ Correlation based decision maker. """
    π_L = 0.30          # Inject lapse probability
    β = 0.1             # Initial guess of β
    r: float = 1.0      # Norm
    name: str = 'Non-Bayesian'
    marker: str = 'x'
    color: str = 'r'

    def __init__(self, get_df: Callable[[float], pd.DataFrame], verbose: bool = False):
        super().__init__(lambda: get_df(self.r), verbose)
        self.get_L = lambda *args: get_df(*args).iloc[:, :-3].to_numpy()


class NonBayesianChoiceModel4Param(NonBayesianChoiceModel):
    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([[self.β], self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β, self.b = θ[0], θ[1:]


class NonBayesianChoiceModel5Param(NonBayesianChoiceModel):
    def predict_permuted(self, θ: np.ndarray) -> np.ndarray:
        self._devectorize_free_params(θ)
        self.L = self.get_L(self.r)
        return super().predict_permuted(θ)

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None), (ε, None)] + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([[self.β, self.r], self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β, self.r, self.b = θ[0], θ[1], θ[2:]


class BiasFreeNonBayesianChoiceModel4Param(NonBayesianChoiceModel4Param):
    def __init__(self, get_df: Callable[..., pd.DataFrame], verbose: bool = False):
        super().__init__(get_df, verbose)
        self.β = np.ones(len(self.structures)) * self.β

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] * (len(self.structures))

    def _vectorize_free_params(self) -> np.ndarray:
        return self.β

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β = θ


class NonBayesianChoiceModel7Param(NonBayesianChoiceModel4Param):
    def __init__(self, get_df: Callable[..., pd.DataFrame], verbose: bool = False):
        super().__init__(get_df, verbose)
        self.β = np.ones(len(self.structures)) * self.β

    def _bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(0, None)] * (len(self.structures)) + [(None, None)] * (len(self.structures) - 1)

    def _vectorize_free_params(self) -> np.ndarray:
        return np.concatenate([self.β.tolist(), self.b.tolist()])

    def _devectorize_free_params(self, θ: np.ndarray):
        self.β, self.b = θ[:len(self.structures)], θ[len(self.structures):]
