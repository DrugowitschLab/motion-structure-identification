from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, TypeVar
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import OptimizeResult
from os.path import exists, join
import multiprocessing as mp

from stimuli.motion_structure import MotionStructure
import analysis.models as models
from config import root, ExperimentConfig as ExpConfig
n_dots = ExpConfig.n_dots

ModelType = TypeVar('ModelType', bound=models.Model)


class Data(ABC):
    eps = 1e-5
    pids: List[str] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']  # 2-step de-identified pseudo ids
    keys: List[str] = ['ground_truth', 'choice', 'confidence']
    structures: List[str]

    def __init__(self, file: Optional[str] = None):
        import pickle
        self.file = join(root, 'data', file)
        self.data = []
        with open(self.file, 'rb') as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break
        self.n_trials = len(self.data)
        self.idx = np.arange(self.n_trials)
        self.df = pd.DataFrame({key: self.extract(key) for key in self.keys})

    def extract(self, key):
        value = []
        for i in self.idx:
            value.append(self.data[i][key])
        return value

    def build_model(self, Model: Type[ModelType], verbose: bool = False) -> ModelType:
        if issubclass(Model, models.BayesianIdealObserver):
            return Model(lambda σ_obs=0.: self.apply_kalman_filters(σ_obs), verbose)
        elif issubclass(Model, models.NonBayesianChoiceModel):
            return Model(lambda norm=2.: self.measure_r_distance(self.permuted_structures(), norm), verbose)
        else:
            raise NotImplementedError

    def load_model(self, Model: Type[models.Model], res: OptimizeResult) -> models.Model:
        if issubclass(Model, models.BayesianIdealObserver):
            model = self.build_model(models.BayesianIdealObserver)
            model.load_res(res)
            return model
        elif issubclass(Model, models.NonBayesianChoiceModel):
            model = self.build_model(models.NonBayesianChoiceModel)
            model.load_res(res)
            return model
        else:
            raise NotImplementedError

    def cross_validate(self, Model: Type[models.Model], shuffle: bool=False) -> pd.DataFrame:
        csv_file = f'{self.file[:-4]}_{Model.name}_leave-1-out.csv'
        if shuffle:
            df = pd.DataFrame({'ground_truth': self.df['ground_truth'], 'idx': self.idx})
            idx = df.groupby('ground_truth')['idx'].transform(np.random.permutation)
        else:
            idx = self.idx
            if exists(csv_file):
                return pd.read_csv(csv_file)
        p = np.zeros((self.n_trials, len(self.structures)))
        for idx_train, idx_test in KFold(n_splits=self.n_trials).split(self.idx):
            # training
            self.idx = idx[idx_train]
            res = self.build_model(Model).fit()
            # testing
            self.idx = idx[idx_test]
            p[idx_test] += self.load_model(Model, res).predict(res)
        self.idx = np.arange(self.n_trials)
        df = pd.DataFrame(p, columns=self.structures)
        if not shuffle:
            df.to_csv(csv_file, index=False)
        return df

    def empirical_velocity(self) -> np.ndarray:
        v = []
        for i in self.idx:
            x = np.array(self.data[i]['φ'])[:, :n_dots]
            t = np.array(self.data[i]['t'])
            x = x[1:] - x[:-1]
            x = np.where(x < np.pi, x, x - 2 * np.pi)
            x = np.where(x > -np.pi, x, x + 2 * np.pi)
            t = t[1:] - t[:-1]
            v.append(x / t.reshape((-1, 1)))
        return np.array(v)

    def measure_r_distance(self, structures: Dict[str, MotionStructure], p=2):
        def _extract_corr(Σ: np.ndarray):
            return np.array([Σ[i, j] / np.sqrt(Σ[i, i] * Σ[j, j]) for i in range(n_dots) for j in range(i + 1, n_dots)])
        observed_corr = np.array([_extract_corr(np.cov(v_trial.T)) for v_trial in self.empirical_velocity()])
        ideal_corr = {s: _extract_corr(structures[s].Σ) for s in structures}
        df = pd.DataFrame({s: -np.linalg.norm(observed_corr - ideal_corr[s], p, axis=1) for s in structures})
        for k in self.keys:
            df[k] = self.extract(k)
        return df

    def apply_kalman_filters(self, σ_obs: float = 0., σ_imp: float = 0., reps: int = 1, file: Optional[str] = None,
                             glo: float = ExpConfig.glo_H, λ_I: float = ExpConfig.λ_I) -> pd.DataFrame:
        file = file if file is not None else f'{self.file[:-4]}_σ_obs={σ_obs:.2f}_glo={glo:.2f}_λ_I={λ_I:.2f}.csv'
        df = self._apply_kalman_filters(self.permuted_structures(glo, λ_I), σ_obs, σ_imp, reps, csv_file=file)
        return df

    def _apply_kalman_filters(self, structures: Dict[str, MotionStructure], σ_obs: float = 0., σ_imp: float = 0.,
                              reps: int = 1, csv_file: Optional[str] = None):
        from analysis.kalman_filter import apply_kalman_filter as kalman
        if exists(csv_file):
            return pd.read_csv(csv_file, dtype={s: float for s in structures}).loc[self.idx].reset_index(drop=True)
        df = pd.DataFrame(
            data=(
                [kalman(np.random.normal(trial['φ'], σ_imp), trial['t'], structures[s], σ_obs) for s in structures] +
                [trial[k] for k in self.keys] for _ in range(reps) for trial in self.data
            ),
            columns=[s for s in structures] + self.keys,
        )
        if csv_file:
            df.to_csv(csv_file, index=False)
        return df.loc[self.idx].reset_index(drop=True)

    @staticmethod
    @abstractmethod
    def permuted_structures(glo: float = ExpConfig.glo_H, λ_I: float = ExpConfig.λ_I) -> Dict[str, MotionStructure]:
        pass

    @staticmethod
    def _score(df):
        df['confidence'] = ((df['confidence']) == 'high') * 2 + 1
        return ((df['choice'] == df['ground_truth']) * 1. - 0.5) * df['confidence'] + 0.5


def pool(DataType: Type[Data]):
    class PooledData(DataType):
        def __init__(self):
            super().__init__('pooled')
            self.datas = []
            for pid in self.pids:
                data = DataType(pid)
                self.datas.append(data)
                self.data += data.data
                self.n_trials += data.n_trials
            self.idx = np.arange(self.n_trials)
            self.df = pd.DataFrame({key: self.extract(key) for key in ['ground_truth', 'choice', 'confidence']})

        def apply_kalman_filters(self, *args, **kwargs):
            return pd.concat([data.apply_kalman_filters(*args, **kwargs) for data in self.datas], ignore_index=True)

    return PooledData()


if __name__ == '__main__':
    print(__file__)
    for idx_train, idx_test in KFold(n_splits=4).split([4,2,3,4]):
        print(idx_train)
