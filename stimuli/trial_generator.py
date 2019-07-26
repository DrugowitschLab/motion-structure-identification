from stimuli.simulation import StructuredMotionSimulation as Sim
from stimuli.motion_structure import MotionStructure
from utils.data import SimLogger
from config import ExperimentConfig
from analysis.kalman_filter import apply_filters_on_trial
import numpy as np
import pandas as pd
from analysis.model import fit, predict


structures = ['IND', 'GLO', 'CLU', 'SDH']
s = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
p_structures = [0.25, 0.25, 0.25, 0.25]
glo_SDH = 2 / 3
glo_SDH = 0.85
presets = {'IND': MotionStructure(0, 2),
           'GLO': MotionStructure(1, 1 / 4),
           'CLU': MotionStructure(0, 1 / 4),
           'SDH': MotionStructure(glo_SDH, 1 / 4)}
p = np.array([0, 0, 0, 1])
rng = np.random.RandomState()


def generate_trial(structure, seed=None):
    sim = Sim(structure, seed=seed)
    logger = SimLogger()
    logger.reset(sim.φ.copy(), sim.r.copy())
    for frame in range(ExperimentConfig.delay, ExperimentConfig.duration + 1):
        logger.log(*sim.advance())
    return logger.data


def generate_trials(n_trials=200, file=None, σ_R=1.0):
    if file:
        df = pd.read_csv(f'../data/{file}', dtype={l: float for l in s})
    else:
        vals = []
        ground_truth = []
        for i in range(n_trials):
            print(i)
            structure = rng.choice(structures, p=p)
            data = generate_trial(presets[structure])
            t = data['t']
            x = data['φ']
            vals.append(np.array(apply_filters_on_trial(x, t, σ_R, glo_SDH)))
            ground_truth.append(structure)
        df = pd.DataFrame(vals, columns=s)
        df['ground_truth'] = ground_truth
    return df


def recover_𝜃(𝜃_0, 𝜃_star, file=None):
    df = generate_trials(file)
    df['target'] = predict(𝜃_star, df).apply(lambda row: rng.choice(structures, p=row[structures].to_numpy(dtype=float)), axis=1)
    return fit(df, 'target', 𝜃_0).x


if __name__ == '__main__':
    # 𝜃_star = [0.2229186, 7.03126068, 1.86373238, -1.10840216]
    # 𝜃_0 = [0.1, 0, 0, 0]
    # 𝜃_fit = []
    # for i in range(100):
    #     print(i)
    #     𝜃_fit.append(recover_𝜃(𝜃_0, 𝜃_star, file='pilot_0.csv'))
    # 𝜃_fit = np.array(𝜃_fit)
    # print('𝜃_star = ', 𝜃_star)
    # print('𝜃_0 = ', 𝜃_0)
    # print('mean(𝜃_fit) = ', 𝜃_fit.mean(axis=0))
    # print('std(𝜃_fit) = ', 𝜃_fit.std(axis=0))

    from analysis.model import plot_prediction
    plot_prediction(
                    generate_trials(200, σ_R=0.1),
                    # file='../data/pilot_1e-1.csv',
                    𝜃=np.array([0, 0.1, 0, 0, 0]))
