from os.path import join
from experiment import Experiment
from stimuli.motion_structure import MotionStructure
from config import ExperimentConfig


class Exp2(Experiment):
    directory = join(Experiment.directory, 'exp2')
    structures = ['C', 'H']
    presets = {f'{g:.2f}': MotionStructure(g, ExperimentConfig.λ_I) for g in ExperimentConfig.glo_exp2}
    confidence_score = {(True, 'high'): 0, (True, 'low'): 0, (False, 'low'): 0, (False, 'high'): 0}

    def create_axes(self):
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes((0.81, 0.01, 0.18, 0.48)),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }

    def create_counterbalance(self, n_rep, seeds_file=None):
        import numpy as np
        super().create_counterbalance(n_rep, seeds_file)
        self.truth = np.array(list(self.presets.keys()) * (self.n_trials // len(self.presets)))
        np.random.shuffle(self.truth)


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        exp = Exp2(argv[1], ExperimentConfig.n_trials_exp2)
    else:
        exp = Exp2('presentation', 100, is_fullscreen=False)

