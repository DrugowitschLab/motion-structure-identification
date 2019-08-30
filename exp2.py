from experiment import Experiment
from stimuli.motion_structure import MotionStructure
from config import ExperimentConfig


class Exp2(Experiment):
    directory = 'data/exp2'
    structures = ('CLU', 'SDH')
    presets = {f'{g:.2f}': MotionStructure(g, ExperimentConfig.λ_I) for g in ExperimentConfig.glo_exp2}
    confidence = ('low', 'high')
    confidence_score = {'low': {True: 0, False: 0}, 'high': {True: 0, False: 0}}

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
        exp = Exp2(argv[1], 100)
    else:
        exp = Exp2('presentation', 100, is_fullscreen=False)

