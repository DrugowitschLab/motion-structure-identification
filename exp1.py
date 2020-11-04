from os.path import join
from experiment import Experiment
from stimuli.motion_structure import MotionStructure
from config import ExperimentConfig


class Exp1(Experiment):
    directory = join(Experiment.directory, 'exp1')
    structures = ['I', 'G', 'C', 'H']
    p_structures = [0.25, 0.25, 0.25, 0.25]
    presets = {
        'I': MotionStructure(1, ExperimentConfig.λ_T),
        'G': MotionStructure(1, ExperimentConfig.λ_I),
        'C': MotionStructure(0, ExperimentConfig.λ_I),
        'H': MotionStructure(ExperimentConfig.glo_H, ExperimentConfig.λ_I)
    }
    confidence_score = {(True, 'high'): 2, (True, 'low'): 1, (False, 'low'): 0, (False, 'high'): -1}


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        exp = Exp1(argv[1], ExperimentConfig.n_trials_exp1, 2)
    else:
        exp = Exp1('presentation', 200, 2, is_fullscreen=False)
