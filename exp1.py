from experiment import Experiment
from stimuli.motion_structure import MotionStructure
from config import ExperimentConfig


class Exp1(Experiment):
    directory = 'data/exp1'
    structures = ('IND', 'GLO', 'CLU', 'SDH')
    p_structures = [0.25, 0.25, 0.25, 0.25]
    presets = {
        'IND': MotionStructure(1, ExperimentConfig.λ_T),
        'GLO': MotionStructure(1, ExperimentConfig.λ_I),
        'CLU': MotionStructure(0, ExperimentConfig.λ_I),
        'SDH': MotionStructure(ExperimentConfig.glo_SDH, ExperimentConfig.λ_I)
    }
    confidence = ('low', 'high')
    confidence_score = {'low': {True: 1, False: 0}, 'high': {True: 2, False: -1}}


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        exp = Exp1(argv[1], 200, 2)
    else:
        exp = Exp1('sichao', 200, 2)
