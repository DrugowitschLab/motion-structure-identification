from experiment import Experiment
from exp1 import Exp1
from os.path import join


class Training(Exp1):
    directory = join(Experiment.directory, 'training')

    def create_counterbalance(self, n_rep, seeds_file=None):
        import numpy as np
        super().create_counterbalance(n_rep, seeds_file)
        self.truth = np.concatenate([
            np.array(['I'] * 1 + ['G'] * 1 + ['C'] * 1 + ['H'] * 1),
            np.array(['I'] * 3 + ['G'] * 3 + ['C'] * 3 + ['H'] * 3),
            np.array(['I'] * 5 + ['G'] * 4 + ['C'] * 5 + ['H'] * 5),
        ])
        np.random.shuffle(self.truth[16:])

    def init(self):
        super().init()
        if self.idx < self.n_trials:
            self.motion.prompt(f"Next structure: {self.truth[self.idx - 1]}\n"
                               f"Click <left mouse button> or \n"
                               f"press <space> to continue.")

    def choice_made(self, data):
        super().choice_made(data)
        if self.idx < self.n_trials:
            self.motion.prompt(f"Next structure: {self.truth[self.idx]}\n"
                               f"Click <left mouse button> or \n"
                               f"press <space> to continue.")


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        exp = Training(argv[1], 35)
    else:
        exp = Training('sichao', 35)
