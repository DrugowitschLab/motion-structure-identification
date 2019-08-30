from exp1 import Exp1


class Training(Exp1):
    directory = 'data/training'

    def create_counterbalance(self, n_rep, seeds_file=None):
        import numpy as np
        super().create_counterbalance(n_rep, seeds_file)
        self.truth = np.concatenate([
            np.array(['IND'] * 1 + ['GLO'] * 1 + ['CLU'] * 1 + ['SDH'] * 1),
            np.array(['IND'] * 3 + ['GLO'] * 3 + ['CLU'] * 3 + ['SDH'] * 3),
            np.array(['IND'] * 5 + ['GLO'] * 4 + ['CLU'] * 5 + ['SDH'] * 5),
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
