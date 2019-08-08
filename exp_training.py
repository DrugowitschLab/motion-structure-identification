from exp1 import Exp1


class Training(Exp1):
    directory = 'data/training'

    def create_counterbalance(self, n_rep, seeds_file=None):
        import numpy as np
        super().create_counterbalance(n_rep, seeds_file)
        self.truth = np.concatenate([
            np.array(['IND'] * 3 + ['GLO'] * 3 + ['CLU'] * 3 + ['SDH'] * 6),
            np.random.choice(self.structures, 15)
        ])

    def init(self):
        super().init()
        self.motion.prompt(f"Next structure: {self.truth[self.idx]}\n"
                           f"Click <left mouse button> or \n"
                           f"press <space> to continue.")

    def choice_made(self, data):
        super().choice_made(data)
        self.motion.prompt(f"Next structure: {self.truth[self.idx]}\n"
                           f"Click <left mouse button> or \n"
                           f"press <space> to continue.")


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        exp = Training(argv[1], 30)
    else:
        exp = Training('sichao', 30)
