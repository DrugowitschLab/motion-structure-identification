import pickle


class Logger:
    def __init__(self, file):
        self.f = open(file, 'wb+')
        self.data = {}

    def log(self, d):
        self.data.update(d)

    def dump(self):
        pickle.dump(self.data, self.f)

    def close(self):
        self.f.close()


def load_data(file, key=None):
    data = []
    with open(file, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    if key:
        vals = []
        for trial in data:
            vals.append(trial[key])
        return vals
    else:
        return data