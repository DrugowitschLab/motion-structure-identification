import pickle
import csv


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


def dat2csv(file, keys):
    data = load_data(file)
    filename = file[:-4]
    with open(filename + '.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for trial in data:
            writer.writerow({key: trial[key] for key in keys})


class SimLogger:
    def __init__(self):
        self.data = {}

    def reset(self, φ, r):
        self.data = dict(  # Store stimulus history (unless INFRUN)
            t=[0],         # time points of frames (in sim time)
            φ=[φ],     # Angular locations and velocities for all N dots
            r=[r],         # Radial locations and velocities for all N dots
        )

    def log(self, t, φ, r):
        self.data['t'].append(t)
        self.data['φ'].append(φ)
        self.data['r'].append(r)

    def get(self, n):
        return self.data['φ'][-1][:n], self.data['r'][-1][:n]


if __name__ == '__main__':
    file = '../data/sichao_20190726135416.dat'
    dat2csv(file, ['answer', 'choice', 'confidence'])


