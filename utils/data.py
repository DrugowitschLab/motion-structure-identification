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


def modify(old_file, new_file, key=None, val_mapper={}, key_mapper={}):
    with open(old_file, 'rb') as fin, open(new_file, 'wb') as fout:
        while True:
            try:
                trial = pickle.load(fin)
                if key is not None:
                    trial[key] = val_mapper[trial[key]]
                for old_key in key_mapper:
                    trial[key_mapper[old_key]] = trial.pop(old_key)
                pickle.dump(trial, fout)
            except EOFError:
                break


class SimLogger:
    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = dict(  # Store stimulus history
            t=[],          # time points of frames (in sim time)
            φ=[],          # Angular locations and velocities for all dots
            r=[],          # Radial locations and velocities for all dots
        )

    def log(self, t, φ, r):
        self.data['t'].append(t)
        self.data['φ'].append(φ)
        self.data['r'].append(r)


if __name__ == '__main__':
    modify('../data/exp2/sichao_0802.old.dat', '../data/exp2/sichao_0802.dat', key='answer', val_mapper={
        'CS0': '0.00',
        'CS1': '0.20',
        'CS2': '0.35',
        'CS3': '0.55',
        'CS4': '0.75',
    }, key_mapper={
        'answer': 'ground_truth',
        # 'phi': 'φ'
    })
    print(load_data('../data/exp2/sichao_0802.dat')[0].keys())

