# import json
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

    # def __init__(self, file, save_per_trial=True):
    #     self.save_per_trial = save_per_trial
    #     self.f = open(file, 'a+')
    #     self.data = [{}]
    #     self.first_trial = True
    #
    # def dump(self):
    #     if self.save_per_trial:
    #         if self.first_trial:
    #             self.f.write('[')
    #             self.first_trial = False
    #         else:
    #             self.f.write(',')
    #         json.dump(self.data[0], self.f)
    #         self.data = [{}]
    #     else:
    #         self.data.append({})
    #
    # def log(self, d):
    #     self.data[-1].update(d)
    #
    # def close(self):
    #     if self.save_per_trial:
    #         self.f.write(']')
    #     else:
    #         json.dump(self.data, self.f)
    #     self.f.close()
