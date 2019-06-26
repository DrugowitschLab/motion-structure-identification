from datetime import datetime


class Timer:
    def __init__(self):
        self.t_start = None

    def reset(self):
        self.t_start = None

    def restart(self):
        self.t_start = datetime.now()

    def get_seconds(self):
        now = datetime.now()
        if self.t_start is None:
            self.t_start = now
        return (now - self.t_start).total_seconds()


def timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")
