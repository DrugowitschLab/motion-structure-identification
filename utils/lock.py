class Lock:
    def __init__(self):
        self.locked = False

    def lock(self, msg):
        assert not self.locked, msg
        self.locked = True

    def unlock(self):
        self.locked = False