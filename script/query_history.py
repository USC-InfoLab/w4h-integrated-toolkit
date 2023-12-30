class query_history:
    def __init__(self):
        self.data = None

    def set(self, data):
        self.data = data

    def get(self):
        return self.data