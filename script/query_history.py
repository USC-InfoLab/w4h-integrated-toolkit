
class query_history:
    def __init__(self,session):
        self.data = {}
        key_list = list(session.keys())
        for key in key_list:
            self.data[key] = session.get(key)

    def set(self, key,value):
        self.data[key] = value

    def get(self,key):
        return self.data[key]

    def setSession(self,session):
        key_list = list(self.data.keys())
        for key in key_list:
            session[key] = self.data.get(key)
        return session
