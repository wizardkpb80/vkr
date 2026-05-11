class ModelManager:
    def __init__(self):
        self.models = {}
        self.loaded_model = None
        self.loaded_model_id = None

    def fit(self, X, y, config):
        pass

    def load(self, model_id):
        pass

    def unload(self):
        pass

    def predict(self, X):
        pass