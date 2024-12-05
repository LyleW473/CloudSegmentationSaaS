class InvalidQueryException(Exception):
    def __init__(self, message):
        super().__init__(message)

class DataNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelLoadingException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelInferenceException(Exception):
    def __init__(self, message):
        super().__init__(message)

class PredictionsCombiningException(Exception):
    def __init__(self, message):
        super().__init__(message)