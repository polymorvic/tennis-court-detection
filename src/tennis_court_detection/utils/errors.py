

class NotArrayError(Exception):
    def __init__(self, message="Please provide a NumPy array to convert it to NumpyImage"):
        super().__init__(message)