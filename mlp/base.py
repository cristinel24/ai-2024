import numpy as np


class BaseModel:
    @staticmethod
    def _softmax(input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _relu_activation(input, derivative=False):
        if derivative:
            return np.where(input > 0, 1, 0)
        return np.maximum(0, input)

    @staticmethod
    def _cross_entropy_loss(predictions, targets):
        size = predictions.shape[0]
        result = -np.log(predictions[range(size), targets])
        loss = np.sum(result) / size
        return loss

    @staticmethod
    def _accuracy(predictions, labels):
        return np.mean(np.argmax(predictions, axis=1) == labels)
