import numpy as np
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from mlp.base import BaseModel


class MLPModel(BaseModel):
    offset = 15
    reduce = 0.50
    best_accuracy: int = 0

    def __init__(self, df: pd.DataFrame, hidden_size: int = 100, learning_rate: float = 0.001, epochs: int = 500):
        target_column = 'Race'
        x = df.drop(columns=["Race"]).values
        y = df[target_column].astype('category').cat.codes.values

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(x, y, test_size=0.2)

        self.input_size = self.train_data.shape[1]
        self.hidden_size = hidden_size
        self.output_size = len(np.unique(y))
        self.learning_rate = learning_rate
        self.epochs = epochs

        np.random.seed(int(time.time()))
        self.weights_hidden = np.random.randn(self.input_size, hidden_size) * np.sqrt(2. / self.input_size) # He initialization
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_output = np.random.randn(hidden_size, self.output_size) * np.sqrt(2. / hidden_size)
        self.bias_output = np.zeros(self.output_size)

    def _forward_propagation(self, data):
        hidden_layer_input = np.dot(data, self.weights_hidden) + self.bias_hidden
        hidden_layer_output = super()._relu_activation(hidden_layer_input, False)

        out_layer = np.dot(hidden_layer_output, self.weights_output) + self.bias_output
        predictions = super()._softmax(out_layer)
        return hidden_layer_input, hidden_layer_output, predictions

    def _backward_propagation(self, data, labels, hidden_layer_input, hidden_layer_output, predictions):
        batch_size = data.shape[0]

        loss = predictions
        loss[range(batch_size), labels] -= 1
        loss /= batch_size

        gradients_weights_hidden_output = np.dot(hidden_layer_output.T, loss)
        gradients_biases_output = np.sum(loss, axis=0)

        hidden_error = np.dot(loss, self.weights_output.T)
        hidden_error *= super()._relu_activation(hidden_layer_input, True)

        gradients_weights_input_hidden = np.dot(data.T, hidden_error)
        gradients_biases_hidden = np.sum(hidden_error, axis=0)

        self.weights_hidden -= self.learning_rate * gradients_weights_input_hidden
        self.bias_hidden -= self.learning_rate * gradients_biases_hidden
        self.weights_output -= self.learning_rate * gradients_weights_hidden_output
        self.bias_output -= self.learning_rate * gradients_biases_output

    def train(self, batch_size: int = 64):
        counter = 0

        for epoch in range(self.epochs):
            self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)

            for i in range(0, self.train_data.shape[0], batch_size):
                data_batch = self.train_data[i:i + batch_size]
                labels_batch = self.train_labels[i:i + batch_size]

                hidden_layer_input, hidden_layer_output, output_predictions = self._forward_propagation(data_batch)
                self._backward_propagation(data_batch, labels_batch, hidden_layer_input, hidden_layer_output,
                                           output_predictions)

            predictions = self._forward_propagation(self.test_data)[-1]
            accuracy = super()._accuracy(predictions, self.test_labels)
            print(f'Epoch {epoch + 1}|{self.epochs}, Accuracy: {accuracy * 100:.2f}%')

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                counter = 0
            else:
                counter += 1
                if counter == self.offset:
                    self.learning_rate *= self.reduce
                    print(f"Learning rate reduced to: {self.learning_rate}")
                    counter = 0
                    if self.learning_rate < 1e-100:
                        print(f"Learning rate too low. Stopping...")
                        break

        predictions = self._forward_propagation(self.test_data)[-1]
        return super()._accuracy(predictions, self.test_labels)
