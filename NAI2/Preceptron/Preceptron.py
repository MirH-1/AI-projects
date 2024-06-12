import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, random_state=1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        self.weights = rng.random(n_features)
        self.bias = rng.random()

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)


def load_dataset(file_path):
    data = pd.read_csv(file_path, header=None)
    # Assuming the last column is the target class
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if y.dtype == object or y.dtype == 'U':
        unique_classes = np.unique(y)
        class_mapping = {cls: index for index, cls in enumerate(unique_classes)}
        y = np.array([class_mapping[cls] for cls in y], dtype=np.int64)

    return X, y


training_file_path = '/Users/mirhaidar/Downloads/perceptron.data.csv'
testing_file_path = '/Users/mirhaidar/Downloads/perceptron.test.data.csv'

X_train, y_train = load_dataset(training_file_path)
X_test, y_test = load_dataset(testing_file_path)

perceptron = Perceptron(learning_rate=0.01, n_iters=1000)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")


def classify_new_data(perceptron_model):
    while True:
        try:
            new_sample = input("\nEnter new sample features separated by comma or 'exit' to quit:\n")
            if new_sample.lower() == 'exit':
                break

            new_sample = np.array(new_sample.split(','), dtype=np.float32)

            prediction = perceptron_model.predict(new_sample.reshape(1, -1))
            print(f"Predicted class: {prediction[0]}")
        except ValueError:
            print("Could not convert input to a float. Please enter valid numbers separated by commas.")


classify_new_data(perceptron)
