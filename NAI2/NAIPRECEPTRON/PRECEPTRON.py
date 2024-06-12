import random
import pandas as pd





def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            dataset.append([float(x) if i < len(row) - 1 else row[-1] for i, x in enumerate(row)])
    return dataset


def encode_labels(dataset):
    labels = list(set(row[-1] for row in dataset))
    label_map = {label: idx for idx, label in enumerate(labels)}
    for row in dataset:
        row[-1] = label_map[row[-1]]
    return dataset, label_map


def initialize_weights(size):
    return [random.uniform(0, 1) for _ in range(size)]


def dot_product(vector1, vector2):
    return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))


def predict(row, weights, bias):
    return 1 if dot_product(row[:-1], weights) + bias >= 0 else 0


def train_perceptron(train_data, learning_rate, iterations):
    weights = initialize_weights(len(train_data[0]) - 1)
    bias = random.uniform(0, 1)

    for iteration in range(iterations):
        for row in train_data:
            prediction = predict(row, weights, bias)
            error = row[-1] - prediction
            for i in range(len(weights)):
                weights[i] += learning_rate * error * row[i]
            bias += learning_rate * error
    return weights, bias


def evaluate_model(test_data, weights, bias):
    correct = 0
    for row in test_data:
        prediction = predict(row, weights, bias)
        if prediction == row[-1]:
            correct += 1
    return correct / len(test_data) * 100


def main():
    train_file_path = input("enter the training file path: ")
    test_file_path = input("enter the test file path: ")
    learning_rate = float(input("enter the learning rate: "))
    iterations = int(input("enter the number of iterations: "))

    train_data = load_dataset(train_file_path)
    test_data = load_dataset(test_file_path)

    train_data, label_map = encode_labels(train_data)
    test_data, _ = encode_labels(test_data)

    weights, bias = train_perceptron(train_data, learning_rate, iterations)
    accuracy = evaluate_model(test_data, weights, bias)

    print(f"Accuracy: {accuracy}%")


if __name__ == "__main__":
    main()
