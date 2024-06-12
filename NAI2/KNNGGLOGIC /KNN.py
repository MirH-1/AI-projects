import sys
import pandas as pd
from math import sqrt
from collections import defaultdict


class Sample:
    def __init__(self, key, numbers):
        self.key = key
        self.numbers = numbers


def make_list(vector):
    str_array = vector.split(',')
    return Sample("", [float(s) for s in str_array])


def data_reading(path):
    data = []
    df = pd.read_csv(path, header=None)
    for _, row in df.iterrows():
        key = row.iloc[-1]
        numbers = row.iloc[:-1].tolist()
        data.append(Sample(key, numbers))
    return data


def distance_calc(numbers1, numbers2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(numbers1, numbers2)))


def prediction(store_answer, key, vector, sample_size):
    store_answer.sort(key=lambda x: x['distance'])
    prediction_data = defaultdict(int)
    for i in range(min(k, len(store_answer))):  # Check bounds to avoid out-of-range errors
        prediction_data[store_answer[i]['key']] += 1
    max_key = max(prediction_data, key=prediction_data.get)
    if sample_size == 1:
        return max_key
    return "true" if max_key == key else "false"


def check_accuracy(vector):
    count = 0
    test_case_data = [vector] if vector.numbers else test_data
    test_data_size = len(test_case_data)

    for i in range(test_data_size):
        store_answer = [{'key': data[j].key, 'distance': distance_calc(test_case_data[i].numbers, data[j].numbers)} for
                        j in range(len(data))]
        result = prediction(store_answer, test_case_data[i].key, vector, test_data_size)
        if test_data_size > 1 and result == "true":
            count += 1

    accuracy = (count / test_data_size) * 100 if test_data_size > 1 else result
    return f"{accuracy:.2f}%" if test_data_size > 1 else accuracy

#handle
def main(args):
    global k, data, test_data, validate_length
    k = int(args[0])
    data = data_reading(args[1])
    test_data = data_reading(args[2])
    validate_length = len(data[0].numbers)

    print("Accuracy on test data is:", check_accuracy(Sample("", [])))
    input_vector = input(f"Please enter a vector with commas (must be of length {validate_length}): ")
    if input_vector.strip() == "-1":
        return
    user_sample = make_list(input_vector)
    print("Prediction for entered vector:", check_accuracy(user_sample))


if __name__ == "__main__":
    main(sys.argv[1:])
