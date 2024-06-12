import sys
import pandas as pd
from math import sqrt
from collections import defaultdict


class Sample: # represent each sample with a key and list of numbers
    def __init__(self, key, numbers):
        self.key = key
        self.numbers = numbers


def make_list(vector):
    str_array = vector.split(',')
    return Sample("", [float(s) for s in str_array])


def data_reading(path): # reading a CSV file and convert it into a list of sample objects
    data = []
    df = pd.read_csv(path, header=None)
    for _, row in df.iterrows():
        key = row.iloc[-1] #extract the number to create sample objects ->
        numbers = row.iloc[:-1].tolist() # Create Sample Object and Append to List
        data.append(Sample(key, numbers))
    return data


def distance_calc(numbers1, numbers2): #calculating the distance to find the nieghbour.
    return sqrt(sum((a - b) ** 2 for a, b in zip(numbers1, numbers2)))


def prediction(store_answer, key, vector, sample_size):
    store_answer.sort(key=lambda x: x['distance'])#sort answers bby distance
    prediction_data = defaultdict(int)
    for i in range(min(k, len(store_answer))):  # Check bounds to avoid out-of-range errors
        prediction_data[store_answer[i]['key']] += 1
    max_key = max(prediction_data, key=prediction_data.get) #get the key with the maximum count
    if sample_size == 1:
        return max_key
    return "true" if max_key == key else "false" #retuen either true or false for the accuracy


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

    accuracy = (count / test_data_size) * 100 if test_data_size > 1 else result #calculate the accuracy
    return f"{accuracy:.2f}%" if test_data_size > 1 else accuracy #give accuracy in percentage


# reads the data, compute the accuracy, and handle user input.
def main(args):
    global k, data, test_data, validate_length
    k = 5
    data = data_reading(args[1])
    test_data = data_reading(args[2])
    validate_length = len(data[0].numbers)

    print("Accuracy on test data is:", check_accuracy(Sample("", [])))


    print(f"Please enter a vector with commas (must be of length {validate_length}) OR type '-1' to exit:")
    input_vector = input()
    if input_vector.strip() == "-1":
        print("Exiting the program.")
        return
    try:
        user_sample = make_list(input_vector)
        print("Prediction for entered vector:", check_accuracy(user_sample))
    except ValueError:
        print("Invalid input. Please ensure you enter the correct number of floating-point values separated by commas.")


if __name__ == "__main__":
    main(sys.argv[1:])
