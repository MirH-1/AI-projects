import os
import sys
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

# Initialize empty lists for training and testing data
trainingData = []
testData = []

# Table for guesses and dictionary for accuracy tracking
guessTab = []
accuracy = {}

iteration = []
values = []


class neighbour:
    def __init__(self, cls, dist):
        self.cls = cls
        self.dist = dist

class item:
    def __init__(self, row):
        row = row.split(",")
        self.cls = row[-1]
        self.val = list(np.float_(row[:-1]))
        self.neighbours = []
        self.isSorted = True



    def Is(self, cls):
        return self.cls == cls

    def sortNeighbours(self):
        self.neighbours.sort(key=lambda x: x.dist)
        self.isSorted = True

    def guessClass(self, k):
        if not self.isSorted:
            self.sortNeighbours()
        l = [i.cls for i in self.neighbours[:k]]
        return max(set(l), key=l.count)

    def __str__(self):
        return ",".join([str(i) for i in self.val] + [self.cls])

    def distSqr(self, it):
        sum = 0
        for i in range(len(self.val)):
            sum += (self.val[i] - it.val[i])**2
        self.neighbours.append(neighbour(it.cls, sum))
        self.isSorted = False
        return sum

def readTo(file_path, l):
    try:
        # Open the file using 'utf-8-sig' encoding
        with open(file_path, 'r') as file:
            for line in file:
                l.append(item(line.rstrip()))
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error converting a line in {file_path}: {e}")
        sys.exit(1)

def calculateDist():
    for i in trainingData:
        for j in testData:
            j.distSqr(i)

def checkAccuracy():
    for i in testData:
        a = [str(i)]
        for j in range(1, k + 1):
            success = i.Is(i.guessClass(j))
            a.append(success)
            if success:
                accuracy[j] = accuracy.get(j, 0) + 1
        guessTab.append(a)

def printResults():
    ks = list(range(1, k + 1))
    ks.insert(0, "Item")
    acc_percent = [[i, accuracy[i] * 100 / len(testData)] for i in accuracy]
    print(acc_percent[0][1])
    plt.plot(accuracy.keys(), [i * 100 / len(testData) for i in accuracy.values()])
    plt.ylim(ymin=0)
    plt.show()


if __name__ == "__main__":
    home_dir = os.path.expanduser('~')  # Gets the path to the home directory
    downloads_dir = os.path.join(home_dir, 'Downloads')  # Appends the Downloads directory
    trainingFile = "train.data"
    testFile = "test.data"
    k = int(sys.argv[3])

    readTo(trainingFile, trainingData)
    readTo(testFile, testData)
    calculateDist()
    checkAccuracy()
    printResults()

