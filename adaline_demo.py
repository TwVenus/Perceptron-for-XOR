import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Readfile(object):
    def __init__(self, data_path):
        self.file = pd.read_csv(data_path, header=None)
        self.feature_list = np.array(self.file.iloc[0:100, [0, 2]].values)
        self.feature_list[:, 0] = (self.feature_list[:, 0] - self.feature_list[:, 0].mean()) / self.feature_list[:, 0].std()
        self.feature_list[:, 1] = (self.feature_list[:, 1] - self.feature_list[:, 1].mean()) / self.feature_list[:, 1].std()
        self.output_list = np.where(self.file.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)

class Adaline(object):
    def __init__(self, dataset, learning_rate=0.01, bias=1):
        self.feature_list = dataset.feature_list
        self.output_list = dataset.output_list
        self.learning_rate = learning_rate
        self.bias = bias

    def train(self):
        self.weight_list = []
        for i in range(0, self.feature_list.shape[1] + 1):
            self.weight_list.append(round(random.uniform(0.05, -0.05), 2))

        while True:
            self.error_list = []
            pass_count = 0
            cost = 0
            for i in range(0, self.feature_list.shape[0]):
                output_expect = np.sum([(float(self.feature_list[i][x]) * self.weight_list[x]) for x in range(0, self.feature_list.shape[1])]) + self.bias * self.weight_list[2]
                self.error_list.append(self.output_list[i] - output_expect)

            for error in self.error_list:
                cost += (error** 2).sum() / 2.0

            print(round(cost,2))
            if (round(cost,2) <= 3.0):
                break
            else:
                for j in range(0, self.feature_list.shape[1]):
                    update = np.sum([(float(self.feature_list[x][j]) * self.error_list[x]) for x in range(0, self.feature_list.shape[0])])
                    self.weight_list[j] = self.weight_list[j] + update * self.learning_rate
                self.weight_list[2] += np.sum(self.error_list[x] for x in range(0, self.feature_list.shape[0]))* self.learning_rate

class Diagram(object):
    def __init__(self, dataset, adaline):
        self.feature_list = dataset.feature_list
        self.adaline = adaline
        self.setup()
    def setup(self):
        self.adaline.train()
    def draw(self):
        weight = self.adaline.weight_list
        print(weight)
        plt.scatter(self.feature_list[:50, 0], self.feature_list[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(self.feature_list[50:100, 0], self.feature_list[50:100, 1], color='blue', marker='x', label='versicolor')
        plt.legend(loc='upper left')
        input_data_min, input_data_max = self.feature_list[:, 0].min() - 1, self.feature_list[:, 0].max() + 1
        l = np.linspace(input_data_min, input_data_max)
        a, b = -weight[0] / weight[1], -weight[2] / weight[1]
        plt.plot(l, a * l + b, 'b-')
        plt.show()

if __name__ == "__main__":
    dataset = Readfile("iris.txt")
    adaline = Adaline(dataset, learning_rate=0.01, bias=1)
    diagram = Diagram(dataset, adaline)
    diagram.draw()