import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

#Defining the class
class LinearRegression:
    def __init__(self, x , y):
        self.data = x
        self.data = np.insert(self.data, -1, np.ones(len(x)), axis=1)
        self.label = y

        self.B = np.random.random(self.data.shape[1])
        self.alpha = 0.005
        self.iter_ = 2000

    def fit(self, epochs=2000, lr=0.005):
        self.iter_ = epochs
        self.alpha = lr
        cost_history = [0] * self.iter_
        m = len(self.label)

        print("Doing gradient descent")
        for iteration in tqdm(range(self.iter_)):
            #print(iteration)
            # Hypothesis Values
            h = self.data.dot(self.B)
            # Difference b/w Hypothesis and Actual Y
            loss = h - self.label
            # Gradient Calculation
            gradient = self.data.T.dot(loss) / m
            # Changing Values of B using Gradient
            self.B = self.B - self.alpha * gradient
            # New Cost Value
            cost_history[iteration] = self._current_cost()

        return cost_history

    def _current_cost(self):
        return cost_function(self.data, self.label, self.B)
        # return np.sum((self.data.dot(self.B) - self.label) ** 2)/(2 * len(self.label))


    def predict(self , inp):
        inp = np.insert(inp, -1, np.ones(len(inp)), axis=1)
        return inp.dot(self.B)     
