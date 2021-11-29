import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print("explained_variance: ", round(explained_variance, 4))
    print("r2: ", round(r2, 4))
    print("MAE: ", round(mean_absolute_error, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(np.sqrt(mse), 4))


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J


# Defining the class
class LinearRegression:
    def _init_(self, x, y, regularization="none", C=0.1):
        """
        x: x data
        y: y data
        regularization: "none", "l1", "l2"
        """
        self.data = x
        self.data = np.insert(self.data, -1, np.ones(len(x)), axis=1)
        self.label = y
        self.regularization = regularization
        self.B = np.random.random(self.data.shape[1])
        self.alpha = 0.005
        self.iter_ = 2000
        self.C = C

    def fit(self, epochs=2000, lr=0.005):
        self.iter_ = epochs
        self.alpha = lr
        cost_history = [0] * self.iter_
        m = len(self.label)

        print("Doing gradient descent")
        for iteration in tqdm(range(self.iter_)):
            # print(iteration)
            # Hypothesis Values
            h = self.data.dot(self.B)
            # Difference b/w Hypothesis and Actual Y
            loss = h - self.label
            # Gradient Calculation
            if self.regularization == "none":
                gradient = self.data.T.dot(loss)
            elif self.regularization == "l1":
                gradient = self.data.T.dot(loss) + self.C * np.sum(abs(self.B))
            elif self.regularization == "l2":
                gradient = self.data.T.dot(loss) + self.C * np.sum(np.square(self.B))
            else:
                print(f"Regularization \'{self.regularization}\' not recognised")
                return
            # Changing Values of B using Gradient
            self.B = self.B - self.alpha * gradient/m
            # New Cost Value
            cost_history[iteration] = self._current_cost()

        return cost_history

    def _current_cost(self):
        return cost_function(self.data, self.label, self.B)
        # return np.sum((self.data.dot(self.B) - self.label) ** 2)/(2 * len(self.label))

    def predict(self, inp):
        inp = np.insert(inp, -1, np.ones(len(inp)), axis=1)
        return inp.dot(self.B)