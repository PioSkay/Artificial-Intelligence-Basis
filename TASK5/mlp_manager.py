from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

import time
from data_manager import DataManager


class MLPManager:
    def __init__(self) -> None:
        self.results = []

    def hyper_parameter_tuning(self, data: DataManager):
        ml_claas = MLPClassifier(hidden_layer_sizes=(
            self.perceptron,)*self.layers, solver='sgd', max_iter=5000)
        param_grid = {
            'hidden_layer_sizes': (self.perceptron,)*self.layers,
            'max_iter': [50, 100, 150, 2000, 5000],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
        grid = GridSearchCV(ml_claas, param_grid, n_jobs= -1, cv=5)
        ml_claas.fit(data.trainX, data.trainY)
        print(grid.best_params_) 
        return grid.best_params_

    def step(self, data: DataManager, just_values: bool = False):
        """ Multi layer perceptron """
        ml_claas = MLPClassifier(hidden_layer_sizes=(
            self.perceptron,)*self.layers, solver='sgd', max_iter=5000)
        trainX, trainY = data.trainX, data.trainY
        testX, testY = data.testX, data.testY
        
        if just_values == False:
            """ Change the output from string to numbers """
            encoder = LabelEncoder()
            trainY = encoder.fit_transform(trainY)
            testY = encoder.fit_transform(testY)

        """ Learning procedure with timing """
        start = time.time()
        ml_claas.fit(trainX, trainY)
        stop = time.time()

        if just_values == False:
            """ Evaluating cost functions """
            pred = ml_claas.predict_proba(testX)
            cross_entropy_error_ = log_loss(testY, pred)

            y_pred = ml_claas.predict(testX)
            accuracy = accuracy_score(testY, y_pred)
            mean_squared_error_ = mean_squared_error(testY, y_pred)

            self.results.append(
                {"layers": self.layers, "perceptrons": self.perceptron, "learn_time": (stop-start),
                "cross_entropy_error": cross_entropy_error_, "mean_squared_error": mean_squared_error_,
                "class_report": classification_report(testY, y_pred), "accuracy": accuracy_score(testY, y_pred), "mlp_obj": ml_claas
                })
        return ml_claas

    def set_layer_size(self, size):
        self.layers = size

    def set_perceptron_number(self, perceptron):
        self.perceptron = perceptron

    def get_data(self, layers, perceptrons):
        for x in self.results:
            if x["layers"] == layers and x["perceptrons"] == perceptrons:
                return x

    def __str__(self):
        output = 'layers,perceptrons,accuracy,learn_time[s],cross_entropy_error,mean_squared_error\n'
        for x in self.results:
            output += f'{x["layers"]},{x["perceptrons"]},{x["accuracy"]},{x["learn_time"]},{x["cross_entropy_error"]},{x["mean_squared_error"]}\n'
        return output
