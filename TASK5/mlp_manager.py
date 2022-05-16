from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

import time
from data_manager import DataManager


class MLPManager:
    def __init__(self) -> None:
        self.results = []

    def step(self, data: DataManager):
        ml_claas = MLPClassifier(hidden_layer_sizes=(
            self.perceptron,)*self.layers, solver='sgd', max_iter=5000)
        trainX, trainY = data.trainX, data.trainY
        testX, testY = data.testX, data.testY

        start = time.time()
        ml_claas.fit(trainX, trainY)
        stop = time.time()

        pred = ml_claas.predict_proba(testX)
        cross_entropy_error = log_loss(testY, pred)
        cross_val_scr = mean_squared_error(testY, pred)

        y_pred = ml_claas.predict(testX)
        accuracy = accuracy_score(testY, y_pred)
        self.results.append(
            {"layers": self.layers, "perceptrons": self.perceptron, "learn_time": (stop-start),
             "cross_entropy_error": cross_entropy_error, "cross_val_score": cross_val_scr,
             "class_report": classification_report(testY, y_pred), "accuracy": accuracy_score(testY, y_pred), "mlp_obj": ml_claas
             })
        return ml_claas, accuracy

    def set_layer_size(self, size):
        self.layers = size

    def set_perceptron_number(self, perceptron):
        self.perceptron = perceptron

    def get_data(self, layers, perceptrons):
        for x in self.results:
            if x["layers"] == layers and x["perceptrons"] == perceptrons:
                return x

    def __str__(self):
        output = 'layers,perceptrons,accuracy,learn_time[s],cross_entropy_error,cross_val_score\n'
        for x in self.results:
            output += f'{x["layers"]},{x["perceptrons"]},{x["accuracy"]},{x["learn_time"]},{x["cross_entropy_error"]},{x["cross_val_score"]}\n'
        return output
