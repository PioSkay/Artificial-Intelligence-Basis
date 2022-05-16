from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from data_manager import DataManager
from mlp_manager import MLPManager
from plots import *
import pandas as pd
import os

data_manager = DataManager()

print(data_manager.df['Species'].value_counts())

""" Graphs """
data_before_and_after_normalization(data_manager)
split_data(data_manager)
plot_frequency_of_all_species(data_manager)
radviz_of_sepcies(data_manager)
pair_plot_of_versicolor_virginical(data_manager)
pair_plot_of_all_species(data_manager)

data_manager.scaled_data(True)
mlp = MLPManager()
mlp.set_layer_size(3)
mlp.set_perceptron_number(100)
mlp.step(data_manager)
mlp.set_layer_size(4)
mlp.step(data_manager)
print(mlp)
#prediction, accuracy = multilayer_perceptron(trainX, testX, trainY, testY)
#confusion_matrix(prediction, trainX, trainY)