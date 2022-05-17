from data_manager import DataManager
from mlp_manager import MLPManager
from plots import *
import pandas as pd
import os

def graphs():
    """ Graphs """
    data_before_and_after_normalization(data_manager)
    split_data(data_manager)
    plot_frequency_of_all_species(data_manager)
    radviz_of_sepcies(data_manager)
    pair_plot_of_versicolor_virginical(data_manager)
    pair_plot_of_all_species(data_manager)

def output_analysis():
    pereptrons = [10, 15, 50, 100, 150]
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for pereptron in pereptrons:
        for layer in layers:
            print(f"pereptron: {pereptron}, layer: {layer}")
            mlp.set_layer_size(layer)
            mlp.set_perceptron_number(pereptron)
            mlp.step(data_manager)
            print(mlp)
    print(mlp)

data_manager = DataManager()
#graphs()
data_manager.scaled_data(True)
mlp = MLPManager()

mlp.set_layer_size(10)
mlp.set_perceptron_number(100)
pred = mlp.step(data_manager)
print(mlp)
output_analysis()
#print(mlp.get_data(10, 100)["class_report"])
#pred = mlp.step(data_manager, True)
#confusion_matrix(pred, True), data_manager.testX, data_manager.testY)
