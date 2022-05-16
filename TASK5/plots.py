import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from data_manager import DataManager
from pandas.plotting import radviz
''' Date analysis '''

def histogram_of_frequency_to_file(data, column):
    sns.FacetGrid(data, hue="Species", size=5) \
        .map(sns.distplot,column, hist_kws={"alpha":.2}) \
            .add_legend()
    plt.savefig(fname="TASK5/plots/" + __name__ + "_" + column + ".png", figsize=[10, 10])

def data_before_and_after_normalization(data: DataManager):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
    data, scaled_data = data.trainX, data.scaled_data()[0]
    ax1.set_title('Before Scaling')
    sns.kdeplot(data['SepalLengthCm'], ax=ax1)
    sns.kdeplot(data['SepalWidthCm'], ax=ax1)
    sns.kdeplot(data['PetalLengthCm'], ax=ax1)
    sns.kdeplot(data['PetalWidthCm'], ax=ax1)
    ax2.set_title('After Scaling')
    sns.kdeplot(scaled_data['SepalLengthCm'], ax=ax2)
    sns.kdeplot(scaled_data['SepalWidthCm'], ax=ax2)
    sns.kdeplot(scaled_data['PetalLengthCm'], ax=ax2)
    sns.kdeplot(scaled_data['PetalWidthCm'], ax=ax2)
    plt.savefig(fname="TASK5/plots/" + __name__ + "_scaling" + ".png", figsize=[10, 10])

def split_data(data: DataManager):
    data.df.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6), showfliers=False)
    plt.savefig(fname="TASK5/plots/" + __name__ + "_split_data" + ".png", figsize=[10, 10])

def plot_frequency_of_all_species(data: DataManager):
    for x in data.df.columns.values:
        if x != 'Id' and x != 'Species':
            histogram_of_frequency_to_file(data.df, x)

def radviz_of_sepcies(data):
    plt.clf()
    radviz(data.df.drop("Id", axis=1), "Species", color=["#3480eb", "#e69a17", "#3bb864"])
    plt.savefig(fname="TASK5/plots/" + __name__ + "_radviz_of_sepcies" + ".png", figsize=[10, 10])

def pair_plot_of_versicolor_virginical(data):
    plt.clf()
    sns.pairplot(data.df.drop("Id", axis=1)[data.df['Species'] != 'Iris-setosa'], hue="Species", size=3, diag_kind="kde")
    plt.savefig(fname="TASK5/plots/" + __name__ + "_pair_plot_of_versicolor_virginical" + ".png", figsize=[10, 10])

def pair_plot_of_all_species(data):
    plt.clf()
    sns.pairplot(data.df.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
    plt.savefig(fname="TASK5/plots/" + __name__ + "_pair_plot_of_all_species" + ".png", figsize=[10, 10])

''' Output analysis '''

def confusion_matrix(prediction, testX_scaled, testY):
    fig = plot_confusion_matrix(prediction, testX_scaled, testY, display_labels=prediction.classes_)
    fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
    plt.savefig(fname="TASK5/plots/" + __name__ + "_confusion_matrix" + ".png", figsize=[10, 10])

