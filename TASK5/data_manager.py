from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import os

class DataManager:
    def __init__(self) -> None:
        self.load_data()
        self.init_data()

    def load_data(self):
        __dir__ = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        self.df = pd.read_csv(__dir__ + '\Iris.csv')

    def init_data(self):
        x = self.df.drop('Species', axis=1).drop('Id', axis=1)
        y = self.df['Species']
        self.trainX, self.testX, self.trainY, self.testY \
            = train_test_split(x, y, test_size=0.15)

    def get_data_X(self):
        """Return non scaled data X values"""
        return self.trainX, self.testX

    def scaled_data(self, update: bool = False):
        """Return scaled data X values"""
        ss = StandardScaler()
        scaler = ss.fit(self.trainX)
        trainX_transpose = scaler.transform(self.trainX)
        testX_transpose = scaler.transform(self.testX)
        trainX, testX = pd.DataFrame(trainX_transpose, index=self.trainX.index, columns=self.trainX.columns), \
            pd.DataFrame(testX_transpose, index=self.testX.index,
                         columns=self.testX.columns)
        if update == True:
            self.trainX, self.testX = trainX, testX
        else:
            return trainX, testX

    def get_data_Y(self):
        """Return data Y values"""
        return self.trainY, self.testY
