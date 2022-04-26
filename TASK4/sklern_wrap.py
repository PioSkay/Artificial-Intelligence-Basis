from functools import cache
from tabnanny import verbose
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from enum import Enum

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

class Type(Enum):
    linear = 0,
    logistic = 1,
    KNN = 2,
    SVM = 3

class knn_conf:
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(knn_conf, self).__new__(self)
            self.n_neighbors = 10
            self.leaf_size = 30
            self.p = 1
        return self.instance
    def __str__(self):
        return f'n_neighbors:{self.n_neighbors}, leaf_size:{self.leaf_size}, p:{self.p}'

class logistic_conf:
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(logistic_conf, self).__new__(self)
            self.solver = "newton-cg"
            self.C = 0.1
            self.penalty = "l2"
        return self.instance
    def __str__(self):
        return f'solver:{self.solver}, C:{self.C}, p:{self.penalty}'

class sklearn_wrap:
    def __init__(self, x_train, y_train, type: Type = Type.linear) -> None:
        """
        Parameters
        ----------
        Path the the file 
        """
        if type == Type.linear:
            self.linear = linear_model.LinearRegression()
        elif type == Type.logistic:
            self.linear = linear_model.LogisticRegression(solver=logistic_conf().solver, 
                                                          C=logistic_conf().C,
                                                          penalty=logistic_conf().penalty)
        elif type == Type.KNN:
            self.linear = KNeighborsClassifier(n_neighbors=knn_conf().n_neighbors,
                                               p=knn_conf().p,
                                               leaf_size=knn_conf().leaf_size,
                                               n_jobs=20)
        elif type == Type.SVM:
            self.linear = SVC(kernel='linear', probability=True, verbose=True, cache_size=10000)
        self.linear.fit(x_train, y_train)
    def test(self, x_test, y_test):
        """
        Return
        ----------
        Accuracy of the results
        """
        acc = self.linear.score(x_test, y_test)
        pred = self.linear.predict(x_test)
        return acc, pred

def hyper_common(hyperparam, x_train, y_train, estimator):
    clf = GridSearchCV(estimator, hyperparam, cv=2)
    return clf.fit(x_train, y_train)

def hyper_param_knn(x_train, y_train):
    leaf_size = list([30])
    n_neighbors = list([10,11,12])
    p=[1,2]
    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    #Create new KNN object
    knn_2 = KNeighborsClassifier(n_jobs=20)
    best_model = hyper_common(hyperparameters, x_train, y_train, knn_2)

    print("-------------------------------------------")
    print(best_model.best_estimator_.get_params())
    print("-------------------------------------------")

def hyper_param_logistic(x_train, y_train):
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    logistic = linear_model.LogisticRegression(n_jobs=20)
    hyperparameters = dict(solver=solvers, penalty=penalty, C=c_values)
    best_model = hyper_common(hyperparameters, x_train, y_train, logistic)
    print("-------------------------------------------")
    print(best_model.best_estimator_.get_params())
    print("-------------------------------------------")