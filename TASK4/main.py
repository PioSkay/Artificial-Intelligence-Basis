from pickle import TRUE
from sympy import Integer
from helpers import loadArffFromFile, preprocessKDDDataSet, split
from graphs import DrawSplit, Draw
from sklern_wrap import Type, sklearn_wrap, knn_conf, hyper_param_knn, hyper_param_logistic

model_optimalization_3 = [
    'count',
    'diff_srv_rate',
    'dst_bytes',
    'dst_host_count',
    'dst_host_diff_srv_rate',
    'dst_host_rerror_rate',
    'dst_host_same_src_port_rate',
    'dst_host_same_srv_rate',
    'dst_host_serror_rate',
    'dst_host_srv_count',
    'dst_host_srv_diff_host_rate',
    'dst_host_srv_rerror_rate',
    'dst_host_srv_serror_rate',
    'flag',
    'logged_in',
    'protocol_type',
    'rerror_rate',
    'same_srv_rate',
    'serror_rate',
    'service',
    'src_bytes',
    'srv_rerror_rate',
    'srv_serror_rate',
    'class'
]

model_optimalization_2 = [
    'dst_host_rerror_rate',
    'dst_host_srv_count',
    'dst_host_diff_srv_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_srv_rerror_rate',
    'flag',
    'logged_in',
    'protocol_type',
    'rerror_rate',
    'same_srv_rate',
    'serror_rate',
    'service',
    'src_bytes',
    'srv_rerror_rate',
    'class'
]

model_optimalization = [
    'wrong_fragment',
    'serror_rate',
    'srv_serror_rate',
    'same_srv_rate',
    'dst_host_srv_count',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'dst_host_count',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'class'
]

def manual_selection(config, with_model_opt: bool = True, use_hyper_param: bool = False):
    df = loadArffFromFile('data/KDDTrain+_20Percent.arff')
    df_test = loadArffFromFile('data/KDDTest+.arff')
    
    if with_model_opt == True:
        df = df[model_optimalization_2]
    df = preprocessKDDDataSet(df)
    x_train, y_train = split(df)
    if with_model_opt == True:
        df_test = df_test[model_optimalization_2]
    df_test = preprocessKDDDataSet(df_test)
    x_test, y_test = split(df_test)

    if config[Type.linear] == True:
        l_reg = sklearn_wrap(x_train, y_train, Type.linear)
        if use_hyper_param == True:
            hyper_param_logistic(x_train=x_train, y_train=y_train)
        acc, pred = l_reg.test(x_test, y_test)
        df_test['linear_regression_result'] = pred
        DrawSplit(df_test, filter='linear_regression_result',
                add_to_title=f"Accuracy {acc*100}%", limit=1000)

    if config[Type.logistic] == True:
        log_reg = sklearn_wrap(x_train, y_train, Type.logistic)
        if use_hyper_param == True:
            hyper_param_knn(x_train=x_train, y_train=y_train)
        acc, pred = log_reg.test(x_test, y_test)
        df_test['logistic_regression_result'] = pred
        DrawSplit(df_test, filter='linear_logistic_result',
                  add_to_title=f"Accuracy {acc*100}%", limit=1000)

    if config[Type.KNN] == True:
        knn_reg = sklearn_wrap(x_train, y_train, Type.KNN)
        acc, pred = knn_reg.test(x_test, y_test)
        df_test['KNN_result'] = pred
        DrawSplit(df_test, filter='KNN_result',
                add_to_title=f"Accuracy {acc*100}%", limit=1000)
    
    if config[Type.SVM] == True:
        svm_reg = sklearn_wrap(x_train, y_train, Type.SVM)
        acc, pred = svm_reg.test(x_test, y_test)
        df_test['SVM_result'] = pred
        DrawSplit(df_test, filter='SVM_result',
                add_to_title=f"Accuracy {acc*100}%", limit=1000)

def draw_graphs():
    df = loadArffFromFile('data/KDDTrain+.arff')
    df = preprocessKDDDataSet(df)
    for i in df.columns:
        if type(df[i][0]) != bytes:
            DrawSplit(df, upper=2000, filter=i, limit=1000)

def corelation():
    df = loadArffFromFile('data/KDDTrain+.arff')
    df = preprocessKDDDataSet(df)
    results = []
    for i in df.columns:
        results.append({'ele':i, 'val':df['class'].corr(df[i])})
    return sorted(results, key = lambda i: i['val'], reverse=True)

i = int(input('1 - Data graphs\n2 - algo\n3 - corelation\n'))
if i == 1:
    draw_graphs()
elif i == 2:
    config = {
        Type.linear: False,
        Type.logistic: False,
        Type.KNN: False,
        Type.SVM: True
    }
    manual_selection(config, True, False)
elif i == 3:
    results = corelation()
    for i in results:
        print(i['ele'], ',', i['val'])