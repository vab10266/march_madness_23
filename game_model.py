import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pickle

def column_selector(df, cols):
    x_cols = []
    for col in cols:
        x_cols.append(df.columns[col])
    for col in cols:
        x_cols.append(df.columns[20+col])

    X = df[x_cols]
    y = df['RESULT']
    return X, y

def k_folds(df, cols, model_class, *args):
    years = df.YEAR.unique()
    validation_year = years[0]
    folds_years = years[1:]
    # print(validation_year)
    fold_results = []

    for i in range(len(years) - 1):
        test_year = folds_years[i]
        training_years = np.concatenate((folds_years[:i], folds_years[i+1:]))
        
        train_df = df[df.YEAR.isin(training_years)]
        test_df = df[df.YEAR == test_year]
        
        train_X, train_y = column_selector(train_df, cols)

        clf = model_class(*args)
        # model.fit(train_X, train_y)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_X, train_y)
        
        test_X, test_y = column_selector(test_df, cols)
        # score = clf.score(test_X, test_y)
        preds = clf.predict(test_X)
        acc = accuracy_score(test_y, preds)
        print(test_year, acc)
        fold_results.append(acc)

    full_df = df[df.YEAR.isin(folds_years)]
    full_X, full_y = column_selector(full_df, cols)
    _clf = model_class(*args)
    _clf.fit(full_X, full_y)
    return np.mean(fold_results), _clf
        


if __name__ == '__main__':

    df = pd.read_csv('C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\training_data.csv')
    print(df.columns)
    best = [1, 3, 4, 11]#[1, 6, 17, 10]
    print(df.columns[best])
    result_dict = {}
    # for i in range(1):
    #     i=3
    #     print(i,  (-i + 9.5)**4 - 0.0625)
    #     for j in range(int( (-i + 9.5)**4 - 0.0625)):
    #         cols_list = list(range(20))
    #         selected_cols = random.sample(cols_list, k = i+1)
    #         result, model = k_folds(df, selected_cols, GaussianNB)
    #         result_dict[str(selected_cols)] = result
    # best = max(result_dict, key=result_dict.get)
    # print(best, result_dict[best])

    classifiers = [
        KNeighborsClassifier,
        SVC,
        DecisionTreeClassifier,
        RandomForestClassifier,
        MLPClassifier,
        AdaBoostClassifier,
        GaussianNB,
        QuadraticDiscriminantAnalysis,
    ]
    for i in range(len(classifiers)):
    # i=6
        result, model = k_folds(df, best, classifiers[i])
        print(result)
        result_dict[str(i)] = result
    best = max(result_dict, key=result_dict.get)
    print(best, result_dict[best])

    # save the model to disk
    filename = 'C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\game_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # selected_cols = list(range(20))#[0, 3, 4, 5]
    # print(df.columns[selected_cols])
    # results, model = k_folds(df, selected_cols, RandomForestClassifier)
    # print('='*150)
    # print(results)
    # def perform(fun, **kwargs):
    #     fun(**kwargs)

    # def action1(a=0):
    #     # something
    #     print(a)

    # def action2(a):
    #     # something
    #     print(a)

    # def action3(a, b):
    #     # something
    #     print(a + b)

    # perform(action1)
    # perform(action2, **{'a':1})
    # perform(action3, **{'a':1, 'b':2})