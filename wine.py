#!/usr/bin/env

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def load(path, info=False):
    """
    Loads the cvs files
    :param path:  path to csv destination
    :param info: if True, prints shape and red/white count
    :return: concat of red, white with added colour param
    """
    # Load dataset
    path_red = path+'winequality-red.csv'
    path_white = path+'winequality-white.csv'
    ds_red = pd.read_csv(path_red, sep=';')
    ds_white = pd.read_csv(path_white, sep=';')

    # Add colour att for red/white comparisons
    ds_red["colour"] = "red"
    ds_white["colour"] = "white"
    ds_both = pd.concat([ds_red, ds_white], axis=0)
    if info:
        print(ds_both.groupby('colour').size())
        print("\nShape: ", ds_both.shape)
    return ds_both


def data_analysis(data=None, x=None, y=None):
    """
    Data analysis. Scatter plot if x and y valid (red/blue = red/white wine), else will show a correlation heatmap
    :param data: data from load()
    :param x: x_axis for scatter
    :param y: y axis for scatter
    """
    if data is None:
        return
    elif x is None:
        # calculate the correlation matrix and show heatmap
        corr = data.corr()
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="RdBu_r", center=0)
    elif x in list(data) and y in list(data):
        # Plot scatter
        x_feat = x
        y_feat = y
        colours = {'red': 'red', 'white': 'blue'}
        plt.scatter(data[x_feat], data[y_feat], c=data['colour'].apply(lambda x: colours[x]), s=5)
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
    else:
        print("***\nError: '{}' or '{}' is not a valid column title\nValid: {}\n***".format(x, y, list(data)))
        return
    plt.show()


def validation_set(data):
    """
    Split validation set
    :param data: pd data
    :return:  X_train, X_validation, Y_train, Y_validation
    """
    # Get validation set
    array = data.values
    X = array[:, 0:12]
    Y = array[:, 12]
    validation_size = 0.2
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=123)


def compare_algos(x_t, y_t, folds):
    """
    This will compare various common algorithms, report on the, and return the best one.
    :param x_t: x_training
    :param y_t: y_training
    :param folds: Number of folds in kfold
    :return: best performing model (based on accuracy only)
    """
    n_folds = folds
    seed = 123
    scoring = 'accuracy'
    models = []

    # Select models
    models.append(('SVM ', SVC(gamma='auto')))  # Slow
    models.append(('LogR', LogisticRegression(solver='lbfgs', max_iter=1000)))
    models.append(('KNN ', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('LinD', LinearDiscriminantAnalysis()))
    models.append(('GaNB', GaussianNB()))

    # Run through. Report: mean results, std, time
    print("...Training (%s fold)..." % folds)
    names = []
    kfold = model_selection.KFold(n_folds, random_state=seed)
    start_time = time.time()
    best_mod = ('ERROR', None)
    best_res = 0
    for i, (name, model) in enumerate(models):
        result = model_selection.cross_val_score(model, x_t, y_t, cv=kfold, scoring=scoring)
        names.append(name)
        print("%s: %f  STD: %f  TIME: %.1fs" % (name, result.mean(), result.std(), time.time()-start_time))
        start_time = time.time()
        # Update best model (based on acc only)
        if i == 0:
            best_mod = i
            best_res = result.mean()
        elif best_res < result.mean():
            best_mod = i
            best_res = result.mean()
    print("...Finished: Best model was %s..\n" % models[best_mod][0])
    return models[best_mod]


def fit_and_predict(model, x_t, y_t, x_v, y_v):
    """
    This will fit the best model, and predict on the validation set
    :param model: best model from compare_algos()
    :param x_t: x_training
    :param y_t: y_training
    :param x_v: x_validation
    :param y_v: y_validtion
    """
    print("...Predicting with %s..." % model[0])
    model[1].fit(x_t, y_t)
    predictions = model[1].predict(x_v)
    print("Acc: %s" % accuracy_score(y_v, predictions))
    print(confusion_matrix(y_v, predictions))
    print(classification_report(y_v, predictions))


if __name__ == '__main__':

    # Load data
    root_path = 'C:/Users/Nick/PycharmProjects/WineClassifier/dataset/'  # root path to red/white cvs
    ds_both = load(root_path, False)

    # Get Validation split
    X_train, X_validation, Y_train, Y_validation = validation_set(ds_both)
    #  print(collections.Counter(Y_validation))  # Uncomment to get valid set counts of red/white

    # Data analysis
    x_axis = 'residual sugar'
    y_axis = "pH"
    data_analysis()  # Call with ds_both for heatmap, add x_axis and y_axis for scatter

    # Compare algorithms (with k-fold)
    best_model = compare_algos(X_train, Y_train, 10)

    # Predict and report on validation
    fit_and_predict(best_model, X_train, Y_train, X_validation, Y_validation)
