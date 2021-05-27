from sklearn.utils import shuffle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def cross_validation(X, y, model, folds):
    X_folds = []
    y_folds = []
    np.random.seed(0)
    X, y = shuffle(X, y)
    fold_size = round(X.shape[0] / folds)
    #splitting into folds
    for i in range(folds):
        X_folds.append(X[i * fold_size: (i + 1) * fold_size])
        y_folds.append(y[i * fold_size: (i + 1) * fold_size])

    average_score_val, average_score_train = 0, 0
    for out_fold_index in range(folds):
        # Create train/test
        X_test = X_folds[out_fold_index]
        y_test = y_folds[out_fold_index]
        X_train, y_train = [], []
        for inner_index in range(folds):
            if inner_index != out_fold_index:
                X_train.extend(X_folds[inner_index])
                y_train.extend(y_folds[inner_index])
        # Fit and predict
        model.fit(X_train, y_train)
        predicted_train, predicted_test = model.predict(X_train), model.predict(X_test)
        # Score
        score_val = sum([y_test[i] != predicted_test[i] for i in range(len(y_test))])/len(y_test)
        average_score_val += score_val
        score_train = sum([y_train[i] != predicted_train[i] for i in range(len(y_train))])/len(y_train)
        average_score_train += score_train
    average_score_val /= folds
    average_score_train /= folds
    return average_score_train, average_score_val
  
  


def logistic_regression_results(X_train, y_train, X_test, y_test):
    lam = [1e-4, 1e-2, 1, 100, 10**4]
    avg_train_list, avg_val_list, test_error = [], [], []
    for l in lam:
        model = LogisticRegression(C= 1/l, max_iter=2000, class_weight=None)
        #cross validation with lam
        average_score_train, average_score_val = cross_validation(X_train, y_train, model, 5)
        avg_train_list.append(average_score_train)
        avg_val_list.append(average_score_val)

        #calculate test error
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        error_test = sum([y_test[i] != predict[i] for i in range(len(y_test))]) / len(y_test)
        test_error.append(error_test)

    res = {}
    for  tr, v, ts, l in zip(avg_train_list, avg_val_list, test_error, lam):
        res['logistic_regression_lambda_' + str(l)] = (tr, v, ts)
    return res

def main():
    iris_data = load_iris()
    X, y = iris_data['data'], iris_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    errors = logistic_regression_results(X_train, y_train, X_test, y_test)
    lam = [1e-4, 1e-2, 1, 100, 10 ** 4]

    # plot error
    fig, ax = plt.subplots()
    x = np.arange(len(lam))
    width = 0.3
    values =  list(errors.values())
    ax.bar(x - width / 3, [val[0] for val in values], width, color='b', label='train error')
    ax.bar(x + width / 3,  [val[1] for val in values], width, color='g', label='validation error')
    ax.bar(x + width * (2/3),  [val[2] for val in values], width, color='r', label='test error')

    ax.set_xticks(x)
    ax.set_xticklabels(lam)
    ax.set_ylabel('Errors')
    ax.set_xlabel('lambda')
    ax.set_title("Errors of logistic regression CV by lam")
    ax.legend()
    fig.tight_layout()
    plt.show()
