# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from collections import defaultdict
import seaborn as sns

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
#from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler

from utils import plot_pr, plot_roc, plot_confusion_matrix, GENRE_LIST

from fft import read_fft

genre_list = GENRE_LIST


def train_model2(clf_factory, X, Y, name, plot=False):
    labels = np.unique(Y)
    clf=clf_factory()
    cv = ShuffleSplit(n_splits=len(X), test_size=0.3, random_state=0)
    scoring = ['precision_macro', 'recall_macro']
    #scores=cross_val_score(clf_factory, X, y, cv=cv)
    scores = cross_validate(clf, X, Y, scoring=scoring,
                            cv=5, return_train_score=True)
    print(scores)
    return scores
                      




def train_model(clf_factory, X, Y, name, plot=False):
    labels = np.unique(Y)
    #try scaling our data
    scaler = StandardScaler()
    X= scaler.fit_transform(X)

    cv = ShuffleSplit(n_splits=len(X), test_size=0.3, random_state=0)
    #Should this be StartifiedShuffleSplit?
    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(
        list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = []  # just to later get the median

    cms = []

    for train, test in cv.split(X):
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = clf_factory()
        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

        for label in labels:
            y_label_test = np.asarray(y_test == label, dtype=int)
            proba = clf.predict_proba(X_test)
            proba_label = proba[:, label]

            precision, recall, pr_thresholds = precision_recall_curve(
                y_label_test, proba_label)
            pr_scores[label].append(auc(recall, precision))
            precisions[label].append(precision)
            recalls[label].append(recall)
            thresholds[label].append(pr_thresholds)

            fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
            roc_scores[label].append(auc(fpr, tpr))
            tprs[label].append(tpr)
            fprs[label].append(fpr)

    if plot:
        for label in labels:
            print("Plotting %s" % genre_list[label])
            scores_to_sort = roc_scores[label]
            median = np.argsort(scores_to_sort)[int(len(scores_to_sort) / 2)]

            desc = "%s %s" % (name, genre_list[label])
            plot_pr(pr_scores[label][median], desc, precisions[label][median],
                    recalls[label][median], label='%s vs rest' % genre_list[label])
            plot_roc(roc_scores[label][median], desc, tprs[label][median],
                     fprs[label][median], label='%s vs rest' % genre_list[label])

    all_pr_scores = np.array(list(pr_scores.values())).flatten()
    summary = (np.mean(scores), np.std(scores),
               np.mean(all_pr_scores), np.std(all_pr_scores))
    print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


def create_model():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(n_jobs=8, max_iter=50)
    return clf

def create_model2():
    from sklearn.linear_model import LogisticRegressionCV
    clf = LogisticRegressionCV(max_iter=1000, multi_class='multinomial',n_jobs=3)
    return clf


if __name__ == "__main__":
    X, y = read_fft(genre_list)
    if 1:
        train_avg, test_avg, cms = train_model(
            create_model, X, y, "Log Reg FFT", plot=True)

        cm_avg = np.mean(cms, axis=0)
        cm_norm = cm_avg / np.sum(cm_avg, axis=0)

        plot_confusion_matrix(cm_norm, genre_list, "fft",
                            "Confusion matrix of an FFT based classifier")
    if 0:
        scores = train_model2(
            create_model2, X, y, "Log Reg FFT", plot=False)

        