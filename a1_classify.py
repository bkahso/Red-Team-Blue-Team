from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import csv
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.sum(np.diag(C))/np.sum(C)


def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    r = []
    for i in range(4):
        r.append(C[i, i]/np.sum(C[i, :]))
    return r


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    p = []
    for j in range(4):
        p.append(C[j, j]/np.sum(C[:, j]))
    return p


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2
0-
    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    file = np.load(filename)
    data = file['arr_0']
    y = data[:, -1:]
    x = data[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    accuracies = []
    recalls = []
    precisions = []
    cMatrices = []

    #Train linear SVM, store confusion matrix, accuracy, recall and precision
    linear_svm = LinearSVC(max_iter = 10000)
    linear_svm.fit(X_train, y_train.ravel())
    linear_pred = linear_svm.predict(X_test)
    linearC = confusion_matrix(y_test, linear_pred)
    cMatrices.append(linearC)
    accuracies.append(accuracy(linearC)), recalls.append(recall(linearC)), precisions.append(precision(linearC))

    #Train SVM with RBF, store confusion matrix, accuracy, recall and precision
    radial_svm = SVC(gamma=2)
    radial_svm.fit(X_train, y_train.ravel())
    radial_pred = radial_svm.predict(X_test)
    radialC = confusion_matrix(y_test, radial_pred)
    cMatrices.append(radialC)
    accuracies.append(accuracy(radialC)), recalls.append(recall(radialC)), precisions.append(precision(radialC))

    #Train Random Forest, store confusion matrix, accuracy, recall and precision
    forest = RandomForestClassifier(n_estimators=10, max_depth=5)
    forest.fit(X_train, y_train.ravel())
    forest_pred = forest.predict(X_test)
    forestC = confusion_matrix(y_test, forest_pred)
    cMatrices.append(forestC)
    accuracies.append(accuracy(forestC)), recalls.append(recall(forestC)), precisions.append(precision(forestC))

    #Train MLP, store confusion matrix, accuracy, recall and precision
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X_train, y_train.ravel())
    mlp_pred = mlp.predict(X_test)
    mlpC = confusion_matrix(y_test, mlp_pred)
    cMatrices.append(mlpC)
    accuracies.append(accuracy(mlpC)), recalls.append(recall(mlpC)), precisions.append(precision(mlpC))

    #Train Adaboost, store confusion matrix, accuracy, recall and precision
    adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train.ravel())
    adaboost_pred = adaboost.predict(X_test)
    adaboostC = confusion_matrix(y_test, adaboost_pred)
    cMatrices.append(adaboostC)
    accuracies.append(accuracy(adaboostC)), recalls.append(recall(adaboostC)), precisions.append(precision(adaboostC))


    #get index of best classifier
    iBest = accuracies.index(max(accuracies))

    with open('a1_3.1.csv', mode = 'w') as out:
        outWriter = csv.writer(out, delimiter = ',', quotechar = '"')
        for i in range(5):
            write = [i+1]
            write.extend(accuracies[i]), write.extend(recalls[i]), write.extend(precision[i]), write.extend(cMatrices[i][0, :]),\
                    write.extend(cMatrices[i][1, :]), write.extend(cMatrices[i][2, :]), write.extend(cMatrices[i][3, :])
            outWriter.writerow(write)
    return (X_train, X_test, y_train, y_test, iBest)



def class32():
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    X_train, X_test, y_train, y_test, iBest = class31("out.npz")
    classifiers = [LinearSVC(max_iter= 10000), SVC(gamma = 2), RandomForestClassifier(n_estimators=10, max_depth=5), MLPClassifier(alpha=0.5), AdaBoostClassifier()]
    best = classifiers[iBest]
    iters = [1000, 5000, 10000, 15000, 20000]
    accuracies = []
    #for each specified # of iterations, run a test
    for iter in iters:
        #only use iter number of rows
        X = X_train[0:iter, :]
        y = y_train[0:iter, :]
        #fit the classifier to these data pts.
        best.fit(X, y.ravel())
        pred = best.predict(X_test)
        #get confusion matrix and store accuracy
        confusion = confusion_matrix(pred, y_test)
        acc = accuracy(confusion)
        accuracies.append(acc)

    with open('a1_3.2.csv', mode = 'w') as out2:
        outWriter2 = csv.writer(out2, delimiter = ',', quotechar= '"')
        outWriter2.writerow(accuracies)
        outWriter5 = csv.writer(out2, delimiter = " ", quoting=csv.QUOTE_MINIMAL, quotechar= "'")
        outWriter5.writerow(["Increased training samples leads to increased accuracy. Firstly we can hypothesize that this is due to the fact that "
                           "increased training data increases the likelihood that some of the data tested on has been seen before (or something"
                           "similar to it). Moreover, as the # of samples increases, the model can gain more nuanced understanding of the"
                           "differences between the classes. It will both help decrease underfitting (by learning the actual distinctions"
                           "of the classes), but also reduce overfitting since there will be a greater diversity of data points seen, meaning"
                           "it is less likely to fit to any single example's quirks, and will use a wider variety of distinctions."])



    X_1k = X_train[:1000, :]
    y_1k = y_train[:1000, :]

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    ks = [5, 10, 20, 30, 40, 50]
    k1 = []
    k32 = []
    bestFive = []
    classifiers = [LinearSVC(max_iter= 10000), SVC(gamma = 2), RandomForestClassifier(n_estimators=10, max_depth=5), MLPClassifier(alpha=0.5), AdaBoostClassifier()]
    selectors = []
    for k in ks:
        selector = SelectKBest(f_classif, k=k)
        X_1k_new = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_

        pp_indices = list(selector.get_support(True))
        feats1k = []
        for index in pp_indices:
            feats1k.append(pp[index])
        k1.append(feats1k)


        selector2 = SelectKBest(f_classif, k=k)
        X_train_new = selector2.fit_transform(X_train, y_train.ravel())
        pp2 = selector2.pvalues_

        pp2_indices = list(selector2.get_support(True))
        feats32k = []
        for index in pp2_indices:
            feats32k.append(pp2[index])
        k32.append(feats32k)

        if k == 5:
            bestFive.append(X_1k_new), bestFive.append(X_train_new)
            indices1k = selector.get_support(indices= True)
            indices32k = selector.get_support(indices=True)
            selectors.append(indices1k)
            selectors.append(indices32k)
    X_test_1k = np.take(X_test, selectors[0], axis = 1)
    X_test_32k = np.take(X_test, selectors[1], axis = 1)
    classifier_1k = classifiers[i]
    classifier_1k.fit(bestFive[0], y_1k)
    pred_1k = classifier_1k.predict(X_test_1k)
    confusion_1k = confusion_matrix(pred_1k, y_test)
    classifier_full = classifiers[i]
    classifier_full.fit(bestFive[1], y_train.ravel())
    pred_full = classifier_full.predict(X_test_32k)
    confusion_full = confusion_matrix(pred_full, y_test)
    with open('a1_3.3.csv', mode = 'w') as out3:
        outWriter3 = csv.writer(out3, delimiter = ',', quotechar= '"')
        for i in range(len(ks)):
            write = [ks[i]]
            write.extend(k1[i]), write.extend(k32[i])
            outWriter3.writerow(write)
        outWriter3.writerow([accuracy(confusion_1k), accuracy(confusion_full)])
    class34('out.npz', i)



def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    file = np.load(filename)
    data = file['arr_0']
    classifiers = [LinearSVC(max_iter= 10000), SVC(gamma = 2), RandomForestClassifier(n_estimators=10, max_depth=5), MLPClassifier(alpha=0.5), AdaBoostClassifier()]
    tot = [0 for j in range(5)]
    vectors = [np.zeros(5) for l in range(5)]
    y = data[:, -1:]
    X = data[:, :-1]
    kf = KFold(shuffle=True, n_splits=5)
    bigAccuracies = []
    q = 0
    for train_index, test_index in kf.split(X):
        accuracies = []
        j = 0
        #one run of cross validation
        for classifier in classifiers:
            classifier.fit(X[train_index], y[train_index])
            pred = classifier.predict(X[test_index])
            c = confusion_matrix(pred, y[test_index])
            acc = accuracy(c)
            #store accuracy of classifier for this run
            accuracies.append(acc)
            tot[j] += acc
            vectors[j][q] = acc
            j += 1
        q += 1
        bigAccuracies.append(accuracies)
    bestIndex = tot.index(max(tot))
    best = vectors[bestIndex]
    pVals = []
    for t in range(len(classifiers)):
        if t != bestIndex:
            S = stats.ttest_rel(vectors[t], best)
            pVals.append(S[1])
    with open('a1_3.4.csv', mode = 'w') as out4:
        outWriter4 = csv.writer(out4, delimiter = ',', quotechar = '"')
        for i in range(len(bigAccuracies)):
            outWriter4.writerow(bigAccuracies[i])
        outWriter4.writerow(pVals)

if __name__ == "__main__":
    # parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    # args = parser.parse_args()
    class32()

    # TODO : complete each classification experiment, in sequence.
