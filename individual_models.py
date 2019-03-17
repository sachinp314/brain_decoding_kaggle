import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_model(subjectNo, dropInitial=False, modelType='logreg'):
    data = loadmat('data/train_all/train_subject{}.mat'.format(subjectNo))
    X = data['X']
    y = data['y']

    print('train shape before flattening: {}'.format(X.shape))
    if dropInitial:
        X = X[:, :, int(X.shape[2]/3):]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = y.reshape(y.shape[0])
    # print('train shape after flattening: {}'.format(X.shape))

    meanAcc = 0
    kf = KFold(n_splits=6)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if modelType == 'logreg':
            model = LogisticRegression(solver='liblinear', penalty='l1')
        elif modelType == 'rfc':
            model = RandomForestClassifier(
                n_estimators=100, max_depth=2, random_state=0)
        elif modelType == 'svm':
            model = SVC(gamma='auto')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_class = np.round(y_pred)
        # print(confusion_matrix(y_test, y_pred_class))
        acc = accuracy_score(y_test, y_pred_class)
        meanAcc += acc
        # print('ACCURACY = {}'.format(acc))
    meanAcc /= 6
    print('mean accuracy for s{} = {}'.format(subjectNo, meanAcc))
    return meanAcc


def main():
    meanAcc = 0
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        meanAcc += train_model(s, dropInitial=False, modelType='logreg')
    meanAcc /= 16
    print('MEAN ACCURACY ACROSS ALL SUBJECTS = {}'.format(meanAcc))


if __name__ == '__main__':
    main()
