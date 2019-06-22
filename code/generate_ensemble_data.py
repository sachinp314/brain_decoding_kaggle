import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from pickle import dump, load


def train_models(dropInitial=False):
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        print(f'training {s}')

        temp = loadmat('../data/train_all/train_subject{}.mat'.format(s))
        X = temp['X']
        y = temp['y']

        if dropInitial:
            X = X[:, :, int(X.shape[2]/3):]
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        y = y.reshape(y.shape[0])

        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        model = LogisticRegression(solver='liblinear', penalty='l1')
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
        pipeline.fit(X, y)
        if dropInitial:
            fname = f'../models/individual_subjects/model_{s}_drop_initial.pkl'
        else:
            fname = f'../models/individual_subjects/model_{s}.pkl'
        with open(fname, 'wb') as f:
            dump(pipeline, f)


def make_dataset(dropInitial=False):
    subjects, pipelines = [], {}
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        subjects.append(s)
        if dropInitial:
            fname = f'../models/individual_subjects/model_{s}_drop_initial.pkl'
        else:
            fname = f'../models/individual_subjects/model_{s}.pkl'
        with open(fname, 'rb') as f:
            pipelines[s] = load(f)
    columns = ['subject', 'trial', 'actual'] + \
        [f'model_{s}_pred' for s in subjects]
    ensembleData = pd.DataFrame(columns=columns)

    for s in subjects:
        temp = loadmat('../data/train_all/train_subject{}.mat'.format(s))
        X, y = temp['X'], temp['y']
        if dropInitial:
            X = X[:, :, int(X.shape[2]/3):]
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        tempDf = pd.DataFrame(columns=columns)
        tempDf.loc[:, 'trial'] = np.arange(len(X))
        tempDf.loc[:, 'subject'] = s
        tempDf.loc[:, 'actual'] = y
        for other in subjects:
            if s == other:
                pred = np.nan
            else:
                pred = pipelines[other].predict_proba(X)[:, 1]
            tempDf.loc[:, f'model_{other}_pred'] = pred
        ensembleData = ensembleData.append(
            tempDf, ignore_index=True, sort=True)
    if dropInitial:
        fname = '../data/ensemble/ensemble_drop_initial.csv'
    else:
        fname = '../data/ensemble/ensemble.csv'
    ensembleData.to_csv(fname, index=False)


def main(train=True):
    dropInitial=True
    if train:
        train_models(dropInitial)
    make_dataset(dropInitial)


if __name__ == '__main__':
    main()
