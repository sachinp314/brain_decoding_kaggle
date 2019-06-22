import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pickle import dump


def train_subject_classifiers(dropInitial=False):
    X_lens = {0: 0}
    X = None
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        
        data = loadmat('../data/train_all/train_subject{}.mat'.format(s))
        if X is None:
            X = data['X']
        else:
            X = np.concatenate((X, data['X']))
        X_lens[i] = X_lens[i-1] + data['X'].shape[0]
    if dropInitial:
        X = X[:, :, int(X.shape[2]/3):]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        
        y = np.zeros(shape = X.shape[0])
        y[X_lens[i-1]:X_lens[i]] = 1
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        
        scaler = StandardScaler()
        model = LogisticRegression(solver='liblinear', penalty='l1')
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
        pipeline.fit(X_res, y_res)
        if dropInitial:
            fname = f'../models/classify_between_subjects/model_{s}_drop_initial.pkl'
        else:
            fname = f'../models/classify_between_subjects/model_{s}.pkl'
        with open(fname, 'wb') as f:
            dump(pipeline, f)
        print(f'trained classifier for s{s}')


def main():
    train_subject_classifiers(dropInitial=True)


if __name__ == '__main__':
    main()