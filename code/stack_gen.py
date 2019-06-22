import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pickle import load


def stack_gen_eval(cov=True, dropInitial=False):
    if dropInitial:
        fname = '../data/ensemble/ensemble_drop_initial.csv'
    else:
        fname = '../data/ensemble/ensemble.csv'
    data = pd.read_csv(fname, dtype={'subject': str})
    data.fillna(data.mean(), inplace=True)
    subjects = []
    for i in range(1, 17):
        if i < 10:
            subjects.append('0' + str(i))
        else:
            subjects.append(str(i))
    
    if cov:
        X_lens = {0: 0}
        X_orig = None
        for i in range(1, 17):
            if i < 10:
                s = '0' + str(i)
            else:
                s = str(i)
            
            data_orig = loadmat('../data/train_all/train_subject{}.mat'.format(s))
            if X_orig is None:
                X_orig = data_orig['X']
            else:
                X_orig = np.concatenate((X_orig, data_orig['X']))
            X_lens[i] = X_lens[i-1] + data_orig['X'].shape[0]
        if dropInitial:
            X_orig = X_orig[:, :, int(X_orig.shape[2]/3):]
        X_orig = X_orig.reshape(X_orig.shape[0], X_orig.shape[1] * X_orig.shape[2])
        assert len(data) == X_orig.shape[0]

    accs = []
    for i in range(1, 17):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        model1 = LogisticRegression(solver='liblinear', penalty='l2')
        model2 = SVC(gamma='auto', kernel='rbf')
        model3 = RandomForestClassifier(n_estimators=100)
        bases = [('m1', model1), ('m2', model2), ('m3', model3)]
        # model = model1
        model = VotingClassifier(bases, voting='hard')

        X_train = data.loc[data['subject'] != s, [
            f'model_{i}_pred' for i in subjects]]
        y_train = data.loc[data['subject'] != s, 'actual']
        
        if cov:
            weights = np.load('../data/cs/CS_weights_{}.npy'.format(i))
            weights = np.delete(weights, np.s_[X_lens[i-1]:X_lens[i]], 0)
            # print((weights.shape, X_train.shape))
            X_train = scaler.fit_transform(X_train)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            X_train = scaler.fit_transform(X_train)
            model.fit(X_train, y_train)
        
        X_test = data.loc[data['subject'] == s, [f'model_{i}_pred' for i in subjects]]
        X_test = scaler.transform(X_test)
        y_test = data.loc[data['subject'] == s, 'actual']
        # y_pred = pipeline.predict(X_test)
        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred), 2)
        accs.append(acc)
    fname = '../results/stack_gen'
    if cov:
        fname += '_cov'
    if dropInitial:
        fname += '_drop_initial'
    fname += '.txt'
    with open(fname, 'w') as f:
        f.write('\n'.join([str(round(a, 2)) for a in accs]))


def main():
    cov, dropInitial = False, False
    stack_gen_eval(cov, dropInitial)


if __name__ == '__main__':
    main()
