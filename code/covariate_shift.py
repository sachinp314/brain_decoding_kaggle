import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler


def build_origin_datasets(test_subject, build_X):
    '''
    concatenates all subjects from test and train data
    and saves it as another dataset. We add another target column
    that specifies 1 for train set and 0 for test set
    '''
    X, origin = None, None
    for subjectNo in range(1, 17):
        if subjectNo < 10:
            s = '0' + str(subjectNo)
        else:
            s = str(subjectNo)
        data = loadmat('../data/train_all/train_subject{}.mat'.format(s))
        if subjectNo == test_subject: 
            data['origin'] = np.ones(data['y'].shape)
        else:
            data['origin'] = np.zeros(data['y'].shape)
        if X is None or origin is None:
            X, origin = data['X'], data['origin']
        else:
            if build_X:
                X = np.concatenate((X, data['X']))
            origin = np.concatenate((origin, data['origin']))
    print(f'built origin for {test_subject}')
    if build_X:
        np.save('../data/cs/CS_X.npy', X)
    np.save('../data/cs/CS_origin_{}.npy'.format(test_subject), origin)
   

def find_weights(X_orig, y_orig, test_subject):
    '''
    computes weights for each data point that is 
    proportional to the probability of it belonging to
    the testing data
    '''
    clf = RFC(n_estimators=10)
    # clf = LR(solver='lbfgs')
    X = X_orig.reshape(X_orig.shape[0], X_orig.shape[1] * X_orig.shape[2])
    y = y_orig.reshape(y_orig.shape[0])
    predictions = np.zeros(y.shape)
    kf = SKF(n_splits=10, shuffle=True, random_state=1234)
    # kf = KFold(n_splits=10, shuffle=True)
    for train_idx, test_idx in kf.split(X, y):
        # print('Training discriminator model for fold {}'.format(fold))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx] 
        X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    print(f'{test_subject}: ROC-AUC for train and test distributions: {AUC(y, predictions)}')
    weights = predictions # (1/predictions_test) - 1 
    weights /= np.mean(weights) # we do this to re-normalize the computed log-loss
    np.save('../data/cs/CS_weights_{}.npy'.format(test_subject), weights)


def main():
    build=False
    if build:
        for i in range(1, 17):
            build_origin_datasets(i, build_X = (i == 1))
        return
    
    X = np.load('../data/cs/CS_X.npy')
    for i in range(1, 17):
        y = np.load('../data/cs/CS_origin_{}.npy'.format(i))
        print(f'finding weights for {i}')
        find_weights(X, y, i)
    
    
if __name__ == '__main__':
    main()

    