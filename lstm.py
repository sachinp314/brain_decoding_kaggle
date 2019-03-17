import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.model_selection import KFold
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential


def train_model(subjectNo, dropInitial=False):
    data = loadmat('data/train_all/train_subject{}.mat'.format(subjectNo))
    X = data['X']
    y = data['y']

    if dropInitial:
        X = X[:, :, int(X.shape[2]/3):]
    X = np.swapaxes(X, 1, 2)
    print('X shape: {}'.format(X.shape))

    accuracies = []
    kf = KFold(n_splits=6)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

        model = get_model((X_train.shape[1:]))

        print(model.count_params())
        # print(model.summary())

        history = model.fit(X_train, y_train, epochs=10,
                            validation_data=(X_test, y_test),
                            verbose=1)
        accuracies.append(history.history['val_binary_accuracy'][-1])

    accuracies = np.array(accuracies)
    meanAcc, varAcc = accuracies.mean(), accuracies.var()
    print('{}: mean accuracy = {}; variance = {}'.format(
        subjectNo, meanAcc, varAcc))
    return meanAcc


def train_all(dropInitial=False):

    X = np.load('data/train_all/allX.npy')
    y = np.load('data/train_all/allY.npy')

    if dropInitial:
        X = X[:, :, int(X.shape[2]/3):]

    X = np.swapaxes(X, 1, 2)
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    # for now, use first subject as test
    X_train, y_train = X[594:, :, :], y[594:, :]
    X_test, y_test = X[:594, :, :], y[:594, :]
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=42)
    '''

    # Scaling 3D data
    scalers = {}
    for i in range(X_train.shape[1]):
        scalers[i] = StandardScaler()
        X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

    model = get_model((X_train.shape[1:]))

    print(model.count_params())
    # print(model.summary())

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))
    # print(history)


def get_model(inp_shape):
    model = Sequential()
    model.add(LSTM(16, input_shape=inp_shape, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return model


def main(concat=False):
    if concat:
        train_all(dropInitial=True)
    else:
        meanAcc = 0
        for i in range(1, 17):
            if i < 10:
                s = '0' + str(i)
            else:
                s = str(i)
            meanAcc += train_model(s, dropInitial=True)
        meanAcc /= 16
        print('MEAN ACCURACY ACROSS ALL SUBJECTS = {}'.format(meanAcc))


if __name__ == '__main__':
    main(concat=False)
    # train_model('01')
