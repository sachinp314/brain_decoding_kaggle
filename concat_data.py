import numpy as np
from scipy.io import loadmat


def combine_datasets():
    '''
    concatenates all subjects and saves it as another dataset.
    '''
    newX, newY = None, None
    for subjectNo in range(1, 17):
        if subjectNo < 10:
            s = '0' + str(subjectNo)
        else:
            s = str(subjectNo)
        data = loadmat('data/train_all/train_subject{}.mat'.format(s))
        if newX is None or newY is None:
            newX, newY = data['X'], data['y']
        else:
            newX = np.concatenate((newX, data['X']))
            newY = np.concatenate((newY, data['y']))
    print('SHAPE OF COMBINED DATASET: {}'.format(newX.shape))
    np.save('data/train_all/allX.npy', newX)
    np.save('data/train_all/allY.npy', newY)


if __name__ == '__main__':
    combine_datasets()
