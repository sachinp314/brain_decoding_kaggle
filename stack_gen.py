import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def stack_gen_eval():
    data = pd.read_csv('data/ensemble/ensemble.csv', dtype={'subject': str})
    data.fillna(data.mean(), inplace=True)
    subjects = []
    for i in range(1, 17):
        if i < 10:
            subjects.append('0' + str(i))
        else:
            subjects.append(str(i))

    for s in subjects:
        scaler = StandardScaler()
        model = LogisticRegression(solver='liblinear', penalty='l1')
        pipeline = Pipeline([('scaler', scaler), ('model', model)])

        X = data.loc[data['subject'] != s, [
            f'model_{i}_pred' for i in subjects]]
        y = data.loc[data['subject'] != s, 'actual']

        pipeline.fit(X, y)
        y_test = data.loc[data['subject'] == s, 'actual']
        y_pred = pipeline.predict(data.loc[data['subject'] == s, [
                                  f'model_{i}_pred' for i in subjects]])
        print(f'accuracy s{s} is {accuracy_score(y_test, y_pred)}')


if __name__ == '__main__':
    stack_gen_eval()
