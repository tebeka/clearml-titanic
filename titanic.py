import pickle
from argparse import ArgumentParser
from collections.abc import Sequence

import pandas as pd
from clearml import Task
from clearml.config import config_obj
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def as_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: as_plain_dict(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return obj.__class__(as_plain_dict(v) for v in obj)

    return obj


parser = ArgumentParser()
parser.add_argument('-ts', '--test-size', type=float, default=0.3)
parser.add_argument('-q', '--queue', default='')
args = parser.parse_args()

task = Task.init(project_name='lab', task_name='titanic')

if args.queue:
    task.execute_remotely(args.queue)  # Will exit the program


cfg = config_obj.get('auto_scaler')
print('CONFIG:', as_plain_dict(cfg))

df = pd.read_csv('titanic.csv')

feature_cols = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
]
dfX = df[feature_cols].copy()

dfX['Age'].fillna(dfX['Age'].mean(), inplace=True)

embarked = dfX.pop('Embarked')
embarked.fillna(embarked.mode()[0], inplace=True)

embarked_dummies = pd.get_dummies(embarked)

gender = dfX.pop('Sex')
gender_dummies = pd.get_dummies(gender)

dfX = pd.concat([dfX, embarked_dummies, gender_dummies], axis=1)

X = dfX.values
y = df['Survived'].values

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=args.test_size)
clf = RandomForestClassifier()

clf.fit(X_train, y_train)
print('random forest', clf.score(X_test, y_test))
model_file = 'model.pkl'
with open(model_file, 'wb') as out:
    pickle.dump(clf, out)

task.upload_artifact('model', clf, auto_pickle=True)
