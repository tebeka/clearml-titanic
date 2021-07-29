import pickle
from argparse import ArgumentParser

import pandas as pd
from clearml import Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument('-ts', '--test-size', type=float, default=0.3)
parser.add_argument('-c', '--clearml', action='store_true', default=False)
parser.add_argument('-q', '--queue', default='')
args = parser.parse_args()

if args.clearml or args.queue:
    task = Task.init(project_name='lab', task_name='titanic')

if args.queue:
    task.execute_remotely(args.queue)  # Will exit the program

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

if args.clearml:
    task.upload_artifact('model', clf, auto_pickle=True)
