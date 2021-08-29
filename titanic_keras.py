from argparse import ArgumentParser

import pandas as pd
from clearml import Task
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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


clf = Sequential()
clf.add(Dense(5, input_dim=X.shape[1], activation='relu'))
clf.add(Dense(1, input_dim=5, activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=1000)
prediction = [p[0] for p in clf.predict(X_test).tolist()]
result = pd.DataFrame({
    'actual': y_test,
    'predicted': prediction,
})
print(result)
