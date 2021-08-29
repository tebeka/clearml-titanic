from argparse import ArgumentParser

import numpy as np
from clearml import Task
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

parser = ArgumentParser()
parser.add_argument('-c', '--clearml', action='store_true', default=False)
parser.add_argument('-q', '--queue', default='')
args = parser.parse_args()

if args.clearml or args.queue:
    task = Task.init(project_name='lab', task_name='titanic')

if args.queue:
    task.execute_remotely(args.queue)  # Will exit the program


X = np.arange(10000)
y = np.sin(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = Sequential()
clf.add(Dense(5, input_dim=1, activation='relu'))
clf.add(Dense(1, input_dim=5, activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10)
prediction = [p[0] for p in clf.predict(X_test).tolist()]
print(prediction)
