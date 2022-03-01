import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('datasets/citrus.csv')

df.name[df.name == 'orange'] = 0
df.name[df.name == 'grapefruit'] = 1

dataset = df.values

X = dataset[:, 1:6]
y = dataset[:, 0]

minMaxScaler = preprocessing.MinMaxScaler()
X_scale = minMaxScaler.fit_transform(X)

X_test, X_train, y_test, y_train = train_test_split(X_scale, y, test_size=0.3)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = Sequential([
    Dense(32, activation='relu', input_shape=(5, )),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
print(model.evaluate(X_test, y_test))