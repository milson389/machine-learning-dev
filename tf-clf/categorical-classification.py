import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf


df = pd.read_csv('datasets/Iris.csv')

df = df.drop(columns='Id')
category = pd.get_dummies(df.Species)

new_df = pd.concat([df, category], axis=1)
new_df = new_df.drop(columns='Species')

dataset = new_df.values

X = dataset[:, 0:4]
y = dataset[:, 4:7]

minMaxScaler = preprocessing.MinMaxScaler()
X_scale = minMaxScaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, train_size=0.3)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_enc(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nAkurasi telah mencapai > 90%")
            self.model.stop_training = True


callbacks = myCallback()

# model = Sequential([
#     Dense(64, activation='relu', input_shape=(4, )),
#     Dense(64, activation='relu'),
#     Dense(3, activation='softmax')
# ])

model = Sequential([
    Dense(64, activation='relu', input_shape=(4, )),
    Dense(64, activation='relu'),
    Dense(3, activation='sigmoid')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=100, callbacks=[callbacks])

print(model.evaluate(X_test, y_test))

plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower right')
plt.show()