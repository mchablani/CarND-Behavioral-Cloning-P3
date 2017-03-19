import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Dropout, Activation
import numpy as np
import keras as k

log = pd.read_csv('./data/driving_log.csv', header=None, names=['center', 'l', 'r', 'steering', 't', 'b', 's'])
print(log.columns)

images = []
s = log['steering']

for l in log['center']:
  # print(l)
  fn = l.split('/')[-1]
  source_path = "./data/IMG/" + fn
  # print(source_path)
  image = cv2.imread(source_path)
  images.append(image)

print(len(images))
print(len(s))

# image_flipped = np.fliplr(images)
# measurement_flipped = -measurement


# print(images[2].shape)

X_train = np.array(images)
y_train = np.array(s)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')