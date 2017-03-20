import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Dropout, Activation, Cropping2D
import numpy as np
import keras as k

images = []
measurements = []


def getImages(imgPath):
  log = pd.read_csv(imgPath + '/driving_log.csv', header=None, names=['center', 'l', 'r', 'steering', 't', 'b', 's'])
  # print(log.columns)

  images = []
  measurements = []

  # s = log['steering']

  for idx, l in log.iterrows():
    # print(l)
    fn = l['center'].split('/')[-1]
    source_path = imgPath + "/IMG/" + fn
    # print(source_path)
    image = cv2.imread(source_path)
    measurement = l['steering']
    images.append(image)
    measurements.append(measurement)

    # Data Augmentation
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

  print(len(images))
  print(len(measurements))
  return images, measurements
  # image_flipped = np.fliplr(images)
  # measurement_flipped = -measurement


# print(images[2].shape)

images, measurements = getImages("./data")
X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
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