import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Dropout, Activation, Cropping2D, BatchNormalization
import numpy as np
import keras as k

images = []
measurements = []


def getImages(imgPath, type, adj, header):
  if header:
    log = pd.read_csv(imgPath + '/driving_log.csv')
  else:
    log = pd.read_csv(imgPath + '/driving_log.csv', header=None, names=['center', 'left', 'right', 'steering', 't', 'b', 's'])
  # print(log.columns)

  images = []
  measurements = []

  # s = log['steering']

  for idx, l in log.iterrows():
    # print(l)
    fn = l[type].split('/')[-1]
    source_path = imgPath + "/IMG/" + fn
    # print(source_path)
    image = cv2.imread(source_path)
    s = l['steering']
    measurement = float(s) + adj
    images.append(image)
    measurements.append(measurement)

    # Data Augmentation
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

  # print(len(images))
  # print(len(measurements))
  return images, measurements
  # image_flipped = np.fliplr(images)
  # measurement_flipped = -measurement


def getImagesForDir(dir, header=False):
  images = []
  measurements = []
  images, measurements = getImages(dir, 'center', 0.0, header)
  imagesl, measurementsl = getImages(dir, 'left', 1.5, header)
  imagesr, measurementsr = getImages(dir, 'right', -1.5, header)

  images.extend(imagesl)
  images.extend(imagesr)

  measurements.extend(measurementsl)
  measurements.extend(measurementsr)
  return images, measurements


# i, m = getImagesForDir("./data")
# images.extend(i)
# measurements.extend(m)
# print(len(images))

i, m = getImagesForDir("./data.bk", header=True)
images.extend(i)
measurements.extend(m)
print(len(images))

# i, m = getImagesForDir("./data.2")
# images.extend(i)
# measurements.extend(m)
# print(len(images))
#
# i, m = getImagesForDir("./data_1_r")
# images.extend(i)
# measurements.extend(m)
# print(len(images))

# i, m = getImagesForDir("./data_2.1")
# images.extend(i)
# measurements.extend(m)
# print(len(images))

# i, m = getImagesForDir("./data_1_r.1")
# images.extend(i)
# measurements.extend(m)
# print(len(images))

i, m = getImagesForDir("./ndata_1_f.1")
images.extend(i)
measurements.extend(m)
print(len(images))


print(len(images))
print(len(measurements))
X_train = np.array(images)
y_train = np.array(measurements)


# model = Sequential()
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# model.add(Convolution2D(32, 3, 3))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Convolution2D(128, 3, 3))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(BatchNormalization())
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=2, batch_size=128)
model.save('model.h5')