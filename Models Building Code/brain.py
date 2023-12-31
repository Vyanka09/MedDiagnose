import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split



path = '/content/drive/MyDrive/dataset'
train_dir = os.path.join(path, 'train')
path2='/content/drive/MyDrive/dataset'
test_dir = os.path.join(path2, 'valid')
print(train_dir)
print(test_dir)
print(os.listdir(train_dir))

# Hyperparams
IMAGE_SIZE = 128
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 10
BATCH_SIZE = 16

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# data generators
training_data_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_data_generator = ImageDataGenerator(rescale=1./255)
# Data preparation

training_generator = training_data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
validation_generator = validation_data_generator.flow_from_directory(
    test_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")


sample, label = next(validation_generator)
print(sample[0])
print(label[0])

# model
model = Sequential()

model.add(Conv2D(32, 3, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

# model.add(Activation('sigmoid'))
model.summary()
# compile model

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

print(len(training_generator.filenames))
# train model

history=model.fit(
    training_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
)


model.save('braintumor.h5')
 #   validation_steps=len(validation_generator.filenames) // BATCH_SIZE

model = tf.keras.models.load_model('braintumor.h5')


sample1, label1 = next(validation_generator)
predictions = model.predict(sample1)
print(predictions)


def check_results():
   class_names = [ 'No', 'Yes']
   sample1, label1 = next(validation_generator)
   predictions = model.predict(sample1)
   for num in range(len(predictions)):
       if predictions[num] > 0.5:
           print('prediction: '+'Yes'+' ' + str(int(predictions[num]*100))+ '%')
       else:
           print('prediction: '+'No'+' ' + str(100- int(predictions[num]*100))+ '%')

       print('actual: '+ class_names[int(label1[num])])
       plt.imshow(sample1[num])
       plt.show()

check_results()

model.evaluate(training_generator)
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
