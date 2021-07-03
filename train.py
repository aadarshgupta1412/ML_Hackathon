import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sn
import numpy as np
import pandas as pd
import math
import datetime
import platform
import csv
import os
from PIL import Image
mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)
# Save image parameters to the constants that we will use later for data re-shaping and
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape 
IMAGE_CHANNELS = 1
print('IMAGE_WIDTH:', IMAGE_WIDTH);
print('IMAGE_HEIGHT:', IMAGE_HEIGHT);
print('IMAGE_CHANNELS:', IMAGE_CHANNELS);
with open('opdata1.csv', 'r') as file: 
    reader = csv.reader(file)
    x_list = []
    x_train_list = []
    y_list = []
    y_train_list = [] 
    i=0
    for row in reader:
        imageName = row[0]
        operator_name = row[1]
        #print(imageName)
        #path = "C:\\Users\\Aadarsh Gupta\\imgdata" 
        path = os.path.join("imgdata", imageName)
        if(os.path.exists(path)and imageName!=""):
            im = Image.open(path, "r")
            #im = list(im.getdata()) im = np.array(im)
            #if (i>200):
            #    break 
            if(i<0.7*50000):
                im = np.array(im)
                x_list.append(im) 
                if(operator_name=="plus"):
                    y_list.append(10) 
                if(operator_name=="minus"):
                    y_list.append(11) 
                if(operator_name=="mul"): 
                    y_list.append(12) 
                if(operator_name=="div"):
                    y_list.append(13) 
            else:
                im = np.array(im)
                x_train_list.append(im)

                if(operator_name=="plus"):
                    y_train_list.append(10) 
                if(operator_name=="minus"):
                    y_train_list.append(11)
                if(operator_name=="mul"):
                    y_train_list.append(12)
                if(operator_name=="div"): 
                    y_train_list.append(13)
        i+=1
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    x_train_list = np.array(x_train_list)
    y_train_list = np.array(y_train_list)
    # print csv content of row can remove it
    print(x_list.shape)
    print(type(x_train), type(y_train), type(x_list), type(y_list), type(x_train_list))
    x_train = np.concatenate((x_train, x_list))
    y_train = np.concatenate((y_train, y_list))
    x_test = np.concatenate((x_test, x_train_list))
    y_test = np.concatenate((y_test, y_train_list))
    print('x_train:', x_train.shape) 
    print('y_train:', y_train.shape) 
    print('x_test:', x_test.shape) 
    print('y_test:', y_test.shape)
x_train_with_chanels = x_train.reshape( 
    x_train.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)
x_test_with_chanels = x_test.reshape( 
    x_test.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)
print('x_train_with_chanels:', x_train_with_chanels.shape) 
print('x_test_with_chanels:', x_test_with_chanels.shape)

x_train_normalized = x_train_with_chanels / 255 
x_test_normalized = x_test_with_chanels / 255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.25)) 
model.add(tf.keras.layers.Dense(14,activation=tf.nn.softmax))

tf.keras.utils.plot_model( 
    model,
    show_shapes=True,
    show_layer_names=True, 
)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam_optimizer, 
    loss=tf.keras.losses.sparse_categorical_crossentropy, 
    metrics=['accuracy']
)

log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

training_history = model.fit( 
    x_train_normalized,
    y_train,
    epochs=10, 
    validation_data=(x_test_normalized, y_test), 
    callbacks=[tensorboard_callback]
)

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='test set') 
plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy') 
plt.plot(training_history.history['accuracy'], label='training set') 
plt.plot(training_history.history['val_accuracy'], label='test set') 
plt.legend()

%%capture
train_loss, train_accuracy = model.evaluate(x_train_normalized, y_train)
print('Training loss: ', train_loss)
print('Training accuracy: ', train_accuracy)

%%capture
validation_loss, validation_accuracy = model.evaluate(x_test_normalized, y_test)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)

model_name = 'digits_recognition_cnn.h5' 
model.save(model_name, save_format='h5')

loaded_model = tf.keras.models.load_model(model_name)
predictions_one_hot = loaded_model.predict([x_test_normalized]) 
print('predictions_one_hot:', predictions_one_hot.shape)

pd.DataFrame(predictions_one_hot)
predictions = np.argmax(predictions_one_hot, axis=1) 
pd.DataFrame(predictions)

