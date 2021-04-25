from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


import tensorflow as tf
print(tf.__version__)

#  re-size all the images to this
IMAGE_SIZE = [224,224]

tarin_path = 'Malaria_Detection_Dataset/Train'
test_path = 'Malaria_Detection_Dataset/Test'

# Import the Vgg16 Library as shown below and add preprocessing layer to the frequency
# Here we will be using imagenet weights
# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

vgg.summary()

# Transfer Leraning Technique
# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False
  
  
# useful for getting number of classes
folders = glob('Malaria_Detection_Dataset/Train/*')


# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)


# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Malaria_Detection_Dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Malaria_Detection_Dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plot the loss function
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


import tensorflow
from keras.models import load_model
model.save('model_vgg19.h5')


y_pred = model.predict(test_set)

import numpy as np
y_pred = np.argmax(y_pred, axis = 1)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model_vgg19.h5')


img=image.load_img('Malaria_Detection_Dataset/Test/Uninfected/MDTestU (12).png',target_size=(224,224))
x=image.img_to_array(img)
x
x.shape

# Reshape and preprocess the extracted values of the image
x = x/255
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
img_data.shape

# Predicting the states of the test image
model.predict(img_data)
a = np.argmax(model.predict(img_data), axis=1)
if(a==1):
    print("Symptoms of Uninfected Conditions")
else:
    print("Symptoms of Infected Conditions")




