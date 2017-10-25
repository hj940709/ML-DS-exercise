from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense,BatchNormalization,Conv2D, MaxPooling2D, Dropout
import json
#dir = "F:/blob/ds/age-predictor/"
train_dir = "E:/1aHelsinki/project/DS_project/age-predictor/images/train"
validation_dir = 'E:/1aHelsinki/project/DS_project/age-predictor/images/validation'

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(train_dir,target_size=(100, 100),
    batch_size=16,class_mode='categorical',shuffle=True)
validation_generator = datagen.flow_from_directory(validation_dir,target_size=(100, 100),
    batch_size=16,class_mode='categorical',shuffle=True)

with open("./class_indices.json","w") as f:
    f.write(json.dumps(generator.class_indices))

K.clear_session()

#build neural network
model = Sequential()

'''
#convolution layer_1
model.add(Conv2D(64, (4, 4), input_shape=(150, 150, 3)))
#pooling layer_1
model.add(MaxPooling2D((3, 3)))
#convolution layer_2
model.add(Conv2D(64, (3, 3), activation="relu"))
#pooling layer_2
model.add(MaxPooling2D((3, 3)))
#full connection layer_1
model.add(Flatten())
model.add(Dense(128, activation="relu"))
#full connection layer_2
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
'''
model.add(Conv2D(32, (5, 5), padding='same', input_shape=(100, 100, 3)))  
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))  
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))  
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))

model.add(Dropout(0.1))




model.add(Dense(generator.num_class, activation="softmax"))
#loss function, solution method, measure method
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
#train model
model.fit_generator(generator,validation_data=validation_generator,validation_steps=100,
        steps_per_epoch=100,epochs=100)
#evaluate
model.evaluate_generator(validation_generator,steps=100)

model.save(filepath="./predictor.h5",overwrite=True)
