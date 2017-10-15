from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense,BatchNormalization,Conv2D, MaxPooling2D, Dropout
import json
dir = "F:/blob/ds/age-predictor/"

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(dir,target_size=(150, 150),
    batch_size=16,class_mode='categorical',shuffle=True)
with open("./class_indices.json","w") as f:
    f.write(json.dumps(generator.class_indices))

K.clear_session()

model = Sequential()
model.add(Conv2D(64, (4, 4), input_shape=(150, 150, 3)))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((3, 3)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(Dense(generator.num_class, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit_generator(generator,validation_data=generator,validation_steps=100,
        steps_per_epoch=100,epochs=3)

model.evaluate_generator(generator,steps=100)

model.save(filepath="./predictor.h5",overwrite=True)
