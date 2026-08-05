"""
A script to train the GTSRB model used by gtsrb.py.

Run `python train_gtsrb.py` to train the model, export it to ONNX for onnxruntime + Marabou,
and simplify the ONNX graph so that Marabou is more likely to support all its operations.
Note: to obtain sound and complete explanations, train the model from logits directly.
"""
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tf2onnx
import onnx
from onnxsim import simplify

"""
load and process GTSRB data.
"""
gtsrb_path = 'models/gtsrb.pickle'
with open(gtsrb_path, 'rb') as handle:
    gtsrb = pickle.load(handle)
x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_valid = to_categorical(y_valid, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255

"""
train your own GTSRB model.
"""
model_name = 'gtsrb-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
datagen = ImageDataGenerator()
model.fit(datagen.flow(x=x_train, y=y_train, batch_size=64),
          steps_per_epoch=100,
          epochs=30,
          validation_data=(x_valid, y_valid),
          shuffle=1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# model.save('models/' + model_name + '.h5')
onnx_path = 'models/' + model_name + '.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, output_path=onnx_path)

"""
simplify the onnx model, so that Marabou is more likely to support all its operations.
"""
onnx_model, check = simplify(onnx.load(onnx_path))
assert check, "onnx-simplifier could not validate the simplified model"
onnx.save(onnx_model, onnx_path)
