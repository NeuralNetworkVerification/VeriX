from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import tf2onnx
from VeriX import *

"""
download and process MNIST data.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

"""
show a simple example usage of VeriX. 
"""
verix = VeriX(dataset="MNIST",
              image=x_test[10],
              model_path="models/mnist-10x2.onnx")
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=0.05)
exit()

"""
or you can train your own MNIST model.
Note: to obtain sound and complete explanations, train the model from logits directly.
"""
model_name = 'mnist-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# model.save('models/' + model_name + '.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='models/' + model_name + '.onnx')




