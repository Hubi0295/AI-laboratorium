import pandas as pd
import numpy as np
from keras.api.datasets import mnist, cifar10
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape #kształt zbioru treningowego 50000 obrazow, kazdy z nich to macierz 32x32 a każdy z nich to wektor 3 elementowy
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape #kształt zbioru treningowego 60000 obrazow, kazdy z nich to macierz 28x28 obraz w skali szarości
#trzeba rozszerzyć do kolejnego wymiaru

x_train = np.expand_dims(x_train, axis=-1)# dodaj na końcy wymiar
x_train.shape

x_test = np.expand_dims(x_test, axis=-1)# dodaj na końcy wymiar
x_test.shape

y_train = pd.get_dummies(pd.Categorical(y_train)).values #one hot encoding
y_test = pd.get_dummies(pd.Categorical(y_test)).values

from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Lambda

filter_cnt = 32
kernel_size = (3,3)
act_funct = 'selu'
class_cnt = y_train.shape[1]
input_tensor = Input(shape = x_train.shape[1:])
input_tensor.shape

def my_act_fun(tensor):
    return tf.nn.activation.selu(tensor)

output_tensor = input_tensor = Input(x_train.shape[1:])
output_tensor = Conv2D(filter_cnt, kernel_size,activation = act_funct)(output_tensor)
output_tensor = MaxPooling2D(2,2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size,activation = act_funct)(output_tensor)
output_tensor = MaxPooling2D(2,2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size,activation = act_funct)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt,activation = 'softmax')(output_tensor)
output_tensor.shape
ANN = Model(inputs = input_tensor, outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],optimizer = 'adam')
layers = [Conv2D(filter_cnt, kernel_size,activation = act_funct),
 MaxPooling2D(2,2),
 Conv2D(filter_cnt, kernel_size,activation = act_funct),
 MaxPooling2D(2,2),
 Conv2D(filter_cnt, kernel_size,activation = act_funct),
 GlobalAveragePooling2D(),
 Dense(class_cnt, activation = 'softmax')]
output_tensor = input_tensor = Input(x_train.shape[1:])
for layer in layers:
    output_tensor = layer(output_tensor)

from keras.api.layers import Conv2D, MaxPooling2D
from keras.api.layers import concatenate
def add_inseption_module(input_tensor):
    act_func = 'relu'
    paths = [
    [Conv2D(filters = 64, kernel_size=(1,1),
     padding='same', activation=act_func)
     ],
     [Conv2D(filters = 96, kernel_size=(1,1),
     padding='same', activation=act_func),
     Conv2D(filters = 128, kernel_size=(3,3),
     padding='same', activation=act_func)
     ],
     [Conv2D(filters = 16, kernel_size=(1,1),
     padding='same', activation=act_func),
     Conv2D(filters = 32, kernel_size=(5,5),
     padding='same', activation=act_func)
     ],
     [MaxPooling2D(pool_size=(3,3),
     strides = 1, padding='same'),
    Conv2D(filters = 32, kernel_size=(1,1),
     padding='same', activation=act_func)
     ]
    ]
    for_concat = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
            for_concat.append(output_tensor)
        return concatenate(for_concat)
    

output_tensor = input_tensor = Input(x_train.shape[1:])
insept_module_cnt = 2
for i in range(insept_module_cnt):
 output_tensor = add_inseption_module(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt,
 activation='softmax')(output_tensor)
ANN = Model(inputs = input_tensor,
 outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',
 metrics = ['accuracy'], optimizer = 'adam')

from keras.api.utils import plot_model
plot_model(ANN, show_shapes=True)

def ReLOGU(tensor):
 mask = tensor >= 1
 tensor = tf.where(mask, tensor, 1)
 tensor = tf.math.log(tensor)
 return tensor
output_tensor = input_tensor = Input(x_train.shape[1:])
insept_module_cnt = 2
for i in range(insept_module_cnt):
 output_tensor = add_inseption_module(output_tensor)
output_tensor = Conv2D(32, (3,3))(output_tensor)
output_tensor = Lambda(ReLOGU)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt,
activation='softmax')(output_tensor)
ANN = Model(inputs = input_tensor,outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'], optimizer = 'adam')
plot_model(ANN, show_shapes=True)