from keras.api.datasets import mnist
from keras.api.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Jaki kształt mają dane w przypadku obrazów
# kolorowych (3 kanały: R,G,B)?
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 50'000 elementów (obrazów), gdzie każdy to macierz 32x32
# pikseli, gdzie każdy piksel złożony jest z 3 wartości (R,G,B).
# R - intensywność koloru czerwonego,
# G - intensywność koloru zielonego,
# B - intensywność koloru niebieskiego.
x_train.shape

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(60'000, 28, 28) <-- brak wymiaru opisującego liczbę kanałów 
x_train.shape

x_train = np.expand_dims(x_train, axis= -1)
x_test = np.expand_dims(x_test, axis= -1)
# 60'000 elementów (obrazów), gdzie każdy to macierz 28x28
# pikseli, gdzie każdy piksel reprezentowany jest przez 1 wartości.
# Obrazy w skali szarości składają się z pikseli, które
# przechowują informację o jasności, np. 0 - piksel czarny, 255 - piksel biały.
x_train.shape

k = pd.Categorical(y_train)
# Jaki jest efekt funkcj get_dummies?
pd.get_dummies(k)
# Powyżej został zwrócony obiekt typu data frame, aby pobrać tablicę
# numpy należy skorzystać z pola values.
pd.get_dummies(k).values

y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

y_train.shape
y_test.shape

# Otrzymaliśmy one-hot encoding, ale z wartościami True/False.
y_train[0]

y_train = y_train.astype(int)
y_test = y_test.astype(int)


from keras.api.layers import Input, Dense, \
Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

from keras.api.models import Model
from keras.api.utils import plot_model

filter_cnt = 32
class_cnt  = y_train.shape[1]
kernel_size = (3,3)
act_func = 'selu'

# valid - rozmiary wyjścia są mniejsze od odpowiednich 
# rozmiarów wejścia (domyślne ustawienie).

# same – rozmiar wejścia jest równy rozmiarowi wyjścia: taki 
# efekt zapewnia uzupełnienie wejścia zerami.
conv_rule = 'same'

# (60000, 28, 28, 1)
x_train.shape

# (28, 28, 1)
x_train.shape[1:]

input_tensor = Input(x_train.shape[1:])
output_tensor = input_tensor
output_tensor.shape
# Parametr filters w warstwie Conv2D oznacza liczbę 
# filtrów konwolucyjnych, które zostaną zastosowane do 
# wejściowych danych, a wynik to 32 (filter_cnt) "mapy cech"
# (lub "obrazy"), każda stworzona przez konwolucję
# wejściowego obrazu z odpowiednim filtrem.

# Te 32 mapy cech można interpretować jako 32 różne 
# "widoki" danych wejściowych, gdzie każdy filtr wychwytuje 
# inny rodzaj wzorców, np. krawędzie, tekstury. 
output_tensor = Conv2D(filter_cnt, 
                       kernel_size,
                       padding= conv_rule,
                       activation = act_func)(output_tensor)
output_tensor.shape

# MaxPooling2D(2,2)
# Mapy cech zostają zmniejszone czterokrotnie pod względem 
# liczby piksel. Każdy blok 2x2 jest reprezentowany przez 
# jeden piksel (maksymalna wartość w bloku 2x2).
output_tensor = MaxPooling2D(2,2)(output_tensor)
output_tensor.shape

def my_act_func(tensor):
        return tf.math.tanh(tensor) * 0.5

output_tensor = Conv2D(filter_cnt, 
                       kernel_size,
                       padding='valid', 
                       activation = my_act_func)(output_tensor)
output_tensor.shape

# Average Pooling to operacja łączenia, która oblicza 
# średnią wartość elementów mapy cech (w bloku o rozmiarze
# pool_size) i wykorzystuje ją do 
# utworzenia mapy cech o zmniejszonym rozmiarze.
output_tensor = AveragePooling2D(2,2)(output_tensor)
output_tensor.shape

output_tensor = Conv2D(filter_cnt, 
                       kernel_size, 
                       padding='valid',
                       activation = act_func)(output_tensor)
output_tensor.shape

# GlobalAveragePooling2D()
# Dla danego elementu wejściowego, oblicz średnią
# w każdym kanale.
# wejście: (10,2,2,3) <-- 3 kanały
# wyjście: (10,3) 
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor.shape

output_tensor = Dense(class_cnt, activation = 'softmax')(output_tensor)
output_tensor.shape

ANN = Model(inputs = input_tensor, 
            outputs = output_tensor)

ANN.compile(loss = 'categorical_crossentropy', 
            metrics = ['accuracy'], 
            optimizer = 'adam')

plot_model(ANN, show_shapes=True)
############################################################
############################################################
# Opis działania warstwy GlobalAveragePooling2D()
# 
# Definicja tensoru wejściowego o wymiarach (3, 2, 2, 3)
# 3 elementy, każdy to macierz 2x2x3 (2x2, 3 kanały).
# Inaczej: każdy element macierzy 2x2 to wektor o 3 współrzędnych.
# (x,y,z).
input_tensor = np.array([
    # 1 element:
    # M(0,0,0) = [1,2,3]
    # M(0,1,0) = [1,2,3]
    # M(1,0,0) = [1,2,3]
    # M(1,1,0) = [1,2,3]
    [[[1, 2, 3], [1, 2, 3]],    
     [[1, 2, 3], [1, 2, 3]]],  
     
    [[[2, 5, 6], [2, 5, 6]],    # 2 element
     [[4, 5, 6], [4, 5, 6]]],  
     
    [[[7, 8, 9], [7, 8, 9]],    # 3 element
     [[7, 8, 9], [7, 8, 9]]]   
])

# Wymiary tensoru wejściowego
print(input_tensor.shape)

input_layer = Input(shape=(2, 2, 3))
gap_layer = GlobalAveragePooling2D()(input_layer)
model = Model(inputs=input_layer, outputs=gap_layer)

output_tensor = model.predict(input_tensor)
# Warstwa GlobalAveragePooling2D() dla 1 ELEMENTU obliczyła
# średnią współrzędnych x,y,z.
# Kanał 0 (współrzędna x): Wszystkie wartości wynoszą 1, 
# więc średnia wynosi:
#T(0,0)=(1+1+1+1)/4=1

# Kanał 1 (współrzędna y): Wszystkie wartości wynoszą 2,
# więc średnia wynosi:
# T(0,1)=(2+2+2+2)/4=2

# Kanał 2 (współrzędna z): Wszystkie wartości wynoszą 3, 
# więc średnia wynosi:
# T(0,2)=(3+3+3+3)/4=3

# Dla 2 ELEMENTU.
# Kanał 0 (współrzędna x): Wartości wynoszą 2, 2, 4, 4, 
# więc średnia wynosi:
#T(1,0)=(2+2+4+4)/4=3
print("Wymiary wyjścia:", output_tensor.shape)
print("Tensor wyjściowy (po GAP):\n", output_tensor)

############################################################
############################################################

############################################################
############################################################

# Efekt funkcji concatenate przedstawia
# rysunek 9.2 (skrypt s. 84).

# concatenate - łączy kolekcje danych/tensory 
# wzdłuż określonej osi.
# Przykład:

import tensorflow as tf

# Przykładowe tensory
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

print(tensor1)
print(tensor2)

# Łączenie wzdłuż osi 0 (wiersze)
result_0 = tf.concat([tensor1, tensor2],
                      axis=0)
print(result_0)
# W powyższym przykładzie tensory można 
# połączyć wzdłuż osi 0, jeśli liczba kolumn 
# (wymiar osi 1) jest taka sama w obu 
# tensorach.

# Łączenie wzdłuż osi 1 (kolumny)
result_1 = tf.concat([tensor1, tensor2], 
                     axis=1)
print(result_1)

# W powyższym przykładzie tensory można 
# połączyć wzdłuż osi 1, jeśli liczba 
# wierszy (wymiar osi 0) jest taka sama.

tensor3 = tf.constant([[1, 2, 5], [3, 4, 5]])
tensor4 = tf.constant([[5, 6], [7, 8]])

# Liczba wierszy się zgadza, więc tensory
# można połączyć wzdłuż linii kolumn.
result_2 = np.concatenate([tensor3, tensor4],
                          axis = 1)

print(result_2)

# Liczba kolumn jest różna, więc tensorów
# nie można połączyć wzdłuż linii wierszy.
# result_2 = np.concatenate([tensor3, tensor4],
#                          axis = 0)

############################################################
############################################################


from keras.api.layers import Lambda
from keras.api.layers import concatenate # Funkcja
from keras.api.layers import Concatenate # Warstwa
import tensorflow as tf

def add_inseption_module(input_tensor):
        act_func = 'relu'
        paths = [
        [Conv2D(filters = 64, kernel_size=(1,1), 
                padding='same', activation=act_func) ],

        [Conv2D(filters = 96, kernel_size=(1,1),
                 padding='same', activation=act_func),
        Conv2D(filters = 128, kernel_size=(3,3),
                padding='same', activation=act_func) ],

        [Conv2D(filters = 16, kernel_size=(1,1), 
                padding='same', activation=act_func),
        Conv2D(filters = 32, kernel_size=(5,5), 
               padding='same', activation=act_func) ],

        [MaxPooling2D(pool_size=(3,3), strides = 1, 
                      padding='same'),
        Conv2D(filters = 32, kernel_size=(1,1), 
               padding='same', activation=act_func) ]
               ]

        for_concat = []
        for path in paths:
                output_tensor = input_tensor
                for layer in path:
                        output_tensor = layer(output_tensor)

                for_concat.append(output_tensor)
        #return concatenate(for_concat)
        # Jeżeli korzystamy z warstwy Concatenate
        return for_concat

output_tensor = input_tensor = Input(x_train.shape[1:])

insept_module_cnt = 2
for i in range(insept_module_cnt):
        # output_tensor = add_inseption_module(output_tensor)
        output_tensor = Concatenate()(add_inseption_module(output_tensor))

output_tensor = Lambda(lambda tensor: tf.math.log(tensor))(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)

ANN = Model(inputs = input_tensor, outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy', 
            metrics = ['accuracy'], optimizer = 'adam')

plot_model(ANN)
plot_model(ANN, show_shapes=True)