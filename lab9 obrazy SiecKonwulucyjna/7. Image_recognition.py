from keras.api.layers import Conv2D, Flatten, Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import Adam, SGD, Nadam
from keras.api.datasets import mnist

from keras.api.layers import AveragePooling2D, MaxPooling2D

from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
# list(colormaps) 

# Nadam (Nesterov-accelerated Adaptive Moment Estimation)

# Wczytanie zbioru danych MNIST (zbiór obrazów cyfr pisma odręcznego)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Informacje o rozmiarze obrazów
x_train[0].shape
print(f"Każdy obraz w zbiorze MNIST reprezentowany jest\
      w postaci macierz {x_train[0].shape}")

# Wyświetlenie kilku elementów
plt.imshow(x_train[0])
plt.show()
print(f"Cyfra na obrazku to {y_train[0]}")
plt.imshow(x_train[1])
plt.show()
print(f"Cyfra na obrazku to {y_train[1]}")

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Efekt powyższej operacji
x_train[0].shape
x_test[2].shape


# (28, 28) to wymiary obrazu (szerokość i wysokość w pikselach).
# 1 oznacza liczbę kanałów (dla obrazów w skali szarości mamy
# jeden kanał; dla obrazów kolorowych byłoby 3, np. RGB).

# Bez tej informacji TensorFlow/Keras nie byłby w stanie 
# prawidłowo ustawić filtrów w warstwach konwolucyjnych czy dense.

# Liczba wartosci unikatowych w y_train, inaczej
# liczba roznych cyfr znajdujacych sie na obrazach.
class_cnt = np.unique(y_train).shape[0]
class_cnt

# cnt - count (liczba/ilosc)

# Liczba filtrów
filter_cnt = 32
# Liczba neuronow
neuron_cnt = 32
# Tempo uczenia
learning_rate = 0.0001
# Funkcja aktywacji
act_func = 'relu'
# Rozmiar filtru, inaczej rozmiar jądra filtru
kernel_size = (3,3)
# Utworzenie modelu sekwencyjnego
model = Sequential()

conv_rule = 'same'

# Zwróć uwagę na rezultat poniższych funkcji
x_train.shape[:]
x_train.shape[1:]

# Dodanie pierwszej warstwy wejściowej
model.add(Input(shape=x_train.shape[1:]))
# Dodanie kolejnej warstwy będącej warstwą konwolucyjną
model.add(Conv2D( filters=filter_cnt,
                  kernel_size = kernel_size, 
                  padding = conv_rule,  # wypełnienie
                  activation = act_func))

# Parametr filters w warstwie Conv2D oznacza liczbę 
# filtrów konwolucyjnych, które zostaną zastosowane do 
# wejściowych danych, a wynik to 32 (filter_cnt) "mapy cech"
# (lub "obrazy"), każda stworzona przez konwolucję
# wejściowego obrazu z odpowiednim filtrem.

# Te 32 mapy cech można interpretować jako 32 różne 
# "widoki" danych wejściowych, gdzie każdy filtr wychwytuje 
# inny rodzaj wzorców, np. krawędzie, tekstury lub bardziej 
# złożone struktury (w głębszych warstwach).

# Dodanie kolejnej warstwy (warstwy spłaszczającej)
model.add(Flatten())
# Dodanie warstwy wyjściowej
model.add(Dense(class_cnt, activation='softmax'))

model.compile(optimizer=Adam(learning_rate),
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])
# W celu monitorowania dokładności przewidywań, aby
# łatwiej było ocenić jakość modelu, można ustawić dodatkowe
# metryki

model.fit(x_train, 
          y_train, 
          epochs = class_cnt , 
          validation_data=(x_test, y_test))

# ################################################ #
# ################################################ #
def plot_model_metrics(model, epochs=1000):
      historia = model.history.history
      floss_train = historia['loss']
      floss_test = historia['val_loss']
      acc_train = historia['accuracy']
      acc_test = historia['val_accuracy']

      fig,ax = plt.subplots(1,2, figsize=(20,10))
      epo = np.arange(0, epochs)
      ax[0].plot(epo, floss_train, label = 'floss_train')
      ax[0].plot(epo, floss_test, label = 'floss_test')
      ax[0].set_title('Funkcje strat')
      ax[0].legend()
      ax[1].set_title('Dokladnosci')
      ax[1].plot(epo, acc_train, label = 'acc_train')
      ax[1].plot(epo, acc_test, label = 'acc_test')
      ax[1].legend()
      plt.show()

def cm_for_nn(model, X_test, y_test):
    # y_pred jest 10 wymiarowym wektorem, będącym rozkładem
    # prawdopodobieństwa (softmax w ostatniej warstwie)
    y_pred = model.predict(X_test)
    # Znajdź w każdym wierszu macierzy y_pred indeks
    # elementu, który zawiera największą wartość i zwróć
    # numer indeksu.
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.xlabel('Wartosci przewidziane')
    plt.ylabel('Wartości rzeczywiste')
    plt.title('Confusion Matrix')
    plt.show()
# ################################################ #
# ################################################ #
plot_model_metrics(model, class_cnt)
cm_for_nn(model, x_test, y_test)


# model.evaluate() na danych testowych, aby 
# obliczyć stratę i metryki określone w model.compile()
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f} %")

# ################################################ #
# ################################################ #


filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3,3)
pooling_size = (2,2)
conv_rule = 'same'

model = Sequential()

model.add(Input(shape = x_train.shape[1:]))
     
model.add(Conv2D(filters=filter_cnt, 
                 kernel_size = kernel_size, 
                 padding = conv_rule, 
                 activation = act_func))

model.add(MaxPooling2D(pooling_size))
# Agreguje dane w lokalnych regionach przez wybranie
# maksymalnej wartości z każdego okna 2x2, co zmniejsza 
# wymiary przestrzenne map cech (wysokość i szerokość) o połowę.

# Mapy cech zostają zmniejszone czterokrotnie pod względem 
# liczby pikseli (każdy blok 2x2 jest reprezentowany przez 
# jeden piksel).

model.add(Flatten())

# Aby funkcja SparseCategoricalCrossentropy działała prawidłowo, wyniki z 
# ostatniej warstwy modelu powinny być prawdopodobieństwami dla każdej 
# klasy. Stąd najlepiej zastosować 'softmax'.
model.add(Dense(class_cnt, activation='softmax'))

model.compile(optimizer=Adam(learning_rate), 
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])

# Funkcja straty ,,SparseCategoricalCrossentropy'' jest stosowana, gdy 
# etykiety (np. y_train, y_test) są podane jako liczby całkowite 
# reprezentujące klasy (np. 0, 1, 2, ..., n-1), a nie jako wektory "one-hot".

# Etykiety dla SparseCategoricalCrossentropy: [0, 1, 2]
# Etykiety dla CategoricalCrossentropy (one-hot): [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Wyświetlenie informacji o utworzonym modelu
model.summary()

model.fit(x_train, 
          y_train, 
          epochs = class_cnt , 
          validation_data=(x_test, y_test))

plot_model_metrics(model, class_cnt)
cm_for_nn(model, x_test, y_test)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f} %")


# Próba odtworzenia sieci neuronowej ze strony 
# https://adamharley.com/nn_vis/cnn/3d.html

# Keras (i ogólnie w TensorFlow) można utworzyć własną 
# funkcję aktywacji i przekazać ją do modelu (warstwy).

# Przykład budowy własnej funkcji aktywacji
from keras.api.activations import relu
from keras.api.activations import tanh

# ReLU plus stała 0.5
def custom_activation(x):
    return relu(x) + 0.5  

# W Kerasie możesz także użyć warstwy Lambda, aby zdefiniować 
# niestandardową funkcję aktywacji w miejscu, bez konieczności 
# tworzenia osobnej funkcji.
from keras.api.layers import Lambda, Resizing
from keras.api.optimizers import Adadelta, Adagrad, AdamW


cnn = Sequential()

cnn.add(Input(shape=x_train.shape[1:]))

cnn.add(Resizing(32, 32))  # Zmiana rozmiaru na 32x32

cnn.add(Conv2D(filters=6,
               kernel_size=(5,5),
               padding="valid",
               activation=None))

# Powyższa warstwa nie ma przypisanej żadnej funkcji aktywacji.
# Wyjście tej warstwy jest po prostu liniowe (warstwa przepuszcza dane).

# Ponieważ w Conv2D nie określono funkcji aktywacji, wyjście z warstwy 
# konwolucyjnej jest po prostu liniowe, tzn. jest to wynik działania 
# filtrów na obrazie wejściowym.

# Dodanie warstwy z niestandardową funkcją aktywacji. Poniższa funkcja
# działa na wyjściu warstwy poprzedniej, która przepuszcza dane. Stąd
# całość zachowuje się tak, jakby 1.7159*tanh(0.666667*x) była funkcją
# aktywacji warstwy Conv2D
cnn.add(Lambda(lambda x: 1.7159*tanh((2.0/3.0)*x)))

cnn.add(MaxPooling2D(pool_size=(2,2)))
# Pooling operuje na każdej mapie cech z osobna. 
# To znaczy, że dla każdej z 6 map cech zostanie przeprowadzona 
# operacja poolingowa, czyli redukcja wymiarów (downsampling) mapy cech.

cnn.add(Conv2D(filters=16,
               kernel_size=(5,5),
               padding="valid",
               activation=None))
# Liczba map cech, które stanowią wejście dla warstwy konwolucyjnej, zależy
# od liczby kanałów (lub map cech) wyjściowych z poprzedniej warstwy. Dla każdej 
# warstwy konwolucyjnej w Keras wejściem są wszystkie mapy cech wygenerowane 
# przez warstwy przed nią.

cnn.add(Lambda(lambda x: 1.7159*tanh((2.0/3.0)*x)))

cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Flatten())

cnn.add(Dense(120, activation=None))
cnn.add(Lambda(lambda x: 1.7159*tanh((2.0/3.0)*x)))

cnn.add(Dense(100, activation=None))
cnn.add(Lambda(lambda x: 1.7159*tanh((2.0/3.0)*x)))

# Aby funkcja SparseCategoricalCrossentropy działała prawidłowo, wyniki z 
# ostatniej warstwy modelu powinny być prawdopodobieństwami dla każdej 
# klasy. Stąd najlepiej zastosować 'softmax'.
cnn.add(Dense(10, activation='softmax'))

cnn.compile(optimizer=Nadam(learning_rate),
            loss="SparseCategoricalCrossentropy",
            metrics=['accuracy'])

cnn.summary()

cnn.fit(x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1)

plot_model_metrics(cnn, 10)
cm_for_nn(cnn, x_test, y_test)

# model.evaluate() na danych testowych, aby 
# obliczyć stratę i metryki określone w model.compile()
loss, accuracy = cnn.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f} %")



# Sparse Categorical Crossentropy
# Wzór:
# Loss = -1/N * sum_{i=1}^{N} log(p_{yi})
#
# Interpretacja:
# Suma logarytmów, gdzie argumentem jest przewidywane 
# prawdopodobieństwo.

# Przykład:
# Model przewiduje prawdopodobieństwa (po softmaxie) dla
# 3 klas: [0.1, 0.6, 0.3], gdzie poprawna predykcja to
# klasa 2 (indeksując od 0), wtedy strata wynosi
# Loss = - log(0.3) = - 1.204

# Przykład
# [0.25 0.3 0.45]  # poprawna klasa = 1
# [​0.7 0.15 0.15]  # poprawna klasa = 2
# [​0.2 0.3 0.5]​    # poprawna klasa = 2

# Loss = -1/3 * [ log(0.3) + log (0.15) + log(0.5)]