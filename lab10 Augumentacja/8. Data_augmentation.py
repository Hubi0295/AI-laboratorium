from keras.api.datasets import fashion_mnist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Przeczytaj najpierw opis zbioru fashion_mnist, który
# znajduje się w opisie funkcji load_data().
(X_train, y_train), (X_test, y_test)  = fashion_mnist.load_data()
# 60'000 obrazów w skali szarości o wymiarach 28x28
X_train.shape
# 10'000 obrazów w skali szarości o wymiarach 28x28
y_train.shape

# Dodanie dodatkowego wymiaru (kanał obrazu)
# Jeśli zamierzasz rozszerzyć model o warstwy konwolucyjne (np. Conv2D), 
# to np.expand_dims stanie się potrzebne, ponieważ warstwy konwolucyjne 
# oczekują trójwymiarowych danych wejściowych (wysokość x szerokość x liczba_kanałów).
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

# 60'000 elementów, gdzie każdy element to macierz 28x28 plus jedna komórka
X_train.shape
# 10'000 elementów, gdzie każdy element to macierz 28x28 plus jedna komórka
X_test.shape

# (28, 28) to wymiary obrazu (szerokość i wysokość w pikselach).
# 1 oznacza liczbę kanałów (dla obrazów w skali szarości mamy
# jeden kanał; dla obrazów kolorowych byłoby 3, np. RGB).

# Bez tej informacji TensorFlow/Keras nie byłby w stanie 
# prawidłowo ustawić filtrów w warstwach konwolucyjnych czy dense.

# Co przedstawia 9 początkowych obrazów?
y_train[0:9]
# Według opisu zbioru
#  The classes are:
#
#    | Label | Description |
#    |:-----:|-------------|
#    |   0   | T-shirt/top |
#    |   1   | Trouser     | Spodnie
#    |   2   | Pullover    | Sweter
#    |   3   | Dress       | Sukienka
#    |   4   | Coat        | Płaszcz
#    |   5   | Sandal      | Sandały
#    |   6   | Shirt       | Koszula
#    |   7   | Sneaker     |
#    |   8   | Bag         | Tora
#    |   9   | Ankle boot  | Buty za kostkę?

plt.imshow(X_train[0,:,:,0], cmap='gray')
plt.title("Obraz o indeksie 0")
plt.show()

# 0 to indeks obrazu, który chcemy wyświetlić,
# : oznacza wybór wszystkich wierszy i kolumn,
# 0 na końcu usuwa ostatni wymiar (kanał), 
# dzięki czemu otrzymujemy czystą macierz o wymiarach (28, 28).

y_train[0:9]
fig, axs = plt.subplots(3,3)
axs[0,0].imshow(X_train[0,:,:,0], cmap='gray')
axs[0,1].imshow(X_train[1,:,:,0], cmap='gray')
axs[0,2].imshow(X_train[2,:,:,0], cmap='gray')
axs[1,0].imshow(X_train[3,:,:,0], cmap='gray')
axs[1,1].imshow(X_train[4,:,:,0], cmap='gray')
axs[1,2].imshow(X_train[5,:,:,0], cmap='gray')
axs[2,0].imshow(X_train[6,:,:,0], cmap='gray')
axs[2,1].imshow(X_train[7,:,:,0], cmap='gray')
axs[2,2].imshow(X_train[8,:,:,0], cmap='gray')
plt.show()

# Konwersja danych na typ danych kategoryczny. Typ kategoryczny jest 
# bardziej efektywny pod względem pamięci, szczególnie gdy dane zawierają 
# powtarzające się wartości.

k = pd.Categorical(y_train)
k.categories # Unikalne kategorie
k.codes      # Unikalne kategorie

# Po konwersji możesz uzyskać dostęp do:
# Unikalnych kategorii za pomocą ,,categories'',
# Kodów (indeksów przypisanych do kategorii) za pomocą ,,codes''.

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

y_train[0]
y_test[0]

from matplotlib import rcParams
rcParams['font.size'] = 48

def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5*arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic, cmap = 'gray')
        axis.axis('off')
    fig.tight_layout()
    return fig

org_images = X_train[:10,:,:,0]
org_images.shape

show_pictures(org_images).suptitle("Zdjęcia pierwotne")

# : wybiera wszystkie wiersze.
# ::-1 odwraca kolejność kolumn.

odbicia_poziome = org_images[:,:,::-1]
odbicia_poziome.shape
show_pictures(odbicia_poziome).suptitle("Odbicia poziome")

# ::-1 odwraca kolejność wierszy.
# : wybiera wszystkie kolumny.

odbicia_pionowe = org_images[:,::-1,:]
odbicia_pionowe.shape
show_pictures(odbicia_pionowe).suptitle("Odbicia pionowe")

# Składnia: start:stop:step
# Indeksowanie tablicy w NumPy za pomocą : pozwala wybrać podzbiór 
# danych w danym wymiarze. step określa krok, w jakim elementy są 
# wybierane. Gdy step = -1, NumPy przegląda elementy w odwrotnej kolejności.


# Odbicie poziome:
# Wystarczy odwrócić kolejność kolumn (w naszym przypadku to trzeci wymiar).

# Odbicie pionowe:
# Wystarczy odwrócić kolejność wierszy (drugi wymiar).

# PIL - Python Imaging Library
# Wymagana może być biblioteka Pillow
from PIL import Image

rotated_images = org_images.copy()
img_size = org_images.shape[1:]
img_size

angles = np.random.randint(-30,30, len(rotated_images))
for i, img in enumerate(rotated_images):
    img = Image.fromarray(img).rotate(angles[i], expand = True).resize(img_size)
    rotated_images[i] = np.array(img)

show_pictures(rotated_images).suptitle("Obrazy zmodyfikowane")

# Image.fromarray(img).rotate(angles[i], expand = True).resize(img_size)
# Najpierw tworzymy obiekt typu Image na podstawie macierzy
# pikseli. Następnie dokonujemy rotacji obrazu ze wskazaniem
# na rozszerzenie rozmiaru, jeżeli zachodzi taka konieczność.
# Na koniec skalujemy obraz do pierwotnych wymiarów, 
# ponieważ rozmiar obrazu może ulec zmianie.


crop_images = org_images.copy()
img_size = org_images.shape[1:]
for i, img in enumerate(crop_images):
    # Return random integers from low (inclusive)
    # to high (exclusive).
    # Trzeci parametr to liczba wylosowanych liczb
    left, upper = np.random.randint(0, 5, 2)
    right, lower = np.random.randint(23, 28, 2)
    img = Image.fromarray(img).crop((left, upper, right, lower)).resize(img_size)
    crop_images[i] = np.array(img)

show_pictures(crop_images).suptitle("Przycięte obrazy")



# ########################################################### #
#                           AUTOKODER
# ########################################################### #

# Autoenkoder składa się z dwóch głównych komponentów:
#  kodera (encoder) i dekodera (decoder).

# Wyobraź sobie sytuację:
# Masz rysunek o wymiarach 28×28 pikseli – to mały obrazek cyfry napisanej odręcznie,
# np. 5. Ten obrazek to 784 liczby (28 * 28), ponieważ każdy piksel ma jakąś 
# wartość (od 0 do 255).

# Twoim celem jest nauczenie się uproszczonej wersji tego obrazka, która zajmuje 
# mniej miejsca, ale pozwala później odtworzyć obraz jak najdokładniej.
# Jak działa autoenkoder?

# Kompresja (kodowanie):
#   Autoenkoder próbuje znaleźć najważniejsze cechy obrazu, takie jak kształt 
#   cyfry 5. Ignoruje szczegóły, które nie są istotne (np. szum lub 
#   drobne nierówności w linii). Dzięki temu zapisuje uproszczoną wersję obrazu, 
#   np. jako wektor o długości 64 (zamiast 784 liczb).

# Rekonstrukcja (dekodowanie):
#   Na podstawie tej uproszczonej wersji (64 liczby) autoenkoder próbuje 
#   odtworzyć oryginalny obraz. Celem jest, aby obraz wynikowy był 
#   jak najbardziej podobny do oryginału.


from keras.api.models import Model
from keras.api.layers import Input, Dense, Dropout, Reshape, \
    BatchNormalization, Lambda
from keras.api.optimizers import Adam
from keras.api.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

act_func = 'selu'
hidden_dims = 64

# ########################################################### #
#                           ENKODER
# ########################################################### #
encoder_layers = [
    Reshape((28*28,)),
    BatchNormalization(), # Powoduje przeskalowanie i przesunięcie danych (np. może powstać wartość -0.3)
    Dense(512,activation=act_func),
    Dense(128,activation=act_func),
    Dense(hidden_dims, activation=act_func)]

encoder_input = Input(shape = X_train.shape[1:])
x = encoder_input

# Forward pass. Przejście po wszystkich warstwach w celu ustalenia
# rozmiaru i typu wyjścia.
for layer in encoder_layers:
    x = layer(x)

encoder_output = x
encoder = Model(inputs=encoder_input, outputs=encoder_output)
encoder.summary()
# W opisie klasy Model możemy przeczytać:
# There are three ways to instantiate a Model:
# 1. With the "Functional API":
# You start from Input, you chain layer calls to specify
# the model's forward pass, and finally, you create your model from 
# inputs and outputs:
#
#   inputs = keras.Input(shape=(37,))
#   x = keras.layers.Dense(32, activation="relu")(inputs)
#   outputs = keras.layers.Dense(5, activation="softmax")(x)
#   model = keras.Model(inputs=inputs, outputs=outputs)


# Encoder  
# Input (28,28)
# 28*28=784
# 512
# 128
# 64

# Decoder
# Input (64,)
# 128
# 512
# 28*28=784
# Output (28,28)

# ########################################################### #
#                           DEKODER
# ########################################################### #

decoder_layers = [
    Dense(128,activation=act_func),
    Dense(512,activation=act_func),
    Dense(28*28,activation='sigmoid'), # wynik z przedziału [0,1]
    Lambda(lambda x: x*255), # wynik z przedziału [0,255]
    Reshape((28,28))]

decoder_input = Input(shape=encoder_output.shape[1:])
x = decoder_input

encoder_output.shape
decoder_input.shape

for layer in decoder_layers:
    x = layer(x)

decoder_output = x

decoder = Model(inputs = decoder_input, outputs = decoder_output)
decoder.summary()

# ########################################################### #
#                           AUTOKODER
# ########################################################### #

aec_output = decoder(encoder(encoder_input))

gen_autoencoder = Model(inputs = encoder_input, outputs = aec_output)
learning_rate = 0.001
gen_autoencoder.compile(optimizer = Adam(learning_rate), 
                        loss = 'MeanSquaredError')

# W autoenkoderze zarówno dane wejściowe (x) jak i dane wyjściowe (y) 
# są takie same, ponieważ celem autoenkodera jest odtworzenie danych wejściowych 
# po ich skompresowaniu do przestrzeni latentnej i ponownym zrekonstruowaniu.
gen_autoencoder.fit(x=X_train,
                    y=X_train, 
                    validation_data=(X_test, X_test), 
                    batch_size=256,
                    epochs=25)

# Uwaga.
# W modelu autokodera wynikiem jest obraz, gdzie każdy piksel
# to liczba z przedziału 0 - 255 (patrz warstwa: Lambda(lambda x: x*255)).
# Zatem jeżeli w oryginalnym obrazie dany piksel ma wartość 255,
# a w obrazie po rekonstrukcji ma wartość 0, to błąd wynosi 255.
# Stąd wartość ,,loss'' będzie raczej znacznie większa od
# wartości ,,loss'' w modelach dotychczas budowanych.



# Przewidywanie dla danych testowych
reconstructed = gen_autoencoder.predict(X_test)
# Instrukcja gen_autoencoder.predict(X_test)
# automatycznie wykonuje zarówno etap kodowania, jak i dekodowania.

rcParams['font.size'] = 12
# liczba obrazów do wyświetlenia
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    # Oryginał
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title("Oryginał")
    plt.axis('off')

    # Rekonstrukcja
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i], cmap='gray')
    plt.title("Rekonstrukcja")
    plt.axis('off')

plt.show()





# W najnowszej wersji TensorFlow 2.x, wiele funkcji, które wcześniej były dostępne 
# w Keras Backend, zostało przeniesionych do TensorFlow. 
import tensorflow as tf
#tf.keras.activations.relu
#tf.keras.activations.tanh

def adding_noise(tensor):
    noise = tf.random.normal(shape=(tf.shape(tensor)), mean=0, stddev=0.5)
    return tensor + noise

# Lambda(adding_noise) - utworzenie warstwy z niestandardową funkcją aktywacji
# Lambda(adding_noise)(encoder_output) - wywołanie niestandardowej funkcji aktywacji.
# Pomimo, że encoder_output nie zawiera żadnych danych (obrazy), to poniższa instrukcja
# tworzy graf obliczeniowy. Keras (i TensorFlow, na którym oparty jest Keras) buduje wewnętrzny graf operacji, który 
# będzie później używany do przepływu danych przez sieć.
noised_encoder_output = Lambda(adding_noise)(encoder_output)
# Inaczej: wyjście encodera jest podpięte do warstwy Lambda.
augmenter_output = decoder(noised_encoder_output)
# Wyjście warstwy Lambda jest podpięte do dekodera.

augmenter = Model(inputs = encoder_input, outputs = augmenter_output)
augmenter.compile(optimizer=Adam(learning_rate),loss='MeanSquaredError')
augmenter.fit(x=X_train,  
            y=X_train,  
            validation_data=(X_test, X_test),      
            batch_size=256,              
            epochs=25)

# Uwaga.
# W modelu autokodera wynikiem jest obraz, gdzie każdy piksel
# to liczba z przedziału 0 - 255 (patrz warstwa: Lambda(lambda x: x*255)).
# Zatem jeżeli w oryginalnym obrazie dany piksel ma wartość 255,
# a w obrazie po rekonstrukcji ma wartość 0, to błąd wynosi 255.
# Stąd wartość ,,loss'' będzie raczej znacznie większa od
# wartości ,,loss'' w modelach dotychczas budowanych.

def filter_data(data, iteration_num):
    augmented_data = data.copy()
    for i in range(iteration_num):
        augmented_data = gen_autoencoder.predict(augmented_data)
    return augmented_data


for i in range(5):
    test_for_augm = X_train[i*10:i*10+10,...]
    augmented_data = test_for_augm.copy()
    show_pictures(test_for_augm)

    augmented_data = augmenter.predict(augmented_data)
    show_pictures(augmented_data)

    augmented_data = filter_data(augmented_data, 5)
    show_pictures(augmented_data)
    plt.show()



# (Niestety) PRZESTARZAŁE! 
from tensorflow.keras.preprocessing.image \
         import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

obr = x_train[0,:,:]
plt.imshow(obr,cmap='gray')
plt.show()

obr = np.expand_dims(obr, axis= -1)

data_gen = ImageDataGenerator(
    rotation_range=30,        # Obrót do 30 stopni
    width_shift_range=0.2,    # Przesunięcie w poziomie
    height_shift_range=0.2,   # Przesunięcie w pionie
    shear_range=0.2,          # Ścinanie
    zoom_range=0.2,           # Zoom
    horizontal_flip=True,     # Odbicie poziome
    fill_mode='nearest' 
)

img_gen = data_gen.flow(
    np.expand_dims(obr, axis = 0),
    batch_size = 1
)

obrazki = np.zeros((10,28,28))

for i in range(10):
    img = next(img_gen)[0]
    obrazki[i] = img[:,:,0];

show_pictures(obrazki)
plt.show()



