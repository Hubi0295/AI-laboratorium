import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_iris
from keras.api.models import Sequential
from keras.api.layers import Input, Dense
from keras.api.optimizers import Adam, RMSprop, SGD
from keras.api.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
# Ustawienia parametrów wykresów
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
np.set_printoptions(precision=2)
# Tworzenie siatki punktów
x = np.arange(0,1,0.01)
y = x.copy()
X,Y = np.meshgrid(x,y)
# Definiowanie wag i obliczanie sumy ważonej
wx = 0.1
wy = 0.3
S = wx*X+wy*Y
# Generowanie wyjścia na podstawie progu
out = S>0.15
# Tworzenie wykresu
fig, ax = plt.subplots(1,1)
ax.imshow(out)
# Ustawienia osi wykresu
ticks = np.around(np.arange(-0.2,1.1,0.2), 3)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
plt.gca().invert_yaxis()
############################################

# Ładowanie danych z zestawu Iris
data = load_iris()
y = data.target
X = data.data
# Konwersja etykiet na kategorie i kodowanie one-hot
y = pd.Categorical(y)
y = pd.get_dummies(y).values
# Definiowanie liczby klas
class_num = y.shape[1]
# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(64, input_shape = (X.shape[1],), activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(class_num, activation = 'softmax'))
# Kompilacja modelu
learning_rate = 0.0001
model.compile(optimizer= Adam(learning_rate),loss='categorical_crossentropy',metrics=(['accuracy']))
# Wyświetlenie podsumowania modelu
model.summary()
# Zapisanie wizualizacji modelu do pliku
plot_model(model,to_file="my_model.png")

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Trenowanie modelu
model.fit(X_train, y_train, batch_size=32, epochs=100,validation_data=(X_test, y_test), verbose=2)
# Pobranie historii trenowania
historia = model.history.history
floss_train = historia['loss']
floss_test = historia['val_loss']
acc_train = historia['accuracy']
acc_test = historia['val_accuracy']
# Wizualizacja funkcji strat i dokładności
fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 100)
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].set_xlabel('Epoki')
ax[0].set_ylabel('Wartość funkcji strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()
fig, ax = plt.subplots(1, 1, figsize=(20,10))
epochs = np.arange(0, 100)
ax.plot(epochs, floss_train, label='floss_train', color='red')
ax.plot(epochs, floss_test, label='floss_test', color='blue')
ax.set_title('Funkcje strat')
ax.legend()

############################################

# Ponowny podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
accs = []
# Normalizacja danych
scaler = StandardScaler()
# Kroswalidacja z użyciem KFold
for train_index, test_index in KFold(5).split(X_train):
    X_train_cv = X_train[train_index,:]
    X_test_cv = X_train[test_index,:]
    y_train_cv = y_train[train_index,:]
    y_test_cv = y_train[test_index,:]
    X_train_cv = scaler.fit_transform(X_train_cv)
    X_test_cv = scaler.transform(X_test_cv)
        # Trenowanie modelu na danych kroswalidacyjnych
    model.fit(X_train, y_train, batch_size=32,epochs=10, validation_data=(X_test, y_test),verbose=2)
