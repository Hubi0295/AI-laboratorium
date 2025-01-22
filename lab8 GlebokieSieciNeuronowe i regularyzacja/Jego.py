# Wczytanie potrzebnych bibliotek
from keras.api.layers import Dense, Input, BatchNormalization
from keras.api.layers import Dropout, GaussianNoise
from keras.api.layers import LayerNormalization
from keras.api.models import Sequential
from keras.api.optimizers import Adam, SGD
from keras.api.regularizers import l2, l1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Wczytanie danych z pliku Excel
data_const = pd.read_excel('loan_data.xlsx')
data = pd.read_excel('loan_data.xlsx')
data.head()

# Zamiana wartości w kolumnie 'Gender' na wartości binarne (1 dla 'Female', 0 dla 'Male')
# Wartości są True/False, a chcemy aby były 1/0
mask = (data['Gender'].values == 'Female')
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0

# One-hot encoding dla kolumny 'Property_Area'
one_hot = pd.get_dummies(data.Property_Area)
one_hot = one_hot.astype(int)

# Dodanie zakodowanych kolumn do oryginalnych danych i usunięcie kolumny 'Property_Area'
# Dołączenie nowych kolumn do oryginalnej ramki danych
data = pd.concat([data, one_hot], axis=1)
# Usunięcie oryginalnej kolumny Property_Area
data = data.drop(columns=['Property_Area'])
data.head()

# Zamiana wartości w kolumnie 'Married' na wartości binarne
# Przekształcenie pozostałych kolumn o typie kategorycznym
# factorize zwraca dwie wartości dlatego jest , _
data['Married'], _ = pd.factorize(data["Married"])

# Zamiana wartości w kolumnie 'Self_Employed' na wartości binarne
# factorize działa poprawnie dla Yes/No
data['Self_Employed'], _ = pd.factorize(data["Self_Employed"])

# Zamiana wartości w kolumnie 'Education' na wartości binarne (1 dla 'Graduate', 0 dla 'Not Graduate')
# Education zawiera tylko dwie wartości
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})
data.head()

# Przeniesienie kolumny 'Loan_Status' na koniec
ko = data['Loan_Status']
data = data.drop(columns=['Loan_Status'])
data['Loan_Status'] = ko

# Zamiana wartości w kolumnie 'Loan_Status' na wartości binarne (1 dla 'Y', 0 dla 'N')
# Przekształcenie na klasy (0 lub 1)
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})
data.head()

# Konwersja danych do typu float64
vals = data.values.astype(np.float64)

# Podział danych na cechy (X) i etykiety (y)
X = vals[:, :-1]
y = vals[:, -1]

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=False)
print(X.shape)
print(X_train.shape)

# Definiowanie parametrów modelu
neuron_num = 64  # Liczba neuronów
dropout_rate = 0.5  # Intensywność dropout (w %)
stddev = 0.1  # Odchylenie standardowe (krzywa gaussa)
learning_rate = 0.001  # Tempo uczenia

# Definiowanie bloku warstw
block = [Dense, LayerNormalization, BatchNormalization, Dropout, GaussianNoise]
args = [(neuron_num, 'relu'), (), (), (dropout_rate,), (stddev,)]

# Tworzenie modelu z Dropout
model_dropout = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dropout(0.5),  # Wyjaśnienie jak działa Dropout
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer (Warstwa wyjściowa)
])

# Tworzenie modelu z Batch Normalization
model_batch = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    BatchNormalization(),  # Wyjaśnienie jak działa Batch Normalization
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Output layer (Warstwa wyjściowa)
])

# Tworzenie modelu sekwencyjnego
model = Sequential()
model.add(Input(shape=(X.shape[1],)))  # Input Layer

# Liczba powtórzeń bloku warstw
repeat_num = 2

# Sieć będzie składać się z kilku identycznych bloków
for i in range(repeat_num):
    # Funkcja zip łączy w pary (a,b) elementy z podanych list, tworząc tym samym nową listę.
    for layer, arg in zip(block, args):
        # Dodanie warstwy do modelu
        model.add(layer(*arg))

# Output layer (Warstwa wyjściowa)
# Jeśli Twój problem jest binarny (np. klasy 0 i 1), powinieneś użyć funkcji aktywacji sigmoid w ostatniej warstwie.
model.add(Dense(1, activation='sigmoid'))

# Przed rozpoczęciem procesu uczenia model należy skompilować.
# Metrics jest listą!
model.compile(optimizer=Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall', 'Precision'])

# Trenowanie modelu
# Wersja verbose 1 (z paskiem postępu)
model.fit(X_train,
          y_train,
          batch_size=32,
          epochs=100,
          validation_data=(X_test, y_test),
          verbose=2)

# Funkcja do wizualizacji metryk modelu
def plot_model_metrics(model, epochs=100):
    # Pobranie historii trenowania
    historia = model.history.history
    floss_train = historia['loss']
    floss_test = historia['val_loss']
    acc_train = historia['accuracy']
    acc_test = historia['val_accuracy']

    # Tworzenie wykresów strat i dokładności
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    epo = np.arange(0, epochs)
    ax[0].plot(epo, floss_train, label='floss_train')
    ax[0].plot(epo, floss_test, label='floss_test')
    ax[0].set_title('Funkcje strat')
    ax[0].legend()
    ax[1].set_title('Dokładności')
    ax[1].plot(epo, acc_train, label='acc_train')
    ax[1].plot(epo, acc_test, label='acc_test')
    ax[1].legend()
    plt.show()

# Wywołanie funkcji do wizualizacji metryk modelu
plot_model_metrics(model, epochs=100)

# Importowanie dodatkowych bibliotek do tworzenia macierzy konfuzji
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Funkcja do tworzenia macierzy konfuzji dla modelu sieci neuronowej
def cm_for_nn(model, X_test, y_test):
    y_pred = model.predict(X_test)  
    y_pred_classes = (y_pred > 0.5).astype(int) 

    cm = confusion_matrix(y_test, y_pred_classes) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix') 
    plt.show()

# Wywołanie funkcji do tworzenia macierzy konfuzji
cm_for_nn(model, X_test, y_test)

# Definiowanie parametrów modelu
learning_rate = 0.001  

# Tworzenie modelu sekwencyjnego
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu')) 
model.add(Dense(64, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=2)

# Wywołanie funkcji do wizualizacji metryk modelu
plot_model_metrics(model, epochs=100)

# Definiowanie parametrów modelu
neuron_num = 64  # Liczba neuronów
dropout_rate = 0.2  # Intensywność dropout (w %)
noise_factor = 0.1  # Odchylenie standardowe (krzywa gaussa)
learning_rate = 0.001  # Tempo uczenia

# Definiowanie bloku warstw
block = [Dense, Dropout, GaussianNoise]
args = [(neuron_num, 'relu'), (dropout_rate,), (noise_factor,)]

# Tworzenie modelu sekwencyjnego
model = Sequential()
model.add(Input(shape=(X.shape[1],)))  # Input Layer

# Liczba powtórzeń bloku warstw
repeat_num = 5

# Sieć będzie składać się z kilku identycznych bloków
for i in range(repeat_num):
    # Funkcja zip łączy w pary (a,b) elementy z podanych list, tworząc tym samym nową listę.
    for layer, arg in zip(block, args):
        # Dodanie warstwy do modelu
        model.add(layer(*arg))

# Output layer (Warstwa wyjściowa)
# Jeśli Twój problem jest binarny (np. klasy 0 i 1), powinieneś użyć funkcji aktywacji sigmoid w ostatniej warstwie.
model.add(Dense(1, activation='sigmoid'))

# Przed rozpoczęciem procesu uczenia model należy skompilować.
# Metrics jest listą!
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
# Wersja verbose 1 (z paskiem postępu)
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=1)

# Wywołanie funkcji do wizualizacji metryk modelu
plot_model_metrics(model, epochs=100)