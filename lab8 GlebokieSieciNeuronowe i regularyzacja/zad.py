# Importowanie niezbędnych bibliotek z Keras i innych
from keras.api.regularizers import l2, l1
from keras.api.layers import Dense, BatchNormalization
from keras.api.layers import Dropout, GaussianNoise
from keras.api.layers import LayerNormalization
from keras.api.models import Sequential
from keras.api.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie danych z pliku Excel
data = pd.read_excel('loan_data.xlsx')

# Pobranie listy kolumn z danych
columns = list(data.columns)

# Zamiana wartości w kolumnie 'Gender' na wartości binarne (1 dla 'Female', 0 dla 'Male')
mask = (data['Gender'].values == 'Female')
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0

# One-hot encoding dla kolumny 'Property_Area'
one_hot = pd.get_dummies(data.Property_Area)
one_hot = one_hot.astype(int)

# Dodanie zakodowanych kolumn do oryginalnych danych i usunięcie kolumny 'Property_Area'
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

# Zamiana wartości w kolumnie 'Married' na wartości binarne
data['Married'], _ = pd.factorize(data["Married"])

# Zamiana wartości w kolumnie 'Self_Employed' na wartości binarne
data['Self_Employed'], _ = pd.factorize(data["Self_Employed"])

# Zamiana wartości w kolumnie 'Education' na wartości binarne (1 dla 'Graduate', 0 dla 'Not Graduate')
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})

# Zamiana wartości w kolumnie 'Loan_Status' na wartości binarne (1 dla 'Y', 0 dla 'N')
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})

# Konwersja danych do typu float64
vals = data.values.astype(np.float64)

# Podział danych na cechy (X) i etykiety (y)
X = vals[:, :-1]
y = vals[:, -1]

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=False)

# Definiowanie parametrów modelu
neuron_num = 64  # Liczba neuronów w warstwie
do_rate = 0.5  # Współczynnik Dropout
noise = 0.1  # Współczynnik szumu Gaussowskiego
learning_rate = 0.001  # Współczynnik uczenia

# Tworzenie modelu sekwencyjnego
model = Sequential()

# Dodanie pierwszej warstwy Dense z regularizacją L2
model.add(Dense(neuron_num, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))

# Dropout – warstwa, w której zostają deaktywowane losowo wybrane neurony – ma
#   znaczenie regularyzacyjne,
# GaussianNoise – warstwa, w której do wyjść poprzedniej warstwy zostaje dodany
#   szum o rozkładzie Gaussowskim, posiadający zdefiniowane: średnią oraz odchylenie
#   standardowe, ma znaczenie regularyzacyjne,
# BarchNormalization – warstwa, w której obserwacje jednej porcji (batch) zostają
#   znormalizowane w sposób podobny do normalizacji Z-Score (odejmowanie średniej
#   oraz dzielenie przez odchylenie standardowe) - pozwala na przyspieszenie uczenia,
# LayerNormalization – warstwa, w której wyjścia jednej warstwy zostają
#   znormalizowane w sposób podobny do normalizacji Z-Score (odejmowanie średniej
#   oraz dzielenie przez odchylenie standardowe),

# Definiowanie bloku warstw
block = [
    Dense,
    LayerNormalization,
    BatchNormalization,
    Dropout,
    GaussianNoise
]

# Definiowanie argumentów dla każdej warstwy w bloku
args = [
    (neuron_num, 'selu'),  # Dense layer z 64 neuronami i aktywacją 'selu'
    (),  # LayerNormalization bez dodatkowych argumentów
    (),  # BatchNormalization bez dodatkowych argumentów
    (do_rate,),  # Dropout z współczynnikiem 0.5
    (noise,)  # GaussianNoise z współczynnikiem 0.1
]

# Ponowne tworzenie modelu sekwencyjnego
model = Sequential()

# Dodanie pierwszej warstwy Dense bez regularizacji
model.add(Dense(neuron_num, activation='relu', input_shape=(X.shape[1],)))

# Dodanie bloku warstw do modelu, powtarzając go dwa razy
repeat_num = 2
for i in range(repeat_num):
    for layer, arg in zip(block, args):
        model.add(layer(*arg))

# Dodanie ostatniej warstwy Dense z aktywacją 'sigmoid' do klasyfikacji binarnej
model.add(Dense(1, activation='sigmoid'))

# Kompilacja modelu z optymalizatorem Adam, funkcją straty 'binary_crossentropy' i metrykami 'accuracy', 'Recall', 'Precision'
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])

# Trenowanie modelu przez 100 epok z batch_size 32
epochs = 100
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=32
)

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

# Wywołanie funkcji do wizualizacji metryk modelu
plot_model_metrics(model)

# Import necessary libraries
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Input
from keras.optimizers import Adam

# Function to plot confusion matrix for neural network predictions
def cm_for_nn(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Get predictions from the model
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to classes (0 or 1)

    cm = confusion_matrix(y_test, y_pred_classes)  # Compute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Plot heatmap of confusion matrix
    plt.xlabel('Predicted Values')  # Label for x-axis
    plt.ylabel('True Values')  # Label for y-axis
    plt.title('Confusion Matrix')  # Title of the plot
    plt.show()

# Call function to display confusion matrix
cm_for_nn(model, X_test, y_test)

# Building and testing a neural network model with a different architecture

learning_rate = 0.001  # Learning rate for the optimizer

# Create a sequential model
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (sigmoid for binary classification)

# Compile the model with Adam optimizer and binary crossentropy loss function
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=2)

# Plot model performance metrics
plot_model_metrics(model, epochs=100)

# Testing different neural network architectures

neuron_num = 64  # Number of neurons in each layer
dropout_rate = 0.2  # Dropout rate to prevent overfitting
noise_factor = 0.1  # Noise factor for Gaussian noise layer
learning_rate = 0.001  # Learning rate

# Define the types of layers to be used in the model
block = [Dense, Dropout, GaussianNoise]

# Arguments for each layer type
args = [(neuron_num, 'relu'), (dropout_rate,), (noise_factor,)]

# Create a sequential model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))  # Input layer

# Add layers to the model in a loop
repeat_num = 5
for i in range(repeat_num):
    for layer, arg in zip(block, args):
        model.add(layer(*arg))

# Add the output layer (sigmoid for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=1)

# Plot model performance metrics
plot_model_metrics(model, epochs=100)
