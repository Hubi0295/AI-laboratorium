import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import rcParams

# Wczytaj metadane
metadata = pd.read_csv('new_selected_samples.csv', sep=',')
path = 'voices'
files = os.listdir(path)
print(files)

# Ustawienia
fs = 16000  # częstotliwość próbkowania
seconds = 10  # oczekiwany czas trwania w sekundach
expected_length = fs * seconds  # oczekiwana długość sygnału (160000 próbek)

# Przygotowanie macierzy X_raw
X_raw = np.zeros((len(files), expected_length))
for i, file in enumerate(files):
    rate, data = wavfile.read(f"{path}/{file}")
    # Dopasowanie długości danych do oczekiwanej
    if len(data) > expected_length:
        X_raw[i, :] = data[:expected_length]  # Przytnij do 160000 próbek
    elif len(data) < expected_length:
        X_raw[i, :len(data)] = data  # Wypełnij zerami jeśli za krótkie
    else:
        X_raw[i, :] = data  # Długość pasuje

# Wczytaj etykiety
y = metadata

# --- Wizualizacja waveform i FFT ---

# Wizualizacja dla pierwszego pliku dźwiękowego
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Wyświetlenie waveform
ax[0].plot(np.arange(X_raw.shape[1]), X_raw[0, :])
ax[0].set_title('Waveform - pierwszy plik')
ax[0].set_xlabel('Czas [samples]')
ax[0].set_ylabel('Amplituda')

# Oblicz FFT
X_fft = np.abs(fft(X_raw, axis=-1)) / X_raw.shape[1]

# Wyświetlenie FFT dla pierwszego pliku
ax[1].scatter(np.arange(X_fft.shape[1]), X_fft[0, :], s=0.5)
ax[1].set_title('FFT - pierwszy plik')
ax[1].set_xlabel('Częstotliwość [Hz]')
ax[1].set_ylabel('Amplituda')

fig.tight_layout()
plt.show()

# --- Redukcja rozdzielczości FFT ---

# Redukcja rozdzielczości widma (średnia co 'mean_num' próbek)
mean_num = 10
X_fft = np.reshape(
    X_fft[:, : (X_fft.shape[1] // mean_num) * mean_num],
    (X_fft.shape[0], X_fft.shape[1] // mean_num, mean_num)
).mean(axis=-1)

# Wycięcie interesującego przedziału częstotliwości (50-280 Hz)
low_cut = 50
high_cut = 280
X_fft_cut = X_fft[:, low_cut:high_cut]  # Zakres od 50 Hz do 280 Hz

# Normalizacja amplitudy
X_fft_cut = X_fft_cut / np.expand_dims(X_fft_cut.max(axis=1), axis=-1)

# Wizualizacja dla pierwszego pliku (widmo)
freqs = np.arange(low_cut, high_cut)  # Oś częstotliwości
amplitudes = X_fft_cut[0, :]  # Normalizowane amplitudy pierwszego pliku

plt.figure(figsize=(8, 5))
plt.plot(freqs, amplitudes * 60, linewidth=1.5, color='blue')  # *60 dla zgodności ze skalą
plt.title("Widmo o rozdzielczości 1 Hz dla częstotliwości 50–280 Hz")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Normalizacja amplitudy
X_fft_cut = X_fft_cut / np.expand_dims(X_fft_cut.max(axis=1), axis=-1)

# Wizualizacja dla pierwszego pliku
freqs = np.arange(low_cut, high_cut)  # Oś częstotliwości
amplitudes = X_fft_cut[0, :]  # Normalizowane amplitudy pierwszego pliku

plt.figure(figsize=(8, 5))
plt.plot(freqs, amplitudes, linewidth=1.5, color='blue')  # Bez skalowania amplitudy
plt.title("Widmo o rozdzielczości 1 Hz dla częstotliwości 50–280 Hz")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda (znormalizowana)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(0, 1)  # Ograniczenie osi Y od 0.0 do 1.0
plt.show()

# --- PCA - przykład analizy głównych składowych ---
example = np.random.randn(500, 2)
example[:, 1] *= 0.4
rot_matrix = np.array([[1 / 2 ** 0.5, 1 / 2 ** 0.5], [1 / 2 ** 0.5, -1 / 2 ** 0.5]])
example = np.dot(example, rot_matrix)

example_PCAed = PCA(2).fit_transform(example)


# Wizualizacja wyników PCA
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].scatter(example[:, 0], example[:, 1])
ax[1].scatter(example_PCAed[:, 0], example_PCAed[:, 1])

ax[0].set_xlim([-3, 3])
ax[0].set_ylim([-3, 3])
ax[1].set_xlim([-3, 3])
ax[1].set_ylim([-3, 3])

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('PC 1')
ax[1].set_ylabel('PC 2')

ax[0].set_title('Dane pierwotne')
ax[1].set_title('Dane po PCA')

fig.tight_layout()
plt.show()

# --- PCA dla X_raw (dane wejściowe do klasyfikatora) ---
# Przygotowanie X_train jako zbioru danych po FFT
X_train = X_fft_cut  # Możesz użyć danych FFT do PCA

# Dopasowanie PCA i obliczenie wariancji
pca_transform = PCA()
pca_transform.fit(X_train)

# Wariancje wyjaśnione przez poszczególne komponenty
variances = pca_transform.explained_variance_ratio_

# Skumulowane wariancje
cumulated_variances = variances.cumsum()

# Wizualizacja skumulowanych wariancji
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("Indeks komponentu PCA")
plt.ylabel("Skumulowana wariancja")
plt.title("Skumulowana wariancja wyjaśniona przez kolejne komponenty PCA")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Liczba komponentów, które wyjaśniają 95% wariancji
PC_num = (cumulated_variances < 0.95).sum()
print(f'Liczba komponentów, które wyjaśniają 95% wariancji: {PC_num}')
