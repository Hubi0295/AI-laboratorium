import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Wczytanie zbioru danych
df = pd.read_csv("StudentPerformanceFactors.csv",
                 sep = ",");

# Wyświetlenie pierwszych pięciu wierszy ramki danych
df.head()

# Zamiana wartości kategorycznych na wartości liczbowe
# (dla pojedyńczej kolumny)
df["Parental_Involvement"] = df["Parental_Involvement"].replace(
    {"Low":1, "Medium":2, "High": 3})

df["Internet_Access"] = df["Internet_Access"].replace(
    {"Yes":1, "No":0})

# Kodowanie wartości kategorycznych na liczby całkowite
# dla wszystkich kolumn
for col in df.select_dtypes(include=['object']).columns:
    df[col], _ = pd.factorize(df[col])

#---------------------------------------------------
# Wyjaśnienie działania funkcji pd.factorize
# Przykładowa kolumna z wartościami kategorycznymi
data = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])
# Użycie pd.factorize do kodowania
codes, uniques = pd.factorize(data)

print("Zakodowane wartości:", codes)
print("Unikalne wartości:", uniques)
#---------------------------------------------------

# Obliczenie korelacji
correlation_matrix = df.corr()
# Wybór tylko kolumn zawierających wartości liczbowe
#correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Ustawienie rozmiaru wykresu
plt.figure(figsize=(16, 16))

# Wizualizacji macierzy w formie graficznej 
sns.heatmap(correlation_matrix, 
            annot=True, cmap='coolwarm', square=True)

# Wyświetlenie wykresu
plt.show()

# Kodowane One-Hot Encoding ( 1 z n)
# Kodowanie one-hot to technika używana do reprezentacji danych
# kategorycznych w formie numerycznej. Dla każdej unikalnej
# wartości kategorii tworzona jest nowa kolumna (tzw. cecha binarna),
# w której wartość 1 oznacza wystąpienie danej kategorii, a 0 oznacza
# jej brak. Technika ta jest bardzo przydatna w przygotowywaniu danych
# do modeli uczenia maszynowego, które wymagają danych
# numerycznycznych

# Jak działa one-hot encoding?
# Dla każdej unikalnej wartości kategorii w danej kolumnie powstaje
# osobna kolumna. Jeśli w oryginalnej kolumnie znajduje się 5
# różnych wartości (np. 'czerwony', 'zielony', 'niebieski'),
# w wyniku kodowania one-hot powstają 3 nowe kolumny, np.:
    #'kolor_czerwony'
    #'kolor_zielony'
    #'kolor_niebieski'
# Każda z tych kolumn zawiera wartość 1, jeśli dana obserwacja
# należy do tej kategorii, i 0 w przeciwnym wypadku.

# Przykładowa ramka danych z kolumną kategoryczną
df2 = pd.DataFrame({
    'kolor': ['czerwony', 'zielony', 'niebieski', 'czerwony']
})
print(df2)

# Zastosowanie kodowania one-hot
df2_encoded = pd.get_dummies(df2, columns=['kolor'])
print(df2_encoded)