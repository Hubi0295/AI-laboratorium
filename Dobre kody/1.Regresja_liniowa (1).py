import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# z pakietu scikit-learn
from sklearn.linear_model import LinearRegression

# Ustawienie ziarna dla generatora liczb losowych w celu
# powtarzalności danych
np.random.seed(2001)

# Wygenerowanie przykładowych danych
x = np.arange(-3,3,0.1)
# Wartości funkcji tanh(x) są lekko zakłócane/zaburzane
y = np.tanh(x) + np.random.randn(x.shape[0])*0.2
# Funkcja np.random.randn z pakietu NumPy generuje losowe liczby z 
# rozkładu normalnego (o średniej 0 i odchyleniu standardowym 1)

# Generowanie 5 liczb z rozkładu normalnego o średniej 10 
# i odchyleniu standardowym 2
# losowe_liczby = np.random.normal(loc=10, scale=2, size=5)

# Wykres funkcji tanh(x) oraz wygenerowanych danych
plt.plot(x, np.tanh(x), color = "green")
plt.scatter(x,y, marker="s")
plt.legend(['tanh(x)', 'wygenerowane dane'])
plt.show()

# Model wymaga, aby dane były w postaci wektora kolumnowego,
# domyślnie x i y to wektory wierszowe.

# (-1) - automatycznie dostosowując wymiar tablicy 
# na podstawie pozostałych podanych wymiarów. 
# Oznacza to, że NumPy sam obliczy ten wymiar tak, 
# aby liczba elementów pozostała niezmieniona.

x = x.reshape(-1,1) 
y = y.reshape(-1,1)
# kształt musi być taki sam
x.shape
y.shape 
# LinearRegression to nazwa klasy
# w tym miejscu jest tworzony obiekt
model = LinearRegression()
# Dopasowanie/uczenie modelu
model.fit(x,y)
# Predykcja wartości
ypred = model.predict(x)

# Powyższy przypadek dotyczył tylko jednej zmiennej niezależnej,
# czyli model regresji liniowej ma postać y=ax+b
# (równanie prostej)

# coef_ zawiera informacje o przyjętych współczynnikach/wagach
# przez model, ale bez wyrazu wolnego 
model.coef_ # (wartość a)
# wyraz wolny
model.intercept_ # (wartość b)
# Aby pobrać samą liczbę z model.coef_ należy napisać
# model.coef_[0][0]

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, x*model.coef_[0][0] + model.intercept_[0], color = "green")
plt.legend(['Dane', 
            'Model'])
plt.show()

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, ypred)
plt.legend([ 'F(x) - aproksymująca', 
            'f(x) - aproksymowana zaszumiona'])
plt.show()






from sklearn.model_selection import train_test_split
# potrzeby jest moduł openpyxl
bh_data = pd.read_excel('housing.xlsx')
bh_data
# bh_data.values zwraca wartości przechowywane przez ramkę danych
# jako tablicę dwu-wymiarową z pakietu numpy
bh_arr = bh_data.values
# X - wszystkie kolumny oprócz ostatniej
# y - cała ostatnia kolumna
X, y = bh_arr[:,:-1], bh_arr[:,-1]
# bh_arr[:,:-1] - wszystkie wiersze, kolumny od pierwszej do przedostatniej
# bh_arr[:,-1] - wszystkie wiersze, ostatnia kolumna
# znak - w kontekście indeksu oznacza numerację od końca 
# Podział zbioru danych na część treningową oraz część testową
# Znak \ pozwala umieścić instrukcję w więcej niż jednej linijce
X_train, X_test, y_train, y_test = \
train_test_split(X,y, 
                 test_size=0.2, random_state= 221, shuffle=False)


linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
minval = min(y_test.min(), y_pred.min())
maxval = max(y_test.max(), y_pred.max())
plt.scatter(y_test, y_pred)
# [minval, maxval] = zbiór x-ów, [minval, maxval] = zbiór y-ów
plt.plot([minval, maxval], [minval, maxval], color="black")
plt.xlabel('Cena medianowa - rzeczywista (y_test)')
plt.ylabel('Cena medianowa - przewidziana przez model (y_pred)')
plt.show()
# Punkty, znajdujące się na prostej oznaczają poprawną predykcję


from sklearn.metrics import  mean_absolute_error, \
    mean_squared_error, mean_absolute_percentage_error
# MSE (Mean Squared Error) to średnia kwadratów błędów, czyli różnic między 
# wartościami rzeczywistymi ytest​ a przewidywaniami ypred​. 
# MSE mocniej karze duże błędy, ponieważ są one podnoszone do kwadratu
# MSE = 1/n*(sum_{i}^{n} (ypred_i-ytest_i)^2)
# gdzie n - liczba obserwacji
MSE = mean_squared_error(y_test, y_pred, squared=True)

# MAE (Mean Absolute Error) to średnia wartość bezwzględna różnic
# między wartościami rzeczywistymi a przewidywanymi.
# W przeciwieństwie do MSE, MAE nie podnosi błędów do kwadratu,
# dzięki czemu każdy błąd wpływa na wynik w sposób liniowy.
# MAE = 1/n*(sum_{i}^{n} |ypred_i-ytest_i|
# gdzie n - liczba obserwacji, | | - wartość bezwzględna
MAE = mean_absolute_error(y_test, y_pred)

# MAPE (Mean Absolute Percentage Error) jest miarą względną,
# która wyraża średni błąd bezwzględny jako procent wartości
# rzeczywistej. Jest szczególnie przydatna, gdy interesuje
# nas wielkość błędu w stosunku do rzeczywistej wartości
# (np. Gdy chcemy kupić 300g sera w sklepie, a otrzymamy
# 310g błąd jest względnie mały. Natomiast jeżeli pewna osoba przyjmuje
# codziennie 10mg leku i omyłkowo przyjmie podwójną dawkę,
# czyli 20mg, błąd względem referencyjnej wartości jest olbrzymi).
# Wadą MAPE jest to, że nie działa dobrze, gdy wartości rzeczywiste
# są bliskie zera, ponieważ wtedy miara może być niestabilna.
# MAPE = 100%/n*(sum_{i}^{n} |(ypred_i-ytest_i)/ytest_i|
# gdzie n - liczba obserwacji, | | - wartość bezwzględna
MAPE = mean_absolute_percentage_error (y_test, y_pred)

# Mediana (Q2): 
# Linia wewnątrz pudełka, która dzieli dane 
# na pół (50% danych jest mniejszych).

# Interkwartylowy rozstęp (IQR): 
# Różnica między Q3 a Q1 (IQR=Q3−Q1).IQR pokazuje zakres
# wartości, w którym znajduje się środkowe 50% danych.

# Wartości odstające (outliers):
# To punkty danych, które znajdują się powyżej górnego wąsa 
# lub poniżej dolnego wąsa.

# Wysokość pudełka wskazuje na rozrzut danych.

plt.boxplot(y_train)
plt.title("Medianowa wartosc mieszkania")
plt.show()

#   y_train - y_train.mean(): 
# Oblicza różnicę pomiędzy każdą wartością w y_train 
# a średnią (mean) tych wartości. To daje nam odchylenie
# każdej wartości od średniej.
#   / y_train.std(): 
# Dzieli odchylenia przez odchylenie standardowe (std) y_train,
# co normalizuje różnice, tworząc tzw. z-score dla każdej wartości.
# Z-score wskazuje, o ile odchyleń standardowych dana wartość
# znajduje się od średniej.
#
# Z - score:
# z =(x-m)/(std)
# gdzie:
#    z to z-score,
#    x to wartość, dla której obliczamy z-score,
#    m to średnia (mean) wszystkich wartości w zbiorze danych,
#    std to odchylenie standardowe (standard deviation) tych wartości.
#
# Funkcja np.abs oblicza wartość bezwzględną z z-score.
# To sprawia, że niezależnie od tego, czy wartość jest powyżej,
# czy poniżej średniej, skupiamy się na jej odległości od niej.
# W praktyce, wartości odstające to te, które znajdują się
# więcej niż 3 odchylenia standardowe od średniej.
outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
# Znak ~ to operator bitowej negacji w Pythonie. 
# Dla macierzy wartości logicznych zamienia on 
# True na False i False na True. 
# Oznacza to, że odwraca wszystkie wartości logiczne.
X_train_no_outliers = X_train[~outliers,:]
y_train_no_outliers = y_train[~outliers]
# Kopiowanie danych
y_train_mean = y_train.copy()
# Zastąpienie wartości odstających średnią 
y_train_mean[outliers] = y_train.mean()


linReg = LinearRegression()
linReg.fit(X_train, y_train_mean)

# Czy predykcja się poprawiła?
ypred = linReg.predict(X_test)
MSE2 = mean_squared_error(y_test, y_pred, squared=True)
MAE2 = mean_absolute_error(y_test, y_pred)
MAPE2 = mean_absolute_percentage_error (y_test, y_pred)

wagi = linReg.coef_
# bh_data.columns - zwraca nazwy kolumny
niezleżne_cechy = bh_data.columns[:-1]
# funkcja len zwraca długość listy
# wygenerowanie liczb 0,1,2,3... aż do len(niezleżne_cechy)
x = np.arange(len(niezleżne_cechy))
fig, ax = plt.subplots(1,1)
ax.bar(x, wagi)
ax.set_xticks(x)
ax.set_xticklabels(niezleżne_cechy, rotation = 90)
fig.show()
plt.show()


# Funkcja np.stack łączy ciąg tablic wzdłuż określonego kierunku
# axis=-1 oznacza, że nowe cechy będą dodawane jako
# dodatkowe kolumny (w ostatniej osi) w wyniku nowe_dane.
# Wynikowy obiekt będzie macierzą, w której każda kolumna
# odpowiada jednej z nowo utworzonych cech.
nowe_dane = np.stack([X[:, 4]/X[:, 7], 
                      X[:, 4]/X[:, 5], 
                      X[:, 4]*X[:, 3],
                      X[:, 4]/X[:, -1]], 
                        axis=-1)

# Funkcja np. concatenate służy do łączenia dwóch lub więcej tablic
# wzdług określonej osi
X_additional = np.concatenate([X, nowe_dane], axis=-1)



