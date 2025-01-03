import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.copy_on_write = False
# Uwaga
# Aż do wersji pandas 3.0, odwoływanie się do poszczególnych
# elementów ramki danych poprzez nawiasy [][] zwracało
# referencję do oryginalnych wartości 
df = pd.DataFrame({"foo": [10, 20, 30], "bar": [4, 5, 6]})
print(df)
df['foo'][2] = 3 # oryginalny obiekt df jest modyfikowany
print(df)
df['foo'] = [-8,-8,-3] # oryginalny obiekt df jest modyfikowany
print(df)
# Od wersji 3.0 notacja wykorzystująca [][] zwraca kopię danych
# Wprowadzona zmiana nazywa się ,,Copy on write'' (CoW) i
# w starszych wersjach pandas można ją włączyć za pomocą 
# instrukcji:
pd.options.mode.copy_on_write = True
df = pd.DataFrame({"foo": [10, 20, 30], "bar": [4, 5, 6]})
print(df)
df['foo'][2] = 3 # oryginalny obiekt df NIE jest modyfikowany
print(df)
df['foo'] = [-8,-8,-3] # oryginalny obiekt df jest modyfikowany
print(df)
# Od wersji pandas 3.0 modyfikacja konkretnego elementu może 
# odbyć się przy pomocy funkcji df.iloc
df.iloc[0,0] = 3
print(df)


# Wczytanie danych
# potrzeby jest moduł openpyxl
data_const = pd.read_excel('loan_data.xlsx')
data = pd.read_excel('loan_data.xlsx')
# Wyświetlenie 5 początkowych wierszy z ramki danych
data.head()
# Wyświetlenie podstawowych informacji o ramce danych
data.info()

# Uwaga 
# Loan_Status to zmienna zależna

# Pobranie nazw kolumn
columns = list(data.columns)
# Mask to wektor zawierający True/False w zależności czy
# obserwacja jest kobietą, czy mężczyzną.
mask = (data['Gender'].values == 'Female')
# data['Gender'][mask] zwraca wiersze (referencje, od pandas 3.0
# zwraca kopie) kolumny
# Gender, które mają wartość Female
#data['Gender'][mask] = 1
# zamiast ~ można użyć funkcji np.logical_not(mask)
#data['Gender'][~mask] = 0

# Aby zmodyfikować dane bezpośrednio w oryginalnej ramce, 
# należy użyć metody .loc[], która zapewnia bezpośredni
# dostęp do komórek, unikając problemu kopiowania danych.
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0

# Kodowane One-Hot Encoding ( 1 z n )
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

# Wyświetlenie występujących wartości zmiennej kategorycznej
data['Property_Area'].unique()
# Zliczenie wystąpień unikatowych wartości
data['Property_Area'].value_counts()

# Do kolumn można odwoływać się stosując notację
# data.Nazwa_Kolumny
# cat_feature = pd.Categorical(data.Property_Area)
# one_hot = pd.get_dummies(cat_feature)
one_hot = pd.get_dummies(data.Property_Area)
# Jak to wygląda?
print(one_hot)
# Wartości są True/False, a chcemy aby były 1/0
one_hot = one_hot.astype(int)
print(one_hot)
# Dołączenie nowych kolumn do oryginalnej ramki danych
data = pd.concat([data, one_hot], axis = 1)
# Usunięcie oryginalnej kolumny Property_Area
data = data.drop(columns = ['Property_Area'])
# Czy wszystko jest ok?
data.head()

# Przekształcenie pozostałych kolumn o typie kategorycznym
# factorize zwraca dwie wartości dlatego jest , _
data['Married'], _ = pd.factorize(data["Married"])
data['Self_Employed'], _ = pd.factorize(data["Self_Employed"])

data['Education'].unique()
# Education zawiera tylko dwie wartości
data['Education'] = data['Education'].replace(
    {"Graduate":1, "Not Graduate":0})

# Czy wszystko jest ok?
data.head()
data_const.head()

#---------------------------------------------------
# Wyjaśnienie działania funkcji pd.factorize
# Przykładowa kolumna z wartościami kategorycznymi
data2 = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])
# Użycie pd.factorize do kodowania
codes, uniques = pd.factorize(data2)

print("Zakodowane wartości:", codes)
print("Unikalne wartości:", uniques)
#---------------------------------------------------

# Obecnie data zawiera dodatkowe kolumny, więc należy ponownie
# pobrać informacje o nazwach cech
features = data.columns

# Uwaga
# Loan_Status to zmienna zależna
# Należy przenieść kolumnę Loan_Status na koniec tabeli!
ko = data['Loan_Status']
data = data.drop(columns=['Loan_Status'])
# Dodanie kolumny na końcu
data['Loan_Status'] = ko

# Czy wszystko jest ok?
data.head()

data['Loan_Status'].unique()
# factorize działa poprawnie dla Yes/No
data['Loan_Status'] = data['Loan_Status'].replace(
    {"Y":1, "N":0})

# Czy wszystko jest ok?
data.head()

# vals = data.values.astype(np.float)
# Konwertowanie danych na typ float64
vals = data.values.astype(np.float64)
X = vals[:, :-1]
y = vals[:,-1]

# Podział danych na zbiór treningowy/uczący oraz zbiór testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X,y, 
                 test_size=0.2, random_state= 221, shuffle=False)

# Działanie train_test_split:
# Funkcja dzieli dane wejściowe (X i y) na cztery części:
#    X_train: dane wejściowe do uczenia modelu. 
#    X_test: dane wejściowe do testowania modelu.
#    y_train: ewyniki dla zbioru treningowego.
#    y_test: wyniki dla zbioru testowego.
#
# Parametry funkcji train_test_split:
#    X, y: Dane wejściowe i odpowiadające im etykiety.
#    test_size=0.2: Ustalony rozmiar zbioru testowego. 
#       W tym przypadku 20% danych zostanie przeznaczone
#       na testy, a pozostałe 80% na trenowanie.
#    random_state=221: Ustawienie tego parametru pozwala 
#       na uzyskanie powtarzalnych wyników — podział danych
#       na zbiory treningowy i testowy będzie taki sam
#       za każdym uruchomieniem kodu.
#    shuffle=False: Bez przetasowania danych przed
#        podziałem, co oznacza, że dane są dzielone
#        w kolejności, w jakiej występują w zestawie.
#        Może to być istotne, gdy dane są uporządkowane
#        w sposób, który chcemy zachować.

# z pakietu scikit-learn
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

models = [kNN(), SVM()]
model_names = ["k-Nearest Neighbors", "Support Vector Machine"]
# Ustalanie etykiet klas
class_labels = ['Loan_Status: No', 'Loan_Status: Yes']

# Metryki
accuracies = []
precisions = []
recalls = []
f1s = []
roc_aucs = []
# funkcja zip łączy elementy list w pary (x,y) 
# (X-element z listy X)
# (Y-element z listy Y)
for model, name in zip(models, model_names):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predykcje")
    plt.ylabel("Wartości rzeczywiste")
    plt.title(f"Macierz pomyłek dla {name}")
    plt.show()

    # append dodaje element do końca listy
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    roc_aucs.append(roc_auc_score(y_test, y_pred))

# Przedstawienie graficzne metryk
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
wartosci = np.array([accuracies, precisions, recalls, f1s, roc_aucs])
fig, ax = plt.subplots(figsize=(10, 6))
# Ustawienia słupków
width = 0.35  # Szerokość słupków
x = np.arange(len(metrics))  # Pozycje na osi x

# accuracies, precisions .... każda z tych list zawiera dwie wartości
# zatem tablica ,,wartosci'' jest tablicą dwu-wymiarową
# Dodawanie słupków dla obu modeli
ax.bar(x - width/2, wartosci[:, 0], width, label='kNN', color='lightblue')
ax.bar(x + width/2, wartosci[:, 1], width, label='SVM', color='salmon')

ax.set_title('Porównanie metryk dla kNN i SVM')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Wyświetlenie wykresu
plt.tight_layout()
plt.show()



# Tworzenie wykresów pudełkowych dla każdej kolumny
# Rozmiar wykresu
plt.figure(figsize=(16, 12))  
# Funkcja enumerate() generuje automatycznie indeksy. Wynikiem funkcji
# jest indeks elementu oraz wartość. Przydatne gdy
# potrzebujemy zarówno elementu, jak i jego indeksu.
for i, column in enumerate(data.columns):
    # plt.subplot(liczba_wierszy, liczba_kolumn, indeks_aktualnego_wykresu)  
    plt.subplot(3, 5, i + 1)  
    plt.boxplot(data[column])
    plt.title(column)
plt.show()


# Kolumny o wartościach 0 lub 1 wyłączone są ze skalowania
# (ale można też standaryzować wszystkie dane)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Metoda fit oblicza średnią (mean) i odchylenie standardowe (standard deviation)
# dla każdej cechy w zbiorze danych treningowych X_train
scaler.fit(X_train[:,[2,5,6,7,8]])
# Metoda transform przekształca dane treningowe X_train zgodnie
# z wcześniej obliczonymi średnimi i odchyleniami standardowymi.
# z = (x-mean)/(std)

X_train[:,[2,5,6,7,8]] = scaler.transform(X_train[:,[2,5,6,7,8]])
X_test[:,[2,5,6,7,8]] = scaler.transform(X_test[:,[2,5,6,7,8]])

accuracies_2 = []
precisions_2 = []
recalls_2 = []
f1s_2 = []
roc_aucs_2 = []
for model, name in zip(models, model_names):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predykcje")
    plt.ylabel("Wartości rzeczywiste")
    plt.title(f"Macierz pomyłek dla {name}")
    plt.show()

    # append dodaje element do końca listy
    accuracies_2.append(accuracy_score(y_test, y_pred))
    precisions_2.append(precision_score(y_test, y_pred))
    recalls_2.append(recall_score(y_test, y_pred))
    f1s_2.append(f1_score(y_test, y_pred))
    roc_aucs_2.append(roc_auc_score(y_test, y_pred))


# Porównanie metryk

# Szerokość słupków
width = 0.9

# Ustalamy pozycje słupków na osi X
x = np.arange(2)

metryki = np.array([(accuracies, accuracies_2),
                     (precisions, precisions_2),
                     (recalls, recalls_2),
                     (f1s, f1s_2),
                     (roc_aucs, roc_aucs_2)])

# Tworzenie wykresów
fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(16, 9))
# Spłaszczanie tablicy do 1D, aby można było liniowa iterować
axs = axs.flatten()
# Ukrycie ostatniego nieużywanego wykresu
axs[5].axis('off')
# Funkcja enumerate() generuje automatycznie indeksy. Wynikiem funkcji
# jest indeks elementu oraz wartość. Przydatne gdy
# potrzebujemy zarówno elementu, jak i jego indeksu.
# start - parametr określający początkową wartość indeksu (domyślnie 0).
for i, m in enumerate(metryki):
    # Dodawanie słupków dla obu modeli
    # [m[0][1], m[1][1]] - dla SVM
    axs[i].bar(x - width/2, [m[0][0], m[1][0]], width, color=['#4CAF50', '#FF9800'])

    # Etykietowanie
    axs[i].set_ylabel('Dokładność')
    axs[i].set_title(f'Porównanie {["Dokładności", "Precyzji", "Czułości", "F1 Score", "ROC AUC"][i]}')
    
    # Ustawianie oznaczeń na osi X
    axs[i].set_xticks(x - width/2)
    axs[i].set_xticklabels(['Model podstawowy', 'Model zmodyfikowany'])

# Dodanie tytułu całego obrazka
fig.suptitle('Porównanie metryk kNN', fontsize=16)

# Wyświetlenie wykresu
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Dostosowanie, aby tytuł się nie nakładał
plt.show()


# Dotychczasowe dane X_train, X_test zostały przeskalowane

X_train, X_test, y_train, y_test = \
train_test_split(X,y, 
                 test_size=0.2, random_state= 221, shuffle=False)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
model = DT(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predykcje")
plt.ylabel("Wartości rzeczywiste")
plt.title(f"Macierz pomyłek dla Decision Tree Classifier")
plt.show()

plt.figure(figsize=(25,15))
tree_vis = plot_tree(model,feature_names= data.columns[:-1],
                     class_names=['N', 'Y'], fontsize = 20)