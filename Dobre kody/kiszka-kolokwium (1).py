from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data.csv')
#ETAP 1
print('------------------------------ ETAP 1 -------------------------\n')
print(df.head())
print(f"Liczba wierszy: {df.shape[0]}")
print("Nazwy kolumn:")
print(df.columns)
print("Typy wartości w kolumnach:")
print(df.dtypes)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='alcohol', bins=20, edgecolor='black', color='skyblue')
plt.title('Rozklad cechy alcohol')
plt.xlabel('alcohol')
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='color_intensity', bins=20, edgecolor='black', color='skyblue')
plt.title('Rozkład cechy color intensity')
plt.xlabel('Color Intensity')
plt.figure(figsize=(8,6))
sns.boxplot(data=df, y='magnesium')
plt.title('Wykres boxplot dla magnesium')
plt.show()

outliers = (np.abs((df-df.mean())/df.std() > 3)).sum() #wzor na policzenie wartosci odstajacych
print(f"Wystepuja 4  wartosci odstajace")
print(f"Mediana cechy magnesium wynosi: {df['magnesium'].median()}")
print(f"Srednia wartosc cechy magnesium wynosi: {df['magnesium'].mean()}")

#ETAP 2
print('------------------------------ ETAP 2 -------------------------\n')
corr_matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Macierz korelacji cech chemicznych")
plt.show()
print("Cecha najmocniej skorelowana dodatnio z\n cechą zależną: Alcanity of Ash")
print("Cecha najmocniej skorelowana ujemnie z\n cechą zależną: flavanoids")
print("Najmocniej skorelowane dodatnio cechy: od280/od315_of_diluted_wines z flavanoids")

#ETAP 3
print('------------------------------ ETAP 3 -------------------------\n')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=29, shuffle=True)

#ETAP 4
print('------------------------------ ETAP 4 -------------------------\n')
models = [
        ('kns',KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', random_state=29)),
        ('dtree1', DecisionTreeClassifier(random_state=29)),
        ('dtree2', DecisionTreeClassifier(random_state=29, criterion='entropy')),
        ('rfc', RandomForestClassifier(random_state=29))
]
best_model = None
best_accr = -1
best_f1 = -1
worst_model = None
worst_accr = 1
worst_f1 = 1
for model_name, model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accr = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix,annot=True, cmap='Blues', fmt='d')
    plt.xlabel("predykcje")
    plt.ylabel("Wartosci rzeczywiste")
    plt.title(f'Macierz pomyłek {model_name}')
    plt.show()
    print(f"Model: {model_name}")
    print(f"Accuracy: {accr:.2f}")
    print(f"F1-score: {f1:.2f}")
    if accr > best_accr:
        best_accr = accr
        best_f1 = f1
        best_model = model_name
 
    if accr < worst_accr:
        worst_accr = accr
        worst_f1 = f1
        worst_model = model_name

print(f"Najlepszy model to {best_model} z dokladnoscia {best_accr:.2f} z f1-score {best_f1:.2f}")
print(f"Najgorszy model to {worst_model} z dokladnoscia {worst_accr:.2f} zf1-score {worst_f1:.2f}")
