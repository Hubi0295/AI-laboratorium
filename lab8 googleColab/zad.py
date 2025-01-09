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
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)
mask = (data['Gender'].values == 'Female')
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0
one_hot = pd.get_dummies(data.Property_Area)
one_hot = one_hot.astype(int)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])
data['Married'], _ = pd.factorize(data["Married"])
data['Self_Employed'], _ = pd.factorize(data["Self_Employed"])
data['Education'] = data['Education'].replace({"Graduate":1, "Not Graduate":0})
data['Loan_Status'] = data['Loan_Status'].replace({"Y":1, "N":0})
vals = data.values.astype(np.float64)
X = vals[:, :-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 221, shuffle=False)


neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001
model = Sequential()
model.add(Dense(neuron_num, activation='relu',input_shape = (X_train.shape[1],),kernel_regularizer = l2(0.01)))
block = [
 Dense,
 LayerNormalization,
 BatchNormalization,
 Dropout,
 GaussianNoise]
args = [(neuron_num,'selu'),(),(),(do_rate,),(noise,)]
model = Sequential()
model.add(Dense(neuron_num, activation='relu',input_shape = (X.shape[1],)))
repeat_num = 2
for i in range(repeat_num):
 for layer,arg in zip(block, args):
  model.add(layer(*arg))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= Adam(learning_rate),loss='binary_crossentropy',metrics=['accuracy', 'Recall', 'Precision'])

epochs = 100
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=32
)

def plot_model_metrics(model, epochs=100):
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
plot_model_metrics(model)