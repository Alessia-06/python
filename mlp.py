import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

def gaussiana(x):
    return np.exp(-x*x)

# Generazione dei dati
x_gauss = np.linspace(-5, 5, 1000)
y_gauss = gaussiana(x_gauss)

# Divisione in test e train
x_train_gauss, x_test_gauss, y_train_gauss, y_test_gauss = train_test_split(x_gauss, y_gauss, test_size=0.2, shuffle=True)

# Preprocessing dei dati: Reshape per tensorflow
x_train_gauss = x_train_gauss.reshape(-1,1)
x_test_gauss = x_test_gauss.reshape(-1,1)
y_train_gauss = y_train_gauss.reshape(-1,1)
y_test_gauss = y_test_gauss.reshape(-1,1)

# Visualizzazione dati
plt.plot(x_gauss, y_gauss, color='gray', label='gaussiana')
plt.scatter(x_train_gauss, y_train_gauss, color='red', label='train points')
plt.scatter(x_test_gauss, y_test_gauss, color='blue', label='test points')
plt.legend()
plt.tight_layout()
plt.show()

#Creazione del modello
gauss_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Strato di input e primo hidden
    tf.keras.layers.Dense(10, activation='relu'),  # Secondo hidden layer
    tf.keras.layers.Dense(1)  # Strato di output
])

