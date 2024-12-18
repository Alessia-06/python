import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generazione di cerchi non concentrici
x_circles, y_circles = make_circles(n_samples=1000, noise=0.05)

# Visualizzazione
plt.scatter(x_circles[np.where(y_circles == 0)[0], 0], x_circles[np.where(y_circles == 0)[0], 1],
            color='red', label='Classe 0')
plt.scatter(x_circles[np.where(y_circles == 1)[0], 0], x_circles[np.where(y_circles == 1)[0], 1],
            color='blue', label='Classe 1')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.tight_layout()
plt.show()

# Dividere i dati in train e test
x_train_circle, x_test_circle, y_train_circle, y_test_circle = train_test_split(x_circles, y_circles, test_size=0.2, shuffle=True)

# Reshape dei dati
y_train_circle = y_train_circle.reshape(-1, 1)
y_test_circle = y_test_circle.reshape(-1, 1)

# Creazione modello
circle_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),  # Strato di input e primo hidden
    tf.keras.layers.Dense(10, activation='relu'),  # Secondo hidden layer
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Strato di output
])

# Compilazione del modello
lr = 0.005
circle_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=['accuracy'])

# Addestramento del modello
n_epochs = 250
batch_size = 64
history = circle_mlp.fit(x_train_circle, y_train_circle, validation_split=0.2, epochs=n_epochs, batch_size=batch_size, verbose=1)

# Valutazione del modello
test_loss, test_accuracy = circle_mlp.evaluate(x_test_circle, y_test_circle, verbose=0)
print(f"Loss sul set di test: {test_loss}")
print(f"Accuracy sul test set: {test_accuracy}")

# Plot della perdita
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) +1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Andamento della Loss durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Predizioni sul test set
y_pred_circles_probabilities = circle_mlp.predict(x_test_circle)
y_pred_circles = (y_pred_circles_probabilities >= 0.5).astype(int)

# Matrice di confusione
conf_matrix = confusion_matrix(y_test_circle, y_pred_circles)
ConfusionMatrixDisplay(conf_matrix, display_labels=['Rosso', 'Blu']).plot(colorbar=False)
plt.title("Matrice di confusione")
plt.show()