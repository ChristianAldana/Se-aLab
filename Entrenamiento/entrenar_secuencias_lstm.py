import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Cargar los datos de secuencias
data_folder = 'utils/data_dinamica'
secuencias = []
etiquetas = []

for filename in os.listdir(data_folder):
    if filename.endswith('.pkl'):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, 'rb') as f:
            secuencia = pickle.load(f)
            secuencias.append(secuencia)
            # El nombre antes del guión bajo "_" es la etiqueta (ejemplo: Hola_0.pkl → Hola)
            etiqueta = filename.split('_')[0]
            etiquetas.append(etiqueta)

# Convertir a arrays de NumPy
X = np.array(secuencias)  # X tendrá forma (n_secuencias, 30, 63)
labels = sorted(list(set(etiquetas)))  # Lista de etiquetas únicas ordenadas
label_to_index = {label: idx for idx, label in enumerate(labels)}  # Mapa etiqueta → número
y = np.array([label_to_index[label] for label in etiquetas])

# Convertir etiquetas a categóricas (one-hot encoding)
y = to_categorical(y, num_classes=len(labels))

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 63)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# Guardar modelo y etiquetas
model.save('Letras/modelo_dinamico_lstm.h5')
with open('Letras/etiquetas_lstm.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("✅ Modelo LSTM entrenado y guardado correctamente")
