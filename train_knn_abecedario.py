import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Ruta de la carpeta con las imágenes de la letra 'a'
folder_path = 'data/abecedario/a'

# Inicializamos datos
X = []
y = []

# Leemos cada imagen en la carpeta
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(folder_path, filename)
        print(f"Intentando abrir: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))  # Ajustar tamaño si es necesario
            X.append(img.flatten())
            y.append('a')  # Todas son letra 'a'
        else:
            print(f"No se pudo abrir la imagen: {file_path}")

# Convertimos a arrays
X = np.array(X)
y = np.array(y)

# Entrenamos el modelo
if len(X) > 0:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Guardamos el modelo
    joblib.dump(knn, 'model_knn_abecedario_a.pkl')
    print("Modelo entrenado y guardado con éxito.")
else:
    print("No se cargaron imágenes correctamente.")
