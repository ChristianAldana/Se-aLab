import cv2
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('model_knn_abecedario_a.pkl')

# Ruta a una imagen de prueba
test_image_path = 'data/abecedario/a/1.JPG'  # Cambia esto si quieres probar otra imagen

# Leer y procesar la imagen
img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (100, 100))
img_flatten = img.flatten().reshape(1, -1)

# Predecir
prediction = model.predict(img_flatten)

print(f"Predicción: {prediction[0]}")
