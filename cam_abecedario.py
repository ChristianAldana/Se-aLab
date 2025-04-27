import cv2
import numpy as np
import pickle

# Cargar el modelo entrenado
with open("model_knn_abecedario_a.pkl", "rb") as f:
    knn = pickle.load(f)

# Función para preprocesar la imagen
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))  # debe coincidir con el tamaño usado en el entrenamiento
    return img.flatten().reshape(1, -1)

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar un rectángulo donde colocar la mano
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocesar y predecir
    input_data = preprocess(roi)
    pred = knn.predict(input_data)[0]

    # Mostrar predicción en pantalla
    cv2.putText(frame, f"Letra: {pred}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento de Letra", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
