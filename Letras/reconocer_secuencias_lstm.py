import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from collections import deque

# Cargar modelo y etiquetas
model = load_model('Letras/modelo_dinamico_lstm.h5')
with open('Letras/etiquetas_lstm.pkl', 'rb') as f:
    etiquetas = pickle.load(f)

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Para guardar 30 frames recientes
cola_frames = deque(maxlen=30)

# Abrir cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Extraer puntos
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        # Agregar a la cola
        cola_frames.append(data)

        # Si tenemos suficientes frames
        if len(cola_frames) == 30:
            entrada = np.array(cola_frames)
            entrada = entrada.reshape(1, 30, 63)  # (batch, time steps, features)

            # Predecir
            prediccion = model.predict(entrada)
            indice = np.argmax(prediccion)
            etiqueta_predicha = etiquetas[indice]

            # Mostrar predicción
            cv2.putText(frame, f'Seña: {etiqueta_predicha}', (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    else:
        # Si no hay mano
        cola_frames.clear()
        cv2.putText(frame, 'No detecto mano', (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Reconocimiento de Señas Dinámicas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
