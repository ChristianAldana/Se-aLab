import cv2
import mediapipe as mp
import csv
import os
import time

# Crear carpeta para guardar datos
if not os.path.exists('utils/data'):
    os.makedirs('utils/data')

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Archivo CSV donde se guardarán los landmarks
csv_filename = 'utils/data/landmarks.csv'

# Si el archivo no existe, crear encabezado
if not os.path.isfile(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        csv_writer.writerow(header)

# Iniciar captura de video
cap = cv2.VideoCapture(0)

# Escribe la etiqueta que quieres capturar (ej: "hola", "adios", "gracias", etc.)
etiqueta = input("¿Qué seña quieres capturar?: ")

print("Capturando automáticamente... presiona 'q' para salir.")

# Control de tiempo
ultimo_guardado = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Capturar cada 0.5 segundos automáticamente
            if time.time() - ultimo_guardado > 0.5:
                data = [etiqueta]
                for lm in hand_landmarks.landmark:
                    data += [lm.x, lm.y, lm.z]

                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(data)

                ultimo_guardado = time.time()
                print(f"Seña '{etiqueta}' guardada ✅")

    cv2.imshow('Captura Automática de Señales', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
