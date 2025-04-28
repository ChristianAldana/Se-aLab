import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
output_folder = 'utils/data_dinamica'
os.makedirs(output_folder, exist_ok=True)

secuencia_frames = 30
etiqueta = input("¿Qué seña dinámica quieres capturar?: ")

capturando = False
frames_guardados = []
contador_frames = 0

cap = cv2.VideoCapture(0)

print("Presiona 'c' para comenzar la captura. 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        data = []
        for lm in hand_landmarks.landmark:
            data += [lm.x, lm.y, lm.z]

        if capturando:
            frames_guardados.append(data)
            contador_frames += 1

            if contador_frames == secuencia_frames:
                archivo_salida = os.path.join(output_folder, f"{etiqueta}_{len(os.listdir(output_folder))}.pkl")
                with open(archivo_salida, 'wb') as f:
                    pickle.dump(frames_guardados, f)
                print(f"Se guardó una secuencia de '{etiqueta}' ✅")
                frames_guardados = []
                contador_frames = 0
                capturando = False

    if capturando:
        cv2.putText(frame, f'Capturando {etiqueta}: {contador_frames}/{secuencia_frames}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'Presiona "c" para capturar {etiqueta}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Captura de Secuencias Dinámicas', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        capturando = True
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
