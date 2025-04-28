import cv2
import mediapipe as mp

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame")
        break

    # Convertir BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 👇 AQUI Detectamos "A"
            dedo_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dedo_medio = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            dedo_anular = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            dedo_meñique = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            if (dedo_indice.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                dedo_medio.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                dedo_anular.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                dedo_meñique.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                cv2.putText(frame, 'Letra A Detectada', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de Letra A', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
