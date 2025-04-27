import cv2
import mediapipe as mp

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configurar reconocimiento de manos y cara
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame")
        break

    # Convertir BGR a RGB para mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar detección de manos
    hand_results = hands.process(rgb_frame)

    # Procesar detección facial
    face_results = face_detection.process(rgb_frame)

    # Dibujar resultados de manos
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Dibujar resultados de cara
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow('Detección de manos y cara', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
