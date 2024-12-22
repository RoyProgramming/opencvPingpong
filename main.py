import cv2
import mediapipe as mp

# Инициализация объектов MediaPipe для обработки рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Открываем камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем изображение в формат RGB, так как MediaPipe работает с этим форматом
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обрабатываем кадр для детекции рук
    results = hands.process(rgb_frame)

    # Если в кадре есть руки
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Отображаем ключевые точки рук на изображении
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Отображаем результат
    cv2.imshow('Hand Detection', frame)

    # Закрыть окно при нажатии на клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем камеру и окна
cap.release()
cv2.destroyAllWindows()
