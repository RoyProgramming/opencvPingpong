import cv2
import mediapipe as mp

# Инициализация объектов MediaPipe для обработки рук
def initHandelObject():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing, mp_hands

# Нахождение координат кончиков указательных пальцев
def setFingerCoord(landmarks, frame):
    # Координаты кончиков указательных пальцев (для правой и левой руки)
    x1, y1 = landmarks.landmark[8].x, landmarks.landmark[8].y  # Кончик указательного пальца
    x2, y2 = landmarks.landmark[5].x, landmarks.landmark[5].y  # Основание указательного пальца

    height, width = frame.shape[0], frame.shape[1]

    normalX1, normalY1 = int(x1 * width), int(y1 * height)  # для кончика
    normalX2, normalY2 = int(x2 * width), int(y2 * height)  # для основания

    return normalX1, normalY1, normalX2, normalY2

# Рисование линии между кончиками пальцев правой и левой руки
def drawLine(frame, normalX1, normalY1, normalX2, normalY2):
    frame = cv2.line(frame, (normalX1, normalY1), (normalX2, normalY2), (0, 0, 255), 2)
    return frame

# Определение левой или правой руки
def setRightOrLeftHand(landmarks):
    # Обычно landmark[9] — это центр ладони
    x_center = landmarks.landmark[9].x  # Координата центра ладони по оси x

    if x_center < 0.5:  # Если точка центра ладони меньше 0.5, это левая рука
        return 'left'
    else:  # иначе правая
        return 'right'

# Основной метод открытия и обработки обнаружения рук через камеру
def mainWhileCam(hands, mp_drawing, mp_hands):
    # Открываем камеру
    cam = cv2.VideoCapture(0)

    # Переменные для хранения координат для правой и левой руки
    left_hand_coords = None
    right_hand_coords = None

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        # Преобразуем изображение в формат RGB, так как MediaPipe работает с этим форматом
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обрабатываем кадр для детекции рук
        results = hands.process(rgb_frame)

        # Если в кадре есть руки
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Отображаем все точки и соединения на изображении
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Определяем, левая или правая рука
                hand_type = setRightOrLeftHand(landmarks)

                # Получаем координаты кончиков указательных пальцев
                normalX1, normalY1, normalX2, normalY2 = setFingerCoord(landmarks, frame)

                # В зависимости от типа руки сохраняем координаты
                if hand_type == 'left':
                    left_hand_coords = (normalX1, normalY1, normalX2, normalY2)
                elif hand_type == 'right':
                    right_hand_coords = (normalX1, normalY1, normalX2, normalY2)

                # Рисуем линии только если обе руки найдены
                if left_hand_coords and right_hand_coords:
                    left_x1, left_y1, left_x2, left_y2 = left_hand_coords
                    right_x1, right_y1, right_x2, right_y2 = right_hand_coords
                    # Рисуем линию между кончиками указательных пальцев правой и левой руки
                    frame = drawLine(frame, left_x1, left_y1, right_x1, right_y1)

        # Отображаем результат
        cv2.imshow('Hand Detection', frame)

        # Закрыть окно при нажатии на клавишу 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем камеру и окна
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Инициализируем объекты для обработки рук
    hands, mp_drawing, mp_hands = initHandelObject()

    # Открываем камеру и начинаем обработку
    mainWhileCam(hands, mp_drawing, mp_hands)
