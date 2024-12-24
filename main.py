import cv2
import mediapipe as mp
import math


# Инициализация объектов MediaPipe для обработки рук
def initHandelObject():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing, mp_hands


# Нахождение координат кончиков указательных пальцев
def setFingerCoord(landmarks, frame):
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
    x_center = landmarks.landmark[9].x  # Координата центра ладони по оси x

    if x_center < 0.5:
        return 'left'  # Левая рука
    else:
        return 'right'  # Правая рука


# Функция создания мяча
def createBall(frame_width, frame_height):
    ball_pos_x, ball_pos_y = 320, 5  # Начальная позиция мяча
    ball_radius = 10
    speed_x = 0  # Начальная скорость по оси X
    speed_y = 0  # Начальная скорость по оси Y
    gravity_y = 0.5  # Гравитация
    bounce_strength = 1.5  # Коэффициент отскока
    return ball_pos_x, ball_pos_y, ball_radius, speed_x, speed_y, gravity_y, bounce_strength


def checkBallLineCollision(ball_pos_x, ball_pos_y, ball_radius, normalX1, normalY1, normalX2, normalY2):
    line_slope = (normalY2 - normalY1) / (normalX2 - normalX1) if normalX2 != normalX1 else float('inf')

    if line_slope == float('inf'):
        if abs(ball_pos_x - normalX1) < ball_radius:
            return True, line_slope
    else:
        line_y = line_slope * (ball_pos_x - normalX1) + normalY1
        if abs(ball_pos_y - line_y) < ball_radius:
            return True, line_slope

    return False, 0



# Функция для расчета отскока мяча от наклонной линии
def calculateBounce(speed_y, line_slope, bounce_strength):
    if line_slope != float('inf'):
        angle_of_incidence = math.atan(line_slope)
        bounce_angle = 2 * angle_of_incidence
        bounce_speed = speed_y * bounce_strength * math.cos(bounce_angle)
        return -bounce_speed
    else:
        return -speed_y * bounce_strength


# Обработка отскока мяча от стен и верха экрана
def handleWallCollisions(ball_pos_x, ball_pos_y, ball_radius, speed_x, speed_y, frame_width, frame_height,
                         bounce_strength):
    # Отскок от верхней границы
    if ball_pos_y - ball_radius <= 0:
        speed_y = -speed_y * bounce_strength

    # Отскок от нижней границы (появление сверху)
    if ball_pos_y + ball_radius >= frame_height:
        ball_pos_y = ball_radius  # Сбрасываем мяч наверх
        speed_y = 0  # Сброс скорости

    # Отскок от левой границы
    if ball_pos_x - ball_radius <= 0:
        speed_x = -speed_x * bounce_strength

    # Отскок от правой границы
    if ball_pos_x + ball_radius >= frame_width:
        speed_x = -speed_x * bounce_strength

    return speed_x, speed_y, ball_pos_y


# Основной метод открытия и обработки обнаружения рук через камеру
def mainWhileCam(hands, mp_drawing, mp_hands):
    # Открываем камеру
    cam = cv2.VideoCapture(0)

    # Получение ширины и высоты окна камеры
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Width: {frame_width}, Height: {frame_height}")

    left_hand_coords = None
    right_hand_coords = None

    # Начальные параметры мяча
    ball_pos_x, ball_pos_y, ball_radius, speed_x, speed_y, gravity_y, bounce_strength = createBall(frame_width,
                                                                                                   frame_height)

    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    cv2.resizeWindow('Hand Detection', 1280, 720)

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

                if hand_type == 'left':
                    left_hand_coords = (normalX1, normalY1, normalX2, normalY2)
                elif hand_type == 'right':
                    right_hand_coords = (normalX1, normalY1, normalX2, normalY2)

        # Рисуем линию между кончиками указательных пальцев правой и левой руки
        if left_hand_coords and right_hand_coords:
            left_x1, left_y1, left_x2, left_y2 = left_hand_coords
            right_x1, right_y1, right_x2, right_y2 = right_hand_coords
            frame = drawLine(frame, left_x1, left_y1, right_x1, right_y1)

            # Проверка столкновения мяча с линией
            collision, line_slope = checkBallLineCollision(ball_pos_x, ball_pos_y, ball_radius, left_x1, left_y1,
                                                           right_x1, right_y1)
            if collision:
                speed_y = calculateBounce(speed_y, line_slope, bounce_strength)

        # Обновляем позицию мяча
        speed_y += gravity_y  # Применяем гравитацию
        ball_pos_y += speed_y  # Обновляем вертикальную позицию мяча

        # Обработка отскока от стен и верха экрана
        speed_x, speed_y, ball_pos_y = handleWallCollisions(ball_pos_x, ball_pos_y, ball_radius, speed_x, speed_y,
                                                            frame_width, frame_height, bounce_strength)

        # Рисуем мяч на экране
        cv2.circle(frame, (int(ball_pos_x), int(ball_pos_y)), ball_radius, (0, 255, 0), -1)

        # Отображаем результат
        cv2.imshow('Hand Detection', frame)

        # Закрыть окно при нажатии на клавишу 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Инициализация объектов MediaPipe и запуск камеры
hands, mp_drawing, mp_hands = initHandelObject()
mainWhileCam(hands, mp_drawing, mp_hands)

if __name__ == "__main__":
    # Инициализируем объекты для обработки рук
    hands, mp_drawing, mp_hands = initHandelObject()

    # Открываем камеру и начинаем обработку
    mainWhileCam(hands, mp_drawing, mp_hands)
