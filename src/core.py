import cv2
import time
import pyautogui
import mediapipe as mp


# Укорачиваем длинные имена MediaPipe
BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

RunningMode = mp.tasks.vision.RunningMode

MOVE_GESTURE = "two-up"
SCROLL_GESTURE = "pinch-up"
CLICK_GESTURE = "click"

SMOOTHING = 0.05
CLICK_HOLD_DELAY = 0.15      # сколько секунд ждать перед отпусканием
SCROLL_DEAD_ZONE = 0.015     # минимальное движение для скролла
SCROLL_SPEED = 500   
SCREEN_W, SCREEN_H = pyautogui.size()
CLICK_COOLDOWN = 0.5


class GestureRecognition:
    def __init__(self, path_to_model):
        # Настройки Gesture Recognizer
        self.options = GestureRecognizerOptions(
            # Путь к gesture_recognizer.task
            base_options=BaseOptions(
                model_asset_path=str(path_to_model)
            ),

            # Нам нужен поток с камеры
            running_mode=RunningMode.LIVE_STREAM,

            num_hands=2,
            result_callback=self._handle_result
        )       
        self.prev_x = SCREEN_W / 2
        self.prev_y = SCREEN_H / 2
        self.last_click_time = 0
        self.mouse_pressed = False
        self.last_click_gesture_time = 0
        
        self.prev_scroll_x = None
        self.prev_scroll_y = None

    def _handle_result(
        self,
        result: GestureRecognizerResult,
        output_image: mp.Image,
        timestamp_ms: int
    ):
        # Если рука вообще не распознана
        if not result.gestures or not result.hand_landmarks:
            self._release_mouse_if_needed()
            self.prev_scroll_x = None
            self.prev_scroll_y = None
            return

        gesture = result.gestures[0][0]
        gesture_name = gesture.category_name
        confidence = gesture.score

        print(f"Gesture: {gesture_name} | confidence: {confidence:.2f}")

        landmarks = result.hand_landmarks[0]
        index_finger = landmarks[8]

        x = index_finger.x
        y = index_finger.y

        # =========================================================
        # ДВИЖЕНИЕ МЫШИ
        # =========================================================

        if gesture_name == MOVE_GESTURE:

            # Если до этого кнопка была зажата,
            # при смене жеста отпускаем её
            self._release_mouse_if_needed()

            target_x = (1 - x) * SCREEN_W
            target_y = y * SCREEN_H

            mouse_x = (
                self.prev_x +
                (target_x - self.prev_x) * SMOOTHING
            )

            mouse_y = (
                self.prev_y +
                (target_y - self.prev_y) * SMOOTHING
            )

            pyautogui.moveTo(
                mouse_x,
                mouse_y,
                _pause=False
            )

            self.prev_x = mouse_x
            self.prev_y = mouse_y

            self.prev_scroll_x = None
            self.prev_scroll_y = None

        # =========================================================
        # СКРОЛЛ
        # =========================================================

        elif gesture_name == SCROLL_GESTURE:

            self._release_mouse_if_needed()

            if (
                self.prev_scroll_x is not None
                and self.prev_scroll_y is not None
            ):

                dx = x - self.prev_scroll_x
                dy = y - self.prev_scroll_y

                # -----------------------------------------
                # Горизонтальный скролл
                # -----------------------------------------

                if abs(dx) > abs(dy):

                    if abs(dx) > SCROLL_DEAD_ZONE:

                        amount = int(dx * SCROLL_SPEED)

                        if amount != 0:
                            pyautogui.hscroll(
                                amount,
                                _pause=False
                            )

                # -----------------------------------------
                # Вертикальный скролл
                # -----------------------------------------

                else:

                    if abs(dy) > SCROLL_DEAD_ZONE:

                        # Камера:
                        # движение пальца вверх -> y уменьшается
                        # поэтому инвертируем dy
                        amount = int(-dy * SCROLL_SPEED)

                        if amount != 0:
                            pyautogui.scroll(
                                amount,
                                _pause=False
                            )

            self.prev_scroll_x = x
            self.prev_scroll_y = y

        # =========================================================
        # ЗАЖАТИЕ ЛЕВОЙ КНОПКИ
        # =========================================================

        elif gesture_name == CLICK_GESTURE:

            # ==========================================
            # 1. ДВИГАЕМ КУРСОР
            # ==========================================

            target_x = (1 - x) * SCREEN_W
            target_y = y * SCREEN_H

            mouse_x = (
                self.prev_x +
                (target_x - self.prev_x) * SMOOTHING
            )

            mouse_y = (
                self.prev_y +
                (target_y - self.prev_y) * SMOOTHING
            )

            pyautogui.moveTo(
                mouse_x,
                mouse_y,
                _pause=False
            )

            self.prev_x = mouse_x
            self.prev_y = mouse_y

            # ==========================================
            # 2. ЗАЖИМАЕМ ЛЕВУЮ КНОПКУ
            # ==========================================

            self.last_click_gesture_time = time.monotonic()

            if not self.mouse_pressed:

                pyautogui.mouseDown(button="left")

                self.mouse_pressed = True

                print("🖱 LEFT DOWN")

            # Сбрасываем состояние скролла
            self.prev_scroll_x = None
            self.prev_scroll_y = None

    def _release_mouse_if_needed(self):
        """
        Отпускает левую кнопку мыши только после небольшой задержки.

        Это нужно потому, что GestureRecognizer иногда на несколько
        кадров теряет жест.
        """

        if not self.mouse_pressed:
            return

        current_time = time.monotonic()

        if (
            current_time - self.last_click_gesture_time
            > CLICK_HOLD_DELAY
        ):

            pyautogui.mouseUp(button="left")

            self.mouse_pressed = False

            print("🖱 LEFT UP")

    def run(self):

        # Открываем первую камеру
        cap = cv2.VideoCapture(0)

        # Создаём Gesture Recognizer
        with GestureRecognizer.create_from_options(
            self.options
        ) as recognizer:

            while cap.isOpened():

                # Получаем очередной кадр
                success, frame = cap.read()

                if not success:
                    break

                # OpenCV использует BGR,
                # MediaPipe ожидает RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Преобразуем numpy.ndarray → mp.Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                # Timestamp должен увеличиваться
                # для каждого кадра
                timestamp_ms = int(time.time() * 1000)

                # Асинхронно отправляем кадр MediaPipe
                recognizer.recognize_async(
                    mp_image,
                    timestamp_ms
                )

                # ESC → выход
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()