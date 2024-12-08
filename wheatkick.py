import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math

# 初始化 mediapipe 的模組
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化 pynput 的鍵盤控制器
keyboard = Controller()

last_action_time = time.time()  # 最後一次動作的時間
current_action = None           # 用於追蹤當前的動作狀態

# 輕踢和重踢的高度閾值
light_kick_threshold = 0.05  # 左腳輕踢的高度閾值
heavy_kick_threshold = 0.05  # 右腳重踢的高度閾值
speed_threshold = 0.02      # 抬高速度閾值

# 回旋的設定
rotation_threshold = 15  # 旋轉角度閾值（度）
rotation_cooldown = 1.0  # 旋轉動作之間的最小間隔時間（秒）
previous_orientation = None  # 前一幀的身體朝向角度

# 用於追蹤膝蓋位置
previous_left_knee_y = None
previous_right_knee_y = None

def calculate_orientation(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    delta_x = right_shoulder.x - left_shoulder.x
    delta_y = right_shoulder.y - left_shoulder.y
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("無法接收影像幀")
            break
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

            # 計算膝蓋的相對高度
            left_knee_raise = left_hip.y - left_knee.y
            right_knee_raise = right_hip.y - right_knee.y

            # 計算膝蓋的速度
            if previous_left_knee_y is not None:
                left_knee_speed = previous_left_knee_y - left_knee.y
            else:
                left_knee_speed = 0

            if previous_right_knee_y is not None:
                right_knee_speed = previous_right_knee_y - right_knee.y
            else:
                right_knee_speed = 0

            previous_left_knee_y = left_knee.y
            previous_right_knee_y = right_knee.y

            # 左腳輕踢判斷
            if left_knee_raise > light_kick_threshold and left_knee_speed > speed_threshold:
                if current_action != "Left Light Kick" and (current_time - last_action_time) > 0.2:
                    keyboard.press('z')  # 左腳輕踢
                    time.sleep(0.05)
                    keyboard.release('z')
                    print("Left Light Kick")
                    current_action = "Left Light Kick"
                    last_action_time = current_time
                    cv2.putText(img, "Left Light Kick", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 右腳重踢判斷
            if right_knee_raise > heavy_kick_threshold and right_knee_speed > speed_threshold:
                if current_action != "Right Heavy Kick" and (current_time - last_action_time) > 0.2:
                    keyboard.press('c')  # 右腳重踢
                    time.sleep(0.05)
                    keyboard.release('c')
                    print("Right Heavy Kick")
                    current_action = "Right Heavy Kick"
                    last_action_time = current_time
                    cv2.putText(img, "Right Heavy Kick", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 回旋判斷
            current_orientation = calculate_orientation(landmarks)
            if previous_orientation is not None:
                angle_diff = current_orientation - previous_orientation
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360

                if abs(angle_diff) > rotation_threshold:
                    if (current_time - last_action_time) > rotation_cooldown:
                        keyboard.press('z')
                        keyboard.press('c')
                        time.sleep(0.05)
                        keyboard.release('z')
                        keyboard.release('c')
                        if angle_diff > 0:
                            print("Spin Right")
                            cv2.putText(img, "Spin Right", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            print("Spin Left")
                            cv2.putText(img, "Spin Left", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        current_action = "Spin"
                        last_action_time = current_time

            previous_orientation = current_orientation

            # 重置動作
            if current_action is not None and (current_time - last_action_time) > 0.3:
                current_action = None

            # 標記身體節點和骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 顯示影像
        cv2.imshow('Pose Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
