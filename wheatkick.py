import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math

# 初始化 mediapipe 的模組
mp_drawing = mp.solutions.drawing_utils          # Mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # Mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # Mediapipe 姿勢偵測

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化 pynput 的鍵盤控制器
keyboard = Controller()

last_action_time = time.time()  # 最後一次動作的時間
current_action = None           # 用於追蹤當前的動作狀態

# 用於計算腳部移動速度
previous_right_ankle_y = None
previous_left_ankle_y = None
speed_threshold = 0.02  # 定義一個速度閾值，用於識別快速移動

# 用於計算身體旋轉
previous_orientation = None
rotation_threshold = 15  # 旋轉角度閾值（度）
rotation_cooldown = 1.0  # 旋轉動作之間的最小間隔時間（秒）

def calculate_orientation(landmarks):
    # 使用左肩和右肩計算身體的朝向角度
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
        img = cv2.resize(img, (640, 480))               # 調整尺寸至更高解析度，增加辨識準確性
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 將 BGR 轉換成 RGB
        results = pose.process(img_rgb)                 # 取得姿勢偵測結果

        current_time = time.time()

        if results.pose_landmarks:
            # 取得腳和肩的位置
            landmarks = results.pose_landmarks.landmark
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # 初始化先前的腳部位置
            if previous_right_ankle_y is None:
                previous_right_ankle_y = right_ankle.y
            if previous_left_ankle_y is None:
                previous_left_ankle_y = left_ankle.y

            # 計算腳部移動速度
            right_speed = previous_right_ankle_y - right_ankle.y
            left_speed = previous_left_ankle_y - left_ankle.y

            # 更新先前的位置
            previous_right_ankle_y = right_ankle.y
            previous_left_ankle_y = left_ankle.y

            # 新增的踢腿判斷
            # 輕踢：當右腳踝或腳跟高於右膝，且移動速度超過閾值
            if ((right_ankle.y < right_knee.y - 0.02) or (right_heel.y < right_knee.y - 0.02)) and right_speed > speed_threshold:
                if current_action is None and (current_time - last_action_time) > 0.3:
                    keyboard.press('z')  # 輕踢
                    time.sleep(0.05)
                    keyboard.release('z')
                    print("Light Kick")
                    current_action = "Light Kick"
                    last_action_time = current_time
                    # 在畫面上顯示動作
                    cv2.putText(img, "Light Kick", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 重踢：當左腳踝或腳跟高於左膝，且移動速度超過閾值
            elif ((left_ankle.y < left_knee.y - 0.02) or (left_heel.y < left_knee.y - 0.02)) and left_speed > speed_threshold:
                if current_action is None and (current_time - last_action_time) > 0.3:
                    keyboard.press('c')  # 重踢
                    time.sleep(0.05)
                    keyboard.release('c')
                    print("Heavy Kick")
                    current_action = "Heavy Kick"
                    last_action_time = current_time
                    # 在畫面上顯示動作
                    cv2.putText(img, "Heavy Kick", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # 當踢腿動作結束，重置動作狀態
                if current_action is not None and (current_time - last_action_time) > 0.3:
                    current_action = None

            # 新增的迴旋提判斷
            # 計算當前的身體朝向角度
            current_orientation = calculate_orientation(landmarks)

            if previous_orientation is not None:
                # 計算角度差
                angle_diff = current_orientation - previous_orientation

                # 調整角度差範圍到 [-180, 180]
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360

                # 檢查是否達到旋轉閾值
                if abs(angle_diff) > rotation_threshold:
                    # 檢查冷卻時間
                    if (current_time - last_action_time) > rotation_cooldown:
                        # 同時按下 'z' 和 'c' 鍵
                        keyboard.press('z')
                        keyboard.press('c')
                        time.sleep(0.05)
                        keyboard.release('z')
                        keyboard.release('c')
                        if angle_diff > 0:
                            # 向右旋轉
                            print("Spin Right")
                            cv2.putText(img, "Spin Right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            # 向左旋轉
                            print("Spin Left")
                            cv2.putText(img, "Spin Left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        current_action = "Spin"
                        last_action_time = current_time
            # 更新先前的朝向
            previous_orientation = current_orientation

            # 根據姿勢偵測結果，標記身體節點和骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 顯示影像
        cv2.imshow('Pose Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     # 按下 q 鍵停止

# 釋放資源
cap.release()
cv2.destroyAllWindows()
