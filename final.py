import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pynput.keyboard import Key, Controller

# 初始化 Mediapipe 模組
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 初始化鍵盤控制器
keyboard = Controller()

# 設定參數
TILT_THRESHOLD = 10  # 傾斜角度閾值
ROTATION_THRESHOLD = 15  # 旋轉角度閾值（度）
ROTATION_COOLDOWN = 1.0  # 旋轉動作之間的最小間隔時間（秒）
ACTION_COOLDOWN = 0.2  # 一般動作之間的最小間隔時間（秒）

# 動作狀態追蹤
last_action_time = time.time()
current_action = None
previous_orientation = None

# 用於腿部識別
previous_left_knee_y = None
previous_right_knee_y = None
light_kick_threshold = 0.05  # 左腳輕踢的高度閾值
heavy_kick_threshold = 0.05  # 右腳重踢的高度閾值
speed_threshold = 0.02      # 抬高速度閾值

# 初始化攝影機
cap = cv2.VideoCapture(0)

def calculate_facing_direction(shoulder_left, shoulder_right):
    """判斷面向方向"""
    return "right" if shoulder_right.z < shoulder_left.z else "left"

def calculate_tilt_angle(hip, shoulder):
    """
    計算與垂直線的夾角
    垂直站立時為0度
    向前傾斜為正數
    向後傾斜為負數
    """
    dx = shoulder.x - hip.x
    dy = shoulder.y - hip.y
    angle = math.degrees(math.atan2(dx, -dy))
    return angle

def process_tilt(tilt_angle, facing):
    """處理傾斜角度並決定方向"""
    if abs(tilt_angle) > TILT_THRESHOLD:
        if facing == "right":
            if tilt_angle > 0:  # 向前傾
                return "right"
            else:  # 向後傾
                return "left"
        else:  # facing left
            if tilt_angle > 0:  # 向前傾
                return "right"
            else:  # 向後傾
                return "left"
    return None

def calculate_orientation(landmarks):
    """計算身體朝向角度"""
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
        action_triggered = False  # 用於避免多重觸發

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 取得關鍵點
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            
            # ------------------ Movement Module ------------------
            # 判斷面向方向
            facing = calculate_facing_direction(left_shoulder, right_shoulder)
            
            # 計算傾斜角度
            if facing == "right":
                tilt_angle = calculate_tilt_angle(right_hip, right_shoulder)
            else:
                tilt_angle = calculate_tilt_angle(left_hip, left_shoulder)
            
            # 處理傾斜並觸發按鍵
            direction = process_tilt(tilt_angle, facing)
            if direction and (current_time - last_action_time) > ACTION_COOLDOWN:
                if direction == "right":
                    print("Move right")
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                elif direction == "left":
                    print("Move left")
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                last_action_time = current_time
                action_triggered = True
            
            # ------------------ Hand Gesture Module ------------------
            # 優先處理特殊動作
            # 昇龍拳：當右手高於右肩的 y 軸位置
            if right_wrist.y < right_shoulder.y - 0.05 and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('a')  
                    keyboard.press('d')
                    keyboard.press(Key.down)
                    keyboard.release('a')
                    keyboard.release('d')
                    keyboard.release(Key.down)  
                    print("Shoryuken")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Shoryuken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 氣力射出：當左手高於左肩的 y 軸位置
            elif left_wrist.y < left_shoulder.y and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('a')  
                    keyboard.press('d')
                    keyboard.release('a')
                    keyboard.release('d')
                    print("Ki Blast")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Ki Blast", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 輕拳：當右手位置在右肩前方
            elif right_wrist.x < right_shoulder.x - 0.1 and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('a')
                    keyboard.release('a')
                    print("Light Punch")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Light Punch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 重拳：當左手位置在左肩前方
            elif left_wrist.x > left_shoulder.x + 0.1 and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('d')
                    keyboard.release('d')
                    print("Heavy Punch")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Heavy Punch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # ------------------ Leg/Kick Detection Module ------------------
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
            if left_knee_raise > light_kick_threshold and left_knee_speed > speed_threshold and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('z')  # 左腳輕踢
                    keyboard.release('z')
                    print("Left Light Kick")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Left Light Kick", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 右腳重踢判斷
            if right_knee_raise > heavy_kick_threshold and right_knee_speed > speed_threshold and not action_triggered:
                if (current_time - last_action_time) > ACTION_COOLDOWN:
                    keyboard.press('c')  # 右腳重踢
                    keyboard.release('c')
                    print("Right Heavy Kick")
                    last_action_time = current_time
                    action_triggered = True
                    cv2.putText(img, "Right Heavy Kick", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # ------------------ Rotation Detection Module ------------------
            # 計算身體朝向角度
            current_orientation = calculate_orientation(landmarks)
            if previous_orientation is not None:
                angle_diff = current_orientation - previous_orientation
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360

                if abs(angle_diff) > ROTATION_THRESHOLD and (current_time - last_action_time) > ROTATION_COOLDOWN and not action_triggered:
                    if angle_diff > 0:
                        # Spin Right
                        keyboard.press('z')
                        keyboard.press('c')
                        keyboard.release('z')
                        keyboard.release('c')
                        print("Spin Right")
                        cv2.putText(img, "Spin Right", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        # Spin Left
                        keyboard.press('z')
                        keyboard.press('c')
                        keyboard.release('z')
                        keyboard.release('c')
                        print("Spin Left")
                        cv2.putText(img, "Spin Left", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    last_action_time = current_time
                    action_triggered = True

            previous_orientation = current_orientation

            # 重置動作狀態
            if action_triggered:
                current_action = None

            # 顯示傾斜和面向資訊
            cv2.putText(img, f"Facing: {facing}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Tilt: {tilt_angle:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 繪製骨架
            mp_draw.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # 顯示影像
        cv2.imshow('Pose Detection', img)
        
        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
