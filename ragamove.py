import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pynput.keyboard import Key, Controller

# 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
keyboard = Controller()

# 設定參數
TILT_THRESHOLD = 10  # 傾斜角度閾值

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

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 轉換影像格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 取得關鍵點
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # 判斷面向方向
        facing = calculate_facing_direction(left_shoulder, right_shoulder)
        
        # 計算傾斜角度
        if facing == "right":
            tilt_angle = calculate_tilt_angle(right_hip, right_shoulder)
        else:
            tilt_angle = calculate_tilt_angle(left_hip, left_shoulder)

        # 處理傾斜並觸發按鍵
        direction = process_tilt(tilt_angle, facing)
        if direction:
            if direction == "right":
                print("Move right")
                keyboard.press(Key.right)
                time.sleep(0.15)  # 短暫延遲確保按鍵被偵測
                keyboard.release(Key.right)
            elif direction == "left":
                print("Move left")
                keyboard.press(Key.left)
                time.sleep(0.15)  # 短暫延遲確保按鍵被偵測
                keyboard.release(Key.left)
            else:
                print("No movement")


        # 顯示資訊
        info_color = (0, 255, 0)
        cv2.putText(image, f"Facing: {facing}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, info_color, 2)
        cv2.putText(image, f"Tilt: {tilt_angle:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, info_color, 2)
        if direction:
            cv2.putText(image, f"Direction: {direction}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, info_color, 2)

        # 繪製骨架
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 顯示影像
    cv2.imshow('Pose Detection', image)
    
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()