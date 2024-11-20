import cv2
import pyautogui
import mediapipe as mp
import numpy as np
import time
import keyboard  # 需要先安裝: pip install keyboard

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

print("start the game (Press 'q' to quit)")
try:
    while True:
        # 檢查是否按下 'q' 鍵
        if keyboard.is_pressed('q'):
            print("Stopping the program...")
            break
            
        # 擷取主螢幕的全螢幕畫面
        screenshot = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 進行姿勢偵測
        results = pose.process(frame)

        # 如果偵測到姿勢
        if results.pose_landmarks:
            print("Person detected")
            
            # 取得頭部位置（使用鼻子作為參考點）
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            head_x = int(nose.x * frame.shape[1])
            head_y = int(nose.y * frame.shape[0])
            
            # 移動滑鼠到頭部位置
            pyautogui.moveTo(head_x, head_y)
            pyautogui.click()
        else:
            print("No person detected")

except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    # 釋放資源
    pose.close()
    print("Program ended")