import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

cap = cv2.VideoCapture(0)

keyboard = Controller()  # 初始化 pynput 的鍵盤控制器

last_action_time = time.time()  # 最後一次動作的時間
current_action = None  # 用於追蹤當前的動作狀態

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img, (520, 300))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # 將 BGR 轉換成 RGB
        results = pose.process(img2)                    # 取得姿勢偵測結果

        current_time = time.time()

        if results.pose_landmarks:
            # 取得左右手帶點位置
            landmarks = results.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            # 優先處理特殊動作
            # 昇龍拳：當右手高於右肩的 y 軸位置
            if right_wrist.y < right_shoulder.y - 0.05:  # 增加容錯，減少微小抖動影響
                if current_action == None and (current_time - last_action_time) > 0.5:
                    keyboard.press('a')  
                    keyboard.press('d')
                    keyboard.press(Key.down)
                    time.sleep(0.05)  # 短暫延遲，確保所有按鍵被識別
                    keyboard.release('a')
                    keyboard.release('d')
                    keyboard.release(Key.down)  
                    print("Shoryuken")
                    current_action = "Shoryuken"
                    last_action_time = current_time
                    time.sleep(0.1)

            # 氣力射出：當左手高於左肩的 y 軸位置
            elif left_wrist.y < left_shoulder.y:
                if current_action == None and (current_time - last_action_time) > 0.5:
                    keyboard.press('a')  
                    keyboard.press('d')
                    time.sleep(0.05)  # 短暫延遲，確保所有按鍵被識別
                    keyboard.release('a')
                    keyboard.release('d')
                    print("Ki Blast")
                    current_action = "Ki Blast"
                    last_action_time = current_time
                    time.sleep(0.1)

            # 輕拳：當右手位置在右肩前方
            elif right_wrist.x < right_shoulder.x - 0.1:
                if current_action == None and (current_time - last_action_time) > 0.5:
                    keyboard.press('a')
                    time.sleep(0.05)
                    keyboard.release('a')
                    print("Light Punch")
                    current_action = "Light Punch"
                    last_action_time = current_time
                    time.sleep(0.1)

            # 重拳：當左手位置在左肩前方
            elif left_wrist.x > left_shoulder.x + 0.1:
                if current_action == None and (current_time - last_action_time) > 0.5:
                    keyboard.press('d')
                    time.sleep(0.05)
                    keyboard.release('d')
                    print("Heavy Punch")
                    current_action = "Heavy Punch"
                    last_action_time = current_time
                    time.sleep(0.1)

            elif current_action != None:
                current_action = None

        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     # 按下 q 鍵停止

cap.release()
cv2.destroyAllWindows()

