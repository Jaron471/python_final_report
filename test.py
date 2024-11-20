import cv2
import pyautogui
from ultralyticsplus import YOLO
import numpy as np
import time

# 初始化 YOLO 模型
model = YOLO('yolov5s')

while True:
    # 擷取螢幕畫面
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 YOLO 模型進行物件偵測
    results = model(frame)

    # 遍歷所有偵測到的物件
    for result in results:
        if result['label'] == 'person':
            # 取得人像的邊界框
            x1, y1, x2, y2 = result['bbox']
            # 計算頭部位置
            head_x = (x1 + x2) // 2
            head_y = y1

            # 移動滑鼠到頭部位置並點擊
            pyautogui.moveTo(head_x, head_y)
            pyautogui.click()

    # 顯示畫面 (可選)
    cv2.imshow('Frame', frame)

    # 按下 'L' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('L'):
        break

    # 短暫延遲
    time.sleep(0.1)

# 關閉所有視窗
cv2.destroyAllWindows()