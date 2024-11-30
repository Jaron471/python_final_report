import cv2
import mediapipe as mp
import pygame
import numpy as np
from enum import Enum

class PlayerState(Enum):
    IDLE = 1
    DEFENSE = 2
    ATTACK = 3

class Direction(Enum):
    LEFT = -1
    RIGHT = 1

class Player:
    def __init__(self, x, y, width, height, color, is_left_player=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.state = PlayerState.IDLE
        self.health = 100
        self.speed = 5
        self.is_left_player = is_left_player
        self.direction = Direction.RIGHT if is_left_player else Direction.LEFT
        self.y_speed = 5

    def move(self, dx, dy):
        # 水平移動
        self.rect.x += dx
        if dx > 0:
            self.direction = Direction.RIGHT
        elif dx < 0:
            self.direction = Direction.LEFT

        # 垂直移動
        self.rect.y += dy
        
        # 確保玩家不會超出畫面
        screen = pygame.display.get_surface().get_rect()
        # 限制水平移動
        if self.rect.left < screen.left:
            self.rect.left = screen.left
        if self.rect.right > screen.right:
            self.rect.right = screen.right
        # 限制垂直移動
        if self.rect.top < screen.top:
            self.rect.top = screen.top
        if self.rect.bottom > screen.bottom:
            self.rect.bottom = screen.bottom

    def draw(self, surface):
        # 繪製角色
        pygame.draw.rect(surface, self.color, self.rect)
        
        # 繪製方向指示器（三角形）
        direction_indicator = []
        if self.direction == Direction.RIGHT:
            direction_indicator = [
                (self.rect.right, self.rect.centery),
                (self.rect.right - 10, self.rect.centery - 10),
                (self.rect.right - 10, self.rect.centery + 10)
            ]
        else:
            direction_indicator = [
                (self.rect.left, self.rect.centery),
                (self.rect.left + 10, self.rect.centery - 10),
                (self.rect.left + 10, self.rect.centery + 10)
            ]
        pygame.draw.polygon(surface, (255, 255, 0), direction_indicator)
        
        # 繪製血條
        health_bar = pygame.Rect(self.rect.x, self.rect.y - 20, 
                               self.rect.width * (self.health/100), 10)
        pygame.draw.rect(surface, (255, 0, 0), health_bar)

    def update_from_hand(self, hand_landmarks, is_left_hand):
        # 確認這是否是對應的控制手
        if self.is_left_player != is_left_hand:
            return
            
        # 更新位置
        hand_y = hand_landmarks.landmark[0].y
        hand_x = hand_landmarks.landmark[0].x
        
        # 計算移動量
        dx = (hand_x - 0.5) * self.speed * 10
        dy = (hand_y - 0.5) * self.y_speed * 10
        
        # 更新位置和方向
        self.move(dx, dy)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        
        # 初始化 MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        
        # 初始化攝像頭
        self.cap = cv2.VideoCapture(0)
        
        # 初始化 frame
        self.frame = None
        
        # 創建玩家
        self.player1 = Player(100, 400, 50, 100, (255, 0, 0), True)
        self.player2 = Player(650, 400, 50, 100, (0, 255, 0), False)

    def detect_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        
        if distance < 0.1:
            return PlayerState.DEFENSE
        else:
            return PlayerState.ATTACK

    def process_hands(self, results, frame):
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 判斷是左手還是右手
                is_left = handedness.classification[0].label == "Left"
                
                # 根據手的類型更新對應的玩家
                if (is_left and self.player1.is_left_player) or (not is_left and not self.player1.is_left_player):
                    player = self.player1
                else:
                    player = self.player2
                
                # 更新玩家狀態
                player.state = self.detect_gesture(hand_landmarks)
                player.update_from_hand(hand_landmarks, is_left)
                
                # 繪製骨架
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 處理手勢辨識
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.frame = frame  # 儲存當前 frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # 處理手部偵測結果
                self.process_hands(results, frame)
                
                # 顯示攝像頭畫面
                cv2.imshow('Camera Feed', frame)

            # 更新遊戲邏輯
            self.update_game_logic()

            # 繪製遊戲畫面
            self.screen.fill((255, 255, 255))
            self.player1.draw(self.screen)
            self.player2.draw(self.screen)
            pygame.display.flip()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            
            self.clock.tick(60)

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    def update_game_logic(self):
        if self.player1.state == PlayerState.ATTACK and self.player2.state != PlayerState.DEFENSE:
            if abs(self.player1.rect.x - self.player2.rect.x) < 100:
                self.player2.health -= 1
        
        if self.player2.state == PlayerState.ATTACK and self.player1.state != PlayerState.DEFENSE:
            if abs(self.player1.rect.x - self.player2.rect.x) < 100:
                self.player1.health -= 1

if __name__ == "__main__":
    game = Game()
    game.run()