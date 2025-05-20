import cv2 
import mediapipe as mp
import pygame
import numpy as np
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

pygame.init()
pygame.mixer.init()
shot_sound = pygame.mixer.Sound("shot.wav")  # Make sure this file exists

duck_img = cv2.imread("duck.png", cv2.IMREAD_UNCHANGED)
duck_img = cv2.resize(duck_img, (100, 100))

# Webcam setup
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load and resize background
bg_img = cv2.imread("background.jpg")
bg_img = cv2.resize(bg_img, (frame_w, frame_h))

def overlay_transparent(bg, overlay, x, y):
    bg_h, bg_w = bg.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h:
        return bg
    x_start = max(x, 0)
    y_start = max(y, 0)
    w = min(ol_w - max(0, -x), bg_w - x_start)
    h = min(ol_h - max(0, -y), bg_h - y_start)
    if w <= 0 or h <= 0:
        return bg
    overlay_roi = overlay[max(0, -y):max(0, -y)+h, max(0, -x):max(0, -x)+w]
    bg_roi = bg[y_start:y_start+h, x_start:x_start+w]
    if overlay.shape[2] < 4:
        bg[y_start:y_start+h, x_start:x_start+w] = overlay_roi
        return bg
    alpha = overlay_roi[:, :, 3] / 255.0
    for c in range(3):
        bg_roi[:, :, c] = alpha * overlay_roi[:, :, c] + (1 - alpha) * bg_roi[:, :, c]
    bg[y_start:y_start+h, x_start:x_start+w] = bg_roi
    return bg

class Duck:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.randint(50, frame_w - 150)
        self.y = random.randint(50, frame_h - 150)
        self.width = 100
        self.height = 100
        self.appear_time = time.time()

    def draw(self, frame):
        return overlay_transparent(frame, duck_img, int(self.x), int(self.y))

    def should_respawn(self):
        return time.time() - self.appear_time > random.uniform(1.5, 3.5)

duck = Duck()
score = 0
cooldown = 0

cv2.namedWindow("Shoot the Duck 🦆 - Press Q to Quit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Shoot the Duck 🦆 - Press Q to Quit", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    background = bg_img.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    background = duck.draw(background)

    if results.multi_hand_landmarks:
        index = results.multi_hand_landmarks[0].landmark[8]
        x = int(index.x * frame_w)
        y = int(index.y * frame_h)

        # Draw crosshair
        cv2.circle(background, (x, y), 10, (0, 0, 0), 2)
        cv2.line(background, (x - 15, y), (x + 15, y), (0, 0, 0), 2)
        cv2.line(background, (x, y - 15), (x, y + 15), (0, 0, 0), 2)

        if cooldown == 0 and duck.x < x < duck.x + duck.width and duck.y < y < duck.y + duck.height:
            score += 1
            shot_sound.play()
            duck.respawn()
            cooldown = 10

    if cooldown > 0:
        cooldown -= 1

    if duck.should_respawn():
        duck.respawn()

    cv2.putText(background, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 3)

    cv2.imshow("Shoot the Duck 🦆 - Press Q to Quit", background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
