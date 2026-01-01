import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import psutil
import math
from collections import deque

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_model_path = 'hand_landmarker.task'
face_model_path = 'face_landmarker.task'

if not os.path.exists(hand_model_path):
    print("Đang tải Hand Detection model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, hand_model_path)
    print("✓ Hand model tải thành công!")

if not os.path.exists(face_model_path):
    print("Đang tải Face Mesh model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, face_model_path)
    print("✓ Face model tải thành công!")

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

class VirtualButton:
    def __init__(self, x, y, w, h, text, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.clicked = False
    
    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     self.color, -1 if self.clicked else 2)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     (255, 255, 255), 2)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if self.clicked else (255, 255, 255), 2)
    
    def is_over(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h

buttons = [
    VirtualButton(50, 20, 100, 50, "CLEAR", (0, 255, 255)),
    VirtualButton(170, 20, 100, 50, "BLUE", (255, 0, 0)),
    VirtualButton(290, 20, 100, 50, "GREEN", (0, 255, 0)),
    VirtualButton(410, 20, 100, 50, "RED", (0, 0, 255))
]

canvas = None
drawing_color = (0, 255, 255)
draw_points = deque(maxlen=512)
is_drawing = False
prev_point = None
angle = 0
user_name = "JARVIS"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở Webcam")
    exit()

print("🚀 IRON MAN HUD SYSTEM ONLINE")
print("📝 Hướng dẫn:")
print("   - Giơ ngón trỏ + gập ngón giữa = VẼ")
print("   - Chụm ngón trỏ + ngón giữa trên nút = BẤM NÚT")
print("   - Nhấn 'q' = THOÁT")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Không nhận được tín hiệu hình ảnh")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, c), dtype=np.uint8)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    face_result = face_landmarker.detect_for_video(mp_image, frame_count)
    
    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        
        for i in [33, 133, 159, 145, 362, 263, 386, 374]:
            landmark = face_landmarks[i]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        right_eye = face_landmarks[33]
        eye_x = int(right_eye.x * w)
        eye_y = int(right_eye.y * h)
        
        for i, radius in enumerate([30, 40, 50]):
            angle_offset = angle + i * 120
            cv2.circle(frame, (eye_x, eye_y), radius, (0, 255, 255), 2)
            for j in range(0, 360, 30):
                rad = math.radians(j + angle_offset)
                px = int(eye_x + radius * math.cos(rad))
                py = int(eye_y + radius * math.sin(rad))
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)
        
        angle += 5
        if angle >= 360:
            angle = 0
    
    cpu_percent = psutil.cpu_percent(interval=0)
    hud_texts = [
        f"SYSTEM: ONLINE",
        f"USER: {user_name}",
        f"CPU: {cpu_percent:.1f}%",
        f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
    ]
    
    for i, text in enumerate(hud_texts):
        cv2.putText(frame, text, (20, h - 120 + i * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    hand_result = hand_landmarker.detect_for_video(mp_image, frame_count)
    
    if hand_result.hand_landmarks:
        hand_landmarks = hand_result.hand_landmarks[0]
        
        index_tip = hand_landmarks[8]
        index_pip = hand_landmarks[6]
        middle_tip = hand_landmarks[12]
        middle_pip = hand_landmarks[10]
        thumb_tip = hand_landmarks[4]
        
        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)
        middle_x = int(middle_tip.x * w)
        middle_y = int(middle_tip.y * h)
        
        pinch_distance = math.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
        
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
        cv2.circle(frame, (index_x, index_y), 12, (255, 255, 255), 2)
        
        if pinch_distance < 30:
            for button in buttons:
                if button.is_over(index_x, index_y):
                    button.clicked = True
                    
                    if button.text == "CLEAR":
                        canvas = np.zeros((h, w, c), dtype=np.uint8)
                        draw_points.clear()
                    elif button.text == "BLUE":
                        drawing_color = (255, 0, 0)
                    elif button.text == "GREEN":
                        drawing_color = (0, 255, 0)
                    elif button.text == "RED":
                        drawing_color = (0, 0, 255)
                else:
                    button.clicked = False
        else:
            for button in buttons:
                button.clicked = False
        
        index_up = index_tip.y < index_pip.y
        middle_down = middle_tip.y > middle_pip.y
        
        if index_up and middle_down and pinch_distance > 30:
            is_drawing = True
            draw_points.append((index_x, index_y))
            
            for i in range(1, len(draw_points)):
                if draw_points[i - 1] is None or draw_points[i] is None:
                    continue
                
                alpha = i / len(draw_points)
                thickness = int(5 * alpha) + 1
                
                cv2.line(canvas, draw_points[i - 1], draw_points[i], 
                        drawing_color, thickness)
        else:
            is_drawing = False
            if len(draw_points) > 0:
                draw_points.append(None)
    
    for button in buttons:
        button.draw(frame)
    
    frame = cv2.addWeighted(frame, 1, canvas, 0.7, 0)
    
    cv2.imshow('🦾 IRON MAN HUD - Air Canvas', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
face_landmarker.close()

print("✅ HUD System Offline. Goodbye!")