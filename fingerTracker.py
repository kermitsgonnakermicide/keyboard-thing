import cv2
import mediapipe as mp
import time
import json
from pynput import keyboard
import numpy as np
from collections import deque, defaultdict

class FingerKeyMonitor:
    def __init__(self, mapping_file='keyboard_layout.json'):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize finger tracking
        self.finger_tips = [4, 8, 12, 16, 20]  # thumb to pinky landmarks
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.current_finger_positions = {}
        
        # Load keyboard mapping
        with open(mapping_file, 'r') as f:
            self.key_positions = json.load(f)
        print(f"Loaded {len(self.key_positions)} key positions")
        
        # Tracking statistics
        self.pressed_keys = set()
        self.key_usage = defaultdict(int)
        self.finger_usage = defaultdict(int)
        self.recent_presses = deque(maxlen=10)
        self.start_time = time.time()
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.keyboard_listener.start()

    def on_key_press(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else key.name.lower()
            if key_char not in self.pressed_keys:
                self.pressed_keys.add(key_char)
                self.key_usage[key_char] += 1
                
                # Find nearest finger
                if key_char in self.key_positions and self.current_finger_positions:
                    key_pos = self.key_positions[key_char]
                    nearest_finger = min(
                        enumerate(self.current_finger_positions.items()),
                        key=lambda x: self._distance(key_pos, x[1][1])
                    )
                    finger_idx = nearest_finger[0]
                    finger_name = self.finger_names[finger_idx]
                    self.finger_usage[finger_name] += 1
                    self.recent_presses.append((key_char, finger_name))
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else key.name.lower()
            self.pressed_keys.discard(key_char)
        except AttributeError:
            pass

    def _distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def process_frame(self, frame):
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Clear current positions
        self.current_finger_positions.clear()
        
        # Draw hands and get finger positions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get finger positions
                h, w, _ = frame.shape
                for idx, tip_id in enumerate(self.finger_tips):
                    cx = int(hand_landmarks.landmark[tip_id].x * w)
                    cy = int(hand_landmarks.landmark[tip_id].y * h)
                    self.current_finger_positions[idx] = (cx, cy)
                    
                    # Draw finger tips with labels
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.putText(frame, self.finger_names[idx], (cx + 10, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw keyboard overlay
        self._draw_keyboard_overlay(frame)
        
        # Draw statistics
        self._draw_statistics(frame)
        
        return frame

    def _draw_keyboard_overlay(self, frame):
        # Draw all key positions
        for key, pos in self.key_positions.items():
            # Color based on usage intensity
            intensity = min(255, self.key_usage[key] * 10)
            color = (0, 255, 0) if key in self.pressed_keys else (255-intensity, 255-intensity, 255)
            
            cv2.circle(frame, pos, 5, color, -1)
            cv2.putText(frame, key, (pos[0] + 5, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw lines to nearest fingers for pressed keys
            if key in self.pressed_keys and self.current_finger_positions:
                nearest = min(
                    self.current_finger_positions.items(),
                    key=lambda x: self._distance(pos, x[1])
                )
                cv2.line(frame, pos, nearest[1], (0, 255, 255), 2)

    def _draw_statistics(self, frame):
        h, w, _ = frame.shape
        
        # Draw recent key presses
        y_offset = 30
        cv2.putText(frame, "Recent Keys:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        for key, finger in list(self.recent_presses)[-5:]:
            cv2.putText(frame, f"{key}: {finger}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw finger usage statistics
        y_offset = 30
        for finger, count in sorted(self.finger_usage.items(), key=lambda x: x[1], reverse=True):
            cv2.putText(frame, f"{finger}: {count}", (w - 150, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw session duration
        duration = int(time.time() - self.start_time)
        cv2.putText(frame, f"Session Time: {duration}s", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    print("Starting finger tracking monitor...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Initialize monitor
    try:
        monitor = FingerKeyMonitor()
    except FileNotFoundError:
        print("No keyboard mapping file found. Please run the mapper first.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process and display frame
        frame = monitor.process_frame(frame)
        cv2.imshow('Finger-Key Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    monitor.keyboard_listener.stop()

if __name__ == "__main__":
    main()