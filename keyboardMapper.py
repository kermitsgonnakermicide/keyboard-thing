import cv2
import mediapipe as mp
import time
import json
from pynput import keyboard
import numpy as np
from collections import defaultdict

class AutoKeyboardMapper:
    def __init__(self):
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
        self.current_finger_positions = {}
        
        # Key position learning
        self.key_positions = defaultdict(list)  # Stores all positions for each key
        self.final_key_map = {}  # Will store the averaged positions
        self.required_samples = 3  # Number of samples needed for each key
        self.learning_complete = False
        
        # Visualization
        self.debug_frame = None
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press)
        self.keyboard_listener.start()

    def on_key_press(self, key):
        try:
            # Get key character
            key_char = key.char.lower() if hasattr(key, 'char') else key.name.lower()
            
            # If we have current finger positions, record them
            if self.current_finger_positions and key_char:
                # Find the finger closest to where the key was pressed
                positions = list(self.current_finger_positions.values())
                if positions:
                    # Store the position for this key
                    self.key_positions[key_char].append(positions[0])  # Store the most likely finger position
                    samples_collected = len(self.key_positions[key_char])
                    
                    # Draw debug info
                    if self.debug_frame is not None:
                        cv2.putText(self.debug_frame, 
                                  f"Recorded {key_char}: {samples_collected}/{self.required_samples}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    print(f"Recorded position for key '{key_char}' ({samples_collected}/{self.required_samples})")
                    
                    # If we have enough samples, calculate average position
                    if samples_collected >= self.required_samples:
                        positions = self.key_positions[key_char]
                        avg_x = sum(p[0] for p in positions) / len(positions)
                        avg_y = sum(p[1] for p in positions) / len(positions)
                        self.final_key_map[key_char] = (int(avg_x), int(avg_y))
                        print(f"Finalized position for key '{key_char}'")
        except AttributeError:
            pass

    def process_frame(self, frame):
        self.debug_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        self.current_finger_positions.clear()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    self.debug_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get finger tip positions
                h, w, _ = frame.shape
                for idx, tip_id in enumerate(self.finger_tips):
                    cx = int(hand_landmarks.landmark[tip_id].x * w)
                    cy = int(hand_landmarks.landmark[tip_id].y * h)
                    self.current_finger_positions[f"finger_{idx}"] = (cx, cy)
                    
                    # Draw finger tips
                    cv2.circle(self.debug_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
        # Draw current key map
        for key, pos in self.final_key_map.items():
            cv2.circle(self.debug_frame, pos, 3, (0, 255, 0), -1)
            cv2.putText(self.debug_frame, key, (pos[0] + 5, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(self.debug_frame, "Press each key 3 times to map its position", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        cv2.putText(self.debug_frame, "Press 'S' to save and quit", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return self.debug_frame

    def save_mapping(self):
        if self.final_key_map:
            with open('keyboard_layout.json', 'w') as f:
                json.dump(self.final_key_map, f, indent=4)
            print("Keyboard mapping saved to 'keyboard_layout.json'")
            return True
        return False

def main():
    print("Starting automatic keyboard mapping...")
    print("Press each key 3 times while your hands are visible to the camera")
    print("Press 'S' to save the mapping and quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Initialize mapper
    mapper = AutoKeyboardMapper()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame
        debug_frame = mapper.process_frame(frame)
        cv2.imshow('Automatic Keyboard Mapper', debug_frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            if mapper.save_mapping():
                print("Mapping saved successfully!")
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mapper.keyboard_listener.stop()

if __name__ == "__main__":
    main()