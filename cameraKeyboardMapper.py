import cv2
import mediapipe as mp
import json
import numpy as np
from collections import defaultdict
from pynput import keyboard

class KeyboardMapper:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.key_map = {}
        self.layout_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.samples_per_key = 3
        self.key_samples = defaultdict(list)
        self.running = True
        self.last_key_time = 0
        self.last_key_text = ""
        self.finger_used_text = ""
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tracking
        self.finger_tips = [4, 8, 12, 16, 20]  # thumb to pinky landmarks
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.current_finger_positions = {}
        
        # Key name mappings for special keys
        self.key_name_map = {
            'print_screen': 'prtsc',
            'shift': 'lshift',
            'shift_r': 'rshift',
            'ctrl': 'lctrl',
            'ctrl_r': 'rctrl',
            'alt': 'lalt',
            'alt_r': 'ralt',
            'space_bar': 'space',
            'page_up': 'pgup',
            'page_down': 'pgdn',
            'backspace_bar': 'backspace'
        }
        
        # Store keys we still need to map
        self.keys_needed = set([
            'esc', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 
            'f11', 'f12', 'prtsc', 'scrolllock', 'pause', '`', '1', '2', '3', '4', 
            '5', '6', '7', '8', '9', '0', '-', '=', 'backspace', 'insert', 'home', 
            'pgup', 'tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', 
            ']', '\\', 'delete', 'end', 'pgdn', 'caps', 'a', 's', 'd', 'f', 'g', 
            'h', 'j', 'k', 'l', ';', "'", 'enter', 'lshift', 'z', 'x', 'c', 'v', 
            'b', 'n', 'm', ',', '.', '/', 'rshift', 'up', 'lctrl', 'win', 'lalt', 
            'space', 'ralt', 'fn', 'menu', 'rctrl', 'left', 'down', 'right'
        ])
        
        # Initialize keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

    def find_closest_finger_to_screen_center(self):
        if not self.current_finger_positions:
            return None, None
            
        screen_center = (self.width // 2, self.height // 2)
        closest_finger = None
        closest_dist = float('inf')
        closest_pos = None
        
        for finger_idx, pos in self.current_finger_positions.items():
            dist = np.sqrt((pos[0] - screen_center[0])**2 + (pos[1] - screen_center[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_finger = int(finger_idx.split('_')[1])  # Get numeric index from 'finger_X'
                closest_pos = pos
                
        return closest_finger, closest_pos

    def on_key_press(self, key):
        try:
            # Convert key to string representation
            if hasattr(key, 'char'):
                key_char = key.char.lower() if key.char else None
            elif hasattr(key, 'name'):
                key_char = key.name.lower()
            else:
                return
                
            # Map special key names
            key_char = self.key_name_map.get(key_char, key_char)
            
            # If this key is one we need to map
            if key_char in self.keys_needed:
                finger_idx, pos = self.find_closest_finger_to_screen_center()
                if finger_idx is not None:
                    self.finger_used_text = f"Using {self.finger_names[finger_idx]} finger"
                    if self.record_position(key_char, pos):
                        self.keys_needed.remove(key_char)
                        remaining = len(self.keys_needed)
                        print(f"\nMapped '{key_char}' with {self.finger_names[finger_idx]} finger. {remaining} keys remaining.")
                        if remaining > 0:
                            print("\nNeeded keys:", ', '.join(sorted(self.keys_needed)))
            
            # Update last key press info for visualization
            self.last_key_time = cv2.getTickCount()
            self.last_key_text = f"Detected: {key_char}"
            
        except AttributeError:
            pass
        
        # Check for quit command
        if hasattr(key, 'char') and key.char == 'q':
            self.running = False

    def record_position(self, key_pressed, position):
        if position:
            self.key_samples[key_pressed].append(position)
            samples_count = len(self.key_samples[key_pressed])
            print(f"Recorded position {samples_count}/{self.samples_per_key} for '{key_pressed}'")
            
            if samples_count >= self.samples_per_key:
                # Calculate average position
                positions = self.key_samples[key_pressed]
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                self.key_map[key_pressed] = (int(avg_x), int(avg_y))
                return True
        return False

    def get_finger_positions(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        self.current_finger_positions.clear()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get finger tip positions
                h, w, _ = frame.shape
                for idx, tip_id in enumerate(self.finger_tips):
                    cx = int(hand_landmarks.landmark[tip_id].x * w)
                    cy = int(hand_landmarks.landmark[tip_id].y * h)
                    self.current_finger_positions[f"finger_{idx}"] = (cx, cy)
                    
                    # Draw finger tips with names
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, self.finger_names[idx], (cx + 5, cy - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return frame

    def draw_overlay(self, frame):
        # Draw all mapped keys
        for key, pos in self.key_map.items():
            cv2.circle(frame, pos, 3, (0, 255, 0), -1)
            cv2.putText(frame, key, (pos[0] + 5, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw number of remaining keys
        remaining = len(self.keys_needed)
        cv2.putText(frame, 
                   f"Remaining keys: {remaining}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Draw last detected key
        if (cv2.getTickCount() - self.last_key_time) / cv2.getTickFrequency() < 1.0:
            cv2.putText(frame, self.last_key_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 255), 2)
            cv2.putText(frame, self.finger_used_text,
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press each key 3 times with any finger", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'S' to save, 'R' to reset, 'Q' to quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Press any key to begin mapping")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get and draw hand landmarks
            frame = self.get_finger_positions(frame)
            
            # Draw overlay
            frame = self.draw_overlay(frame)
            
            cv2.imshow('Keyboard Mapper', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_mapping()
                break
            elif key == ord('r'):
                self.reset_mapping()
        
        cap.release()
        cv2.destroyAllWindows()
        self.keyboard_listener.stop()

    def save_mapping(self):
        with open('keyboard_layout.json', 'w') as f:
            json.dump(self.key_map, f, indent=4)
        print("Keyboard mapping saved to 'keyboard_layout.json'")

    def reset_mapping(self):
        self.key_map.clear()
        self.key_samples.clear()
        self.keys_needed = set(self.keys_needed)
        print("Mapping reset. Press any key to begin")

def main():
    try:
        mapper = KeyboardMapper()
        mapper.run()
    except KeyboardInterrupt:
        print("\nKeyboard mapping interrupted.")
        mapper.keyboard_listener.stop()

if __name__ == "__main__":
    main()