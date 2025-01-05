import cv2
import mediapipe as mp
import time
import json
import os
from pynput import keyboard
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt

class FingerKeyMonitor:
    def __init__(self, mapping_file='keyboard_layout.json'):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.current_finger_positions = {}
        
        # Add error handling for mapping file
        try:
            if not os.path.exists(mapping_file):
                raise FileNotFoundError(f"Mapping file '{mapping_file}' not found.")
            with open(mapping_file, 'r') as f:
                self.key_positions = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading keyboard layout: {e}")
            # Provide a basic fallback layout
            self.key_positions = {
                'a': {'x1': 100, 'y1': 200, 'x2': 130, 'y2': 230, 'finger': 'pinky'},
                's': {'x1': 140, 'y1': 200, 'x2': 170, 'y2': 230, 'finger': 'ring'}
            }
        
        self.resolution = self._determine_resolution()
        self.pressed_keys = set()
        self.key_usage = defaultdict(int)
        self.finger_usage = defaultdict(int)
        self.incorrect_finger_usage = defaultdict(int)
        self.recent_presses = deque(maxlen=10)
        self.start_time = time.time()
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.keyboard_listener.start()

    def _determine_resolution(self):
        try:
            max_x = max(pos['x2'] for pos in self.key_positions.values())
            max_y = max(pos['y2'] for pos in self.key_positions.values())
            return max(640, max_x + 100), max(480, max_y + 100)
        except (KeyError, ValueError):
            return 640, 480

    def _get_finger_position(self, hand_landmarks, tip_id, frame_shape):
        try:
            h, w = frame_shape[:2]
            cx = int(hand_landmarks.landmark[tip_id].x * w)
            cy = int(hand_landmarks.landmark[tip_id].y * h)
            return cx, cy
        except (AttributeError, IndexError):
            return None

    def on_key_press(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else str(key.name).lower()
            if key_char not in self.pressed_keys:
                self.pressed_keys.add(key_char)
                self.key_usage[key_char] += 1
                
                # Check if the key exists in layout and if we have finger positions
                if key_char in self.key_positions and self.current_finger_positions:
                    key_pos = self._get_center(self.key_positions[key_char])
                    
                    # Find nearest finger
                    distances = []
                    for idx, pos in self.current_finger_positions.items():
                        if pos is not None:  # Check if position is valid
                            dist = self._distance(key_pos, pos)
                            distances.append((idx, dist))
                    
                    if distances:  # Only process if we have valid distances
                        nearest_finger_idx = min(distances, key=lambda x: x[1])[0]
                        finger_name = self.finger_names[nearest_finger_idx]
                        self.finger_usage[finger_name] += 1
                        self.recent_presses.append((key_char, finger_name))
                        
                        # Check correct finger usage
                        correct_finger = self.key_positions[key_char].get('finger', '').lower()
                        if correct_finger and finger_name.lower() != correct_finger:
                            self.incorrect_finger_usage[finger_name] += 1
        except (AttributeError, KeyError, ValueError) as e:
            print(f"Error processing key press: {e}")

    def on_key_release(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else str(key.name).lower()
            self.pressed_keys.discard(key_char)
        except AttributeError:
            pass

    def _distance(self, pos1, pos2):
        try:
            return np.linalg.norm(np.array(pos1) - np.array(pos2))
        except (ValueError, TypeError):
            return float('inf')

    def _get_center(self, pos):
        try:
            return (int((pos['x1'] + pos['x2']) / 2), int((pos['y1'] + pos['y2']) / 2))
        except (KeyError, TypeError):
            return (0, 0)

    def process_frame(self, frame):
        if frame is None:
            return frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        self.current_finger_positions.clear()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Process finger positions
                    for idx, tip_id in enumerate(self.finger_tips):
                        pos = self._get_finger_position(hand_landmarks, tip_id, frame.shape)
                        if pos:
                            self.current_finger_positions[idx] = pos
                            cx, cy = pos
                            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                            cv2.putText(frame, self.finger_names[idx], 
                                      (cx + 10, cy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
                    continue

        try:
            self._draw_keyboard_overlay(frame)
            self._draw_statistics(frame)
        except Exception as e:
            print(f"Error drawing overlays: {e}")

        return frame

    def _draw_keyboard_overlay(self, frame):
        for key, pos in self.key_positions.items():
            try:
                intensity = min(255, self.key_usage[key] * 10)
                color = (0, 255, 0) if key in self.pressed_keys else (255 - intensity, 255 - intensity, 255)
                key_center = self._get_center(pos)
                cv2.circle(frame, key_center, 5, color, -1)
                cv2.putText(frame, key, 
                          (key_center[0] + 5, key_center[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.4, color, 1)
                
                if key in self.pressed_keys and self.current_finger_positions:
                    nearest = min(
                        [(idx, pos) for idx, pos in self.current_finger_positions.items()],
                        key=lambda x: self._distance(key_center, x[1]),
                        default=(None, None)
                    )
                    if nearest[1] is not None:
                        cv2.line(frame, key_center, nearest[1], (0, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing key overlay for {key}: {e}")
                continue

    def _draw_statistics(self, frame):
        try:
            h, w = frame.shape[:2]
            y_offset = 30
            
            # Draw recent keys
            cv2.putText(frame, "Recent Keys:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
            for key, finger in list(self.recent_presses)[-5:]:
                cv2.putText(frame, f"{key}: {finger}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Draw finger usage
            y_offset = 30
            sorted_finger_usage = sorted(self.finger_usage.items(), key=lambda x: x[1], reverse=True)
            for finger, count in sorted_finger_usage:
                cv2.putText(frame, f"{finger}: {count}", (w - 150, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Draw session time
            duration = int(time.time() - self.start_time)
            cv2.putText(frame, f"Session Time: {duration}s", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error drawing statistics: {e}")
            
    def display_summary(self):
        print("\nSession Summary:")
        print("Key Usage:")
        for key, count in self.key_usage.items():
            print(f"  {key}: {count}")
        print("\nFinger Usage:")
        for finger, count in self.finger_usage.items():
            print(f"  {finger}: {count}")
        print("\nIncorrect Finger Usage:")
        for finger, count in self.incorrect_finger_usage.items():
            print(f"  {finger}: {count}")
        self.generate_graphs()

    def generate_graphs(self):
        key_labels = list(self.key_usage.keys())
        key_counts = list(self.key_usage.values())
        plt.figure(figsize=(10, 6))
        plt.pie(key_counts, labels=key_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Key Usage Distribution')
        plt.show()
        finger_labels = list(self.finger_usage.keys())
        finger_counts = list(self.finger_usage.values())
        plt.figure(figsize=(10, 6))
        plt.pie(finger_counts, labels=finger_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Finger Usage Distribution')
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.bar(key_labels, key_counts, color='skyblue')
        plt.title('Key Usage Frequency')
        plt.xlabel('Keys')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.bar(finger_labels, finger_counts, color='lightcoral')
        plt.title('Finger Usage Frequency')
        plt.xlabel('Fingers')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        incorrect_finger_labels = list(self.incorrect_finger_usage.keys())
        incorrect_finger_counts = list(self.incorrect_finger_usage.values())
        plt.figure(figsize=(10, 6))
        plt.pie(incorrect_finger_counts, labels=incorrect_finger_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Incorrect Finger Usage Distribution')
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.bar(incorrect_finger_labels, incorrect_finger_counts, color='purple')
        plt.title('Incorrect Finger Usage Frequency')
        plt.xlabel('Fingers')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()




def main():
    print("Starting finger tracking monitor...")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    monitor = FingerKeyMonitor()

    # Set the resolution based on the key layout
    resolution = monitor.resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.keyboard_listener.stop()
        monitor.display_summary()


if __name__ == "__main__":
    main()
