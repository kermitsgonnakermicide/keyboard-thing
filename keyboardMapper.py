import cv2
import json
import numpy as np

class KeyboardMapper:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.key_map = {}
        self.layout_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Drawing state
        self.start_pos = None
        self.current_rect = None
        self.drawing = False
        
        # Standard 60% keyboard keys in order
        self.keys_to_map = [
            'esc', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'backspace',
            'tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\',
            'caps', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", 'enter',
            'lshift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'rshift',
            'lctrl', 'win', 'lalt', 'space', 'ralt', 'fn', 'menu', 'rctrl'
        ]
        self.current_key_idx = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pos = (x, y)
            self.drawing = True
            self.current_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_rect = (self.start_pos[0], self.start_pos[1], x, y)
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.current_key_idx < len(self.keys_to_map):
                current_key = self.keys_to_map[self.current_key_idx]
                self.key_map[current_key] = {
                    'x1': min(self.start_pos[0], x),
                    'y1': min(self.start_pos[1], y),
                    'x2': max(self.start_pos[0], x),
                    'y2': max(self.start_pos[1], y)
                }
                
                # Draw the rectangle permanently
                cv2.rectangle(self.layout_image, 
                              (self.key_map[current_key]['x1'], self.key_map[current_key]['y1']),
                              (self.key_map[current_key]['x2'], self.key_map[current_key]['y2']),
                              (0, 255, 0), 1)
                
                # Add key label
                cv2.putText(self.layout_image, current_key,
                            (self.key_map[current_key]['x1'] + 5, 
                             self.key_map[current_key]['y1'] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                print(f"Mapped '{current_key}'")
                self.current_key_idx += 1
                
                if self.current_key_idx < len(self.keys_to_map):
                    print(f"Now draw rectangle for: {self.keys_to_map[self.current_key_idx]}")
                else:
                    print("All keys mapped! Press 'S' to save or continue drawing to adjust positions.")

    def run(self):
        window_name = "Keyboard Mapper"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"Draw rectangle for: {self.keys_to_map[0]}")
        
        while True:
            # Read camera frame
            ret, camera_frame = self.cap.read()
            if not ret:
                print("Failed to read camera frame")
                break
            
            # Overlay layout on camera frame
            display_image = camera_frame.copy()
            display_image = cv2.addWeighted(display_image, 0.8, self.layout_image, 0.2, 0)
            
            # Draw current rectangle if drawing
            if self.drawing and self.current_rect:
                cv2.rectangle(display_image, 
                              (self.current_rect[0], self.current_rect[1]),
                              (self.current_rect[2], self.current_rect[3]),
                              (255, 255, 255), 1)
            
            # Draw helper text
            cv2.putText(display_image, "Press 'S' to save, 'R' to reset, 'Q' to quit",
                        (10, self.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.current_key_idx < len(self.keys_to_map):
                cv2.putText(display_image,
                            f"Current key: {self.keys_to_map[self.current_key_idx]}",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_mapping()
            elif key == ord('r'):
                self.reset_mapping()
        
        self.cap.release()
        cv2.destroyAllWindows()

    def save_mapping(self):
        with open('keyboard_layout.json', 'w') as f:
            json.dump(self.key_map, f, indent=4)
        print("Keyboard mapping saved to 'keyboard_layout.json'")

    def reset_mapping(self):
        self.key_map.clear()
        self.current_key_idx = 0
        self.layout_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        print(f"Mapping reset. Draw rectangle for: {self.keys_to_map[0]}")

def main():
    mapper = KeyboardMapper(width=640, height=480)
    mapper.run()

if __name__ == "__main__":
    main()
