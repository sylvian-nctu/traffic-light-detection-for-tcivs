from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pyttsx3
import threading
from PIL import Image, ImageDraw, ImageFont

class TrafficLightDetector:
    def __init__(self):
        # Initialize voice engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize camera
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (1280, 720)}
        )
        self.picam2.configure(preview_config)
        self.picam2.start()
        time.sleep(2)
        
        # Load YOLO model
        print("Loading model...")
        self.model = YOLO('best.pt')
        print("Model loaded successfully!")
        
        # Initialize states
        self.green_light_start_time = None
        self.last_alert_time = 0
        self.last_state = None
        
        # Create fullscreen window
        cv2.namedWindow('Traffic Light Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Traffic Light Detection', cv2.WND_PROP_FULLSCREEN, 
                            cv2.WINDOW_FULLSCREEN)
        
        # Load font
        self.font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 32)
        
    def speak_alert(self, text):
        def speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        current_time = time.time()
        if current_time - self.last_alert_time >= 5:
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            self.last_alert_time = current_time

    def add_status_text(self, image, text):
        # Convert OpenCV image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Set text position (bottom left)
        text_x = 20
        text_y = height - 50
        
        # Draw semi-transparent background
        padding = 10
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw text
        draw.text((text_x, text_y - text_height), text, font=self.font, fill=(255, 255, 255))
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    def run(self):
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Run detection
                results = self.model.predict(
                    source=frame,
                    conf=0.25,
                    verbose=False
                )
                
                red_light_detected = False
                green_light_detected = False
                
                # Process detection results
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        
                        if cls == 4:  # Red light
                            red_light_detected = True
                        elif cls == 3:  # Green light
                            green_light_detected = True
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 255, 0)  # Default green
                        if cls == 4:  # Red light
                            color = (0, 0, 255)
                        elif cls == 5:  # Yellow light
                            color = (0, 255, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                current_time = time.time()
                
                # Update status and alerts
                if red_light_detected:
                    if self.last_state != 'red':
                        self.speak_alert("Red light, please stop")
                        self.last_state = 'red'
                    status_text = "RED LIGHT - STOP!"
                elif green_light_detected:
                    if self.last_state != 'green':
                        self.green_light_start_time = current_time
                        self.last_state = 'green'
                    
                    if current_time - self.green_light_start_time >= 5:
                        self.speak_alert("Green light, please go")
                        status_text = "GREEN LIGHT - GO!"
                    else:
                        status_text = f"GREEN LIGHT (Waiting... {5 - int(current_time - self.green_light_start_time)}s)"
                else:
                    self.last_state = None
                    self.green_light_start_time = None
                    status_text = "Waiting for detection..."
                
                # Add status text
                frame = self.add_status_text(frame, status_text)
                
                # Display frame
                cv2.imshow('Traffic Light Detection', frame)
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                
        except Exception as e:
            print(f"Error occurred: {e}")
            
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()

def main():
    detector = TrafficLightDetector()
    detector.run()

if __name__ == '__main__':
    main()