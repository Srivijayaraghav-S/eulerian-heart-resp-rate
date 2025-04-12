import cv2
import numpy as np
import dlib
from collections import deque
from threading import Thread
import time


PHONE1_URL = "http://192.168.1.48:8080/video"  
PHONE2_URL = 0 

# Stabilization
SMOOTHING_RADIUS = 5  # Frames for moving average
position_buffer = deque(maxlen=SMOOTHING_RADIUS)


# Video Output
OUTPUT_SIZE = (256, 256)  
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'avc1') 
RECORDING_DURATION = 20 


# Face Detection
FACE_DETECTION_THRESHOLD = 0.5  # Confidence threshold for DNN


# Tracking variables
last_camera = None
switch_count = 0



# FACE TRACKER CLASS
class FaceTracker:
    def __init__(self):
        # Initialize face detection model
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.tracker = None
        self.stabilization_buffer = deque(maxlen=SMOOTHING_RADIUS)
        


    def detect_face(self, frame):
        #Detect largest face in frame using DNN
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                    (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        best_face = None
        max_area = 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > FACE_DETECTION_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_face = (x1, y1, x2 - x1, y2 - y1)
        return best_face
    
    def stabilize_position(self, current_pos):
        """Apply smoothing to face position"""
        x, y, w, h = current_pos
        self.stabilization_buffer.append((x, y, w, h))
        if len(self.stabilization_buffer) > 0:
            return np.mean(self.stabilization_buffer, axis=0).astype(int)
        return current_pos
    
    def track_frame(self, frame):
        """Process a frame and return stabilized face ROI"""
        if self.tracker is None:
            # Initial face detection
            face_rect = self.detect_face(frame)
            if face_rect:
                x, y, w, h = face_rect
                self.tracker = dlib.correlation_tracker()
                self.tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))
                return frame[y:y+h, x:x+w]
            return None
        else:
            # Update tracker
            self.tracker.update(frame)
            pos = self.tracker.get_position()
            x, y, w, h = int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
            
            # Apply stabilization
            x_stab, y_stab, w_stab, h_stab = self.stabilize_position((x, y, w, h))
            
            # Return face ROI with 10% padding
            pad = int(w_stab * 0.1)
            y1 = max(0, y_stab - pad)
            y2 = min(frame.shape[0], y_stab + h_stab + pad)
            x1 = max(0, x_stab - pad)
            x2 = min(frame.shape[1], x_stab + w_stab + pad)
            
            return frame[y1:y2, x1:x2]

# CAMERA STREAM CLASS 
class CameraStream:
    def __init__(self, src, name):
        self.cap = cv2.VideoCapture(src)
        self.name = name
        self.frame = None
        self.tracker = FaceTracker()
        self.score = 0
        self.running = True
        
        # Set camera properties
        if isinstance(src, int):  # Only for local cameras
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"{self.name} disconnected. Reconnecting...")
                time.sleep(2)
                continue
                
            self.frame = frame
            face_roi = self.tracker.track_frame(frame)
            
            # Update score if face detected
            if face_roi is not None:
                h, w = face_roi.shape[:2]
                center_x = w/2
                center_y = h/2
                frame_center_x = self.frame.shape[1]/2
                frame_center_y = self.frame.shape[0]/2
                
                # Score based on size and centering
                size_score = w * h
                center_score = 1 - (abs(center_x - frame_center_x)/frame_center_x + 
                                   abs(center_y - frame_center_y)/frame_center_y)/2
                self.score = size_score * center_score
            else:
                self.score = 0
                self.tracker.tracker = None  # Reset tracker if face lost

    def get_face(self):
        #Returns stabilized face ROI and score
        if self.frame is not None and self.score > 0:
            face_roi = self.tracker.track_frame(self.frame)
            return face_roi, self.score, self.name
        return None, 0, self.name
        
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# MAIN FUNCTION
def main():
    global last_camera, switch_count
    
    print("Initializing cameras...")
    
    # Initialize cameras
    cameras = [
        CameraStream(PHONE1_URL, "Phone Camera"),
        CameraStream(PHONE2_URL, "Laptop Webcam")
    ]
    time.sleep(2)  # Allow cameras to initialize

    # Initialize video writer
    out = cv2.VideoWriter('face_tracking_output.mov', FOURCC, FPS, OUTPUT_SIZE)

    try:
        print(f"Starting face tracking (will record for {RECORDING_DURATION} seconds). Press 'q' to quit, 'r' to reset trackers.")
        start_time = time.time()
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Stop recording after duration
            if elapsed > RECORDING_DURATION:
                print(f"Recording completed after {RECORDING_DURATION} seconds")
                break
                
            best_face = None
            best_score = 0
            best_camera = None
            
            # Get faces from all cameras
            faces = []
            for cam in cameras:
                face, score, name = cam.get_face()
                faces.append((face, score, name))
                print(f"{name} Score: {score:.2f}")  # Print scores for each camera
                
                if face is not None and score > best_score:
                    best_face = face
                    best_score = score
                    best_camera = name
            
            # Check for camera switch
            if best_camera and best_camera != last_camera:
                switch_count += 1
                print(f"\nSWITCHING CAMERA: {last_camera} → {best_camera} (Score: {best_score:.2f})")
                print(f"Total switches: {switch_count}\n")
                last_camera = best_camera
            
            # Process best face
            if best_face is not None:
                # Resize and enhance contrast for rPPG
                resized_face = cv2.resize(best_face, OUTPUT_SIZE)
                
                # Apply CLAHE for better rPPG signal
                lab = cv2.cvtColor(resized_face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                processed_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                out.write(processed_face)
                
                # Display remaining recording time and current camera
                remaining = max(0, RECORDING_DURATION - elapsed)
                cv2.putText(processed_face, f"Recording: {remaining:.1f}s | Camera: {best_camera}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Best Stabilized Face", processed_face)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                for cam in cameras:
                    cam.tracker.tracker = None
                print("Trackers reset - reacquiring faces...")

    finally:
        for cam in cameras:
            cam.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Face tracking video saved as 'face_tracking_output.mov'")
        print(f"Total camera switches: {switch_count}")
        print(f"Final camera scores:")
        for cam in cameras:
            print(f"- {cam.name}: {cam.score:.2f}")

if __name__ == "__main__":
    main()