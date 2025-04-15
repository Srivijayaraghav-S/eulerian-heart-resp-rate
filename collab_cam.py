import cv2
import numpy as np
import dlib
from collections import deque
from threading import Thread
import time
import os
import eulerian_main
from shared_state import recording_complete_flag

PHONE1_URL = "http://192.168.116.119:8080/video"  
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

def save_plots(fig, ax1, ax3):
    fig.savefig('plots/facial_score.png', 
               dpi=300, bbox_inches='tight')
    
def save_plots_flask(fig, ax1, ax3):
    fig.savefig('static/plots/facial_score.png', 
               dpi=300, bbox_inches='tight')


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

# # MAIN FUNCTION
# def main():
#     global last_camera, switch_count
    
#     print("Initializing cameras...")
    
#     # Initialize cameras
#     cameras = [
#         CameraStream(PHONE1_URL, "Phone Camera"),
#         CameraStream(PHONE2_URL, "Laptop Webcam")
#     ]
#     time.sleep(2)  # Allow cameras to initialize

#     # Initialize video writer
#     out = cv2.VideoWriter('face_tracking_output.mov', FOURCC, FPS, OUTPUT_SIZE)

#     try:
#         print(f"Starting face tracking (will record for {RECORDING_DURATION} seconds). Press 'q' to quit, 'r' to reset trackers.")
#         start_time = time.time()
        
#         while True:
#             current_time = time.time()
#             elapsed = current_time - start_time
            
#             # Stop recording after duration
#             if elapsed > RECORDING_DURATION:
#                 print(f"Recording completed after {RECORDING_DURATION} seconds")
#                 break
                
#             best_face = None
#             best_score = 0
#             best_camera = None
            
#             # Get faces from all cameras
#             faces = []
#             for cam in cameras:
#                 face, score, name = cam.get_face()
#                 faces.append((face, score, name))
#                 print(f"{name} Score: {score:.2f}")  # Print scores for each camera
                
#                 if face is not None and score > best_score:
#                     best_face = face
#                     best_score = score
#                     best_camera = name
            
#             # Check for camera switch
#             if best_camera and best_camera != last_camera:
#                 switch_count += 1
#                 print(f"\nSWITCHING CAMERA: {last_camera} → {best_camera} (Score: {best_score:.2f})")
#                 print(f"Total switches: {switch_count}\n")
#                 last_camera = best_camera
            
#             # Process best face
#             if best_face is not None:
#                 # Resize and enhance contrast for rPPG
#                 resized_face = cv2.resize(best_face, OUTPUT_SIZE)
                
#                 # Apply CLAHE for better rPPG signal
#                 lab = cv2.cvtColor(resized_face, cv2.COLOR_BGR2LAB)
#                 l, a, b = cv2.split(lab)
#                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#                 l = clahe.apply(l)
#                 lab = cv2.merge((l,a,b))
#                 processed_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
#                 out.write(processed_face)
                
#                 # Display remaining recording time and current camera
#                 remaining = max(0, RECORDING_DURATION - elapsed)
#                 cv2.putText(processed_face, f"Recording: {remaining:.1f}s | Camera: {best_camera}", 
#                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 cv2.imshow("Best Stabilized Face", processed_face)
            
#             # Handle key presses
#             key = cv2.waitKey(1)
#             if key == ord('q'):
#                 break
#             elif key == ord('r'):
#                 for cam in cameras:
#                     cam.tracker.tracker = None
#                 print("Trackers reset - reacquiring faces...")

#     finally:
#         for cam in cameras:
#             cam.stop()
#         out.release()
#         cv2.destroyAllWindows()
#         print(f"Face tracking video saved as 'face_tracking_output.mov'")
#         print(f"Total camera switches: {switch_count}")
#         print(f"Final camera scores:")
#         for cam in cameras:
#             print(f"- {cam.name}: {cam.score:.2f}")

def collab_cam_stream():
    """
    A combined streaming generator that tracks the face,
    records the processed frames, and saves the video to a file.
    It yields MJPEG frames for display on a website and stops automatically
    after RECORDING_DURATION seconds.
    """
    global last_camera, switch_count
    import matplotlib.pyplot as plt

    print("Initializing cameras...")
    # Set up cameras from both sources.
    cameras = [
         CameraStream(PHONE1_URL, "Phone Camera"),
         CameraStream(PHONE2_URL, "Laptop Webcam")
    ]
    time.sleep(2)  # Allow cameras to initialize

    # Ensure the output directory exists.
    output_dir = "videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define the output file name.
    output_file = "face_tracking_output_new.mov"
    # Create a VideoWriter to record the processed frames.
    out = cv2.VideoWriter(output_file, FOURCC, FPS, OUTPUT_SIZE)
    # Real-time score tracking
    score_log = {
        "time": [],
        "phone": [],
        "webcam": [],
        "active_cam": [],
        "switches": []
    }
    elapsed_prev = -1

    # Setup real-time plot
    # plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    phone_line, = ax.plot([], [], label="Phone Camera", color='blue')
    webcam_line, = ax.plot([], [], label="Laptop Webcam", color='orange')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Facial Score")
    ax.set_title("Real-Time Camera Scores with Switch Markers")
    ax.legend()
    ax.grid(True)
    
    start_time = time.time()  # Record the start time

    try:
        while True:
            # Automatically stop after RECORDING_DURATION seconds.
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > RECORDING_DURATION:
                print("Face tracking recording time expired.")
                recording_complete_flag["value"] = True
                break

            best_face = None
            best_score = 0
            best_camera = None
            # Retrieve face ROIs and scores from all cameras.
            for cam in cameras:
                face, score, name = cam.get_face()
                print(f"{name} Score: {score:.2f}")
                if face is not None and score > best_score:
                    best_face = face
                    best_score = score
                    best_camera = name
            
            # Log every full second
            if int(elapsed) != int(elapsed_prev):
                score_log["time"].append(int(elapsed))
                score_log["phone"].append(cameras[0].score)
                score_log["webcam"].append(cameras[1].score)
                score_log["active_cam"].append(best_camera)

                if best_camera != last_camera and last_camera is not None:
                    switch_count += 1
                    score_log["switches"].append(int(elapsed))
                    print(f"\nSWITCHING CAMERA: {last_camera} → {best_camera} (Score: {best_score:.2f})")
                    print(f"Total switches: {switch_count}\n")

                last_camera = best_camera
                elapsed_prev = int(elapsed)

                # Update real-time plot
                phone_line.set_data(score_log["time"], score_log["phone"])
                webcam_line.set_data(score_log["time"], score_log["webcam"])
                ax.relim()
                ax.autoscale_view()

                # Clear old switch lines
                [l.remove() for l in ax.lines[2:]]

                # Re-draw switch lines
                for switch_time in score_log["switches"]:
                    ax.axvline(x=switch_time, color='red', linestyle='--', alpha=0.5)

                # plt.pause(0.01)

            if best_face is not None:
            #     # Resize the processed face to the OUTPUT_SIZE.
            #     processed_face = cv2.resize(best_face, OUTPUT_SIZE)
            #     # Draw the active camera identifier at the bottom left.
            #     # cv2.putText(processed_face, f"Camera: {best_camera}",
            #     #             (10, OUTPUT_SIZE[1] - 10),
            #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            # else:
            #     # When no face is detected, create a blank frame.
            #     processed_face = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)
            
                resized_face = cv2.resize(best_face, OUTPUT_SIZE)
                lab = cv2.cvtColor(resized_face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                processed_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
                # Write the processed frame to the video file.
                out.write(processed_face)
                remaining = max(0, RECORDING_DURATION - time.time() + start_time)
                # cv2.putText(processed_face, f"Recording: {remaining:.1f}s | Camera: {best_camera}",
                #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_face, f"Recording: {remaining:.1f}s | Camera: {best_camera}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                # Encode the processed frame as JPEG for streaming.
            ret, buffer = cv2.imencode('.jpg', processed_face)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                continue
    finally:
        for cam in cameras:
            cam.stop()
        out.release()
        save_plots_flask(fig, ax, None)
        print(f"Face tracking video saved at '{output_file}'")
        print(f"Total camera switches: {switch_count}")
        print(f"Final camera scores:")
        for cam in cameras:
            print(f"- {cam.name}: {cam.score:.2f}")
        eulerian_main.eulerian_main_stream()

def main():
    global last_camera, switch_count
    import matplotlib.pyplot as plt

    print("Initializing cameras...")

    cameras = [
        CameraStream(PHONE1_URL, "Phone Camera"),
        CameraStream(PHONE2_URL, "Laptop Webcam")
    ]
    time.sleep(2)

    out = cv2.VideoWriter('videos/face_tracking_output_new.mov', FOURCC, FPS, OUTPUT_SIZE)

    # Real-time score tracking
    score_log = {
        "time": [],
        "phone": [],
        "webcam": [],
        "active_cam": [],
        "switches": []
    }
    elapsed_prev = -1

    # Setup real-time plot
    # plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    phone_line, = ax.plot([], [], label="Phone Camera", color='blue')
    webcam_line, = ax.plot([], [], label="Laptop Webcam", color='orange')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Facial Score")
    ax.set_title("Real-Time Camera Scores with Switch Markers")
    ax.legend()
    ax.grid(True)

    print(f"Starting face tracking (will record for {RECORDING_DURATION} seconds). Press 'q' to quit, 'r' to reset trackers.")
    start_time = time.time()

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed > RECORDING_DURATION:
                print(f"Recording completed after {RECORDING_DURATION} seconds")
                break

            best_face = None
            best_score = 0
            best_camera = None

            faces = []
            for cam in cameras:
                face, score, name = cam.get_face()
                faces.append((face, score, name))
                print(f"{name} Score: {score:.2f}")
                if face is not None and score > best_score:
                    best_face = face
                    best_score = score
                    best_camera = name

            # Log every full second
            if int(elapsed) != int(elapsed_prev):
                score_log["time"].append(int(elapsed))
                score_log["phone"].append(cameras[0].score)
                score_log["webcam"].append(cameras[1].score)
                score_log["active_cam"].append(best_camera)

                if best_camera != last_camera and last_camera is not None:
                    switch_count += 1
                    score_log["switches"].append(int(elapsed))
                    print(f"\nSWITCHING CAMERA: {last_camera} → {best_camera} (Score: {best_score:.2f})")
                    print(f"Total switches: {switch_count}\n")

                last_camera = best_camera
                elapsed_prev = int(elapsed)

                # Update real-time plot
                phone_line.set_data(score_log["time"], score_log["phone"])
                webcam_line.set_data(score_log["time"], score_log["webcam"])
                ax.relim()
                ax.autoscale_view()

                # Clear old switch lines
                [l.remove() for l in ax.lines[2:]]

                # Re-draw switch lines
                for switch_time in score_log["switches"]:
                    ax.axvline(x=switch_time, color='red', linestyle='--', alpha=0.5)

                # plt.pause(0.01)

            if best_face is not None:
                resized_face = cv2.resize(best_face, OUTPUT_SIZE)
                lab = cv2.cvtColor(resized_face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                processed_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                out.write(processed_face)

                remaining = max(0, RECORDING_DURATION - elapsed)
                cv2.putText(processed_face, f"Recording: {remaining:.1f}s | Camera: {best_camera}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Best Stabilized Face", processed_face)

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
        save_plots(fig, ax, None)
        # plt.ioff()
        # plt.show()
        print(f"Face tracking video saved at 'videos/trial.mov'")
        print(f"Total camera switches: {switch_count}")
        print(f"Final camera scores:")
        for cam in cameras:
            print(f"- {cam.name}: {cam.score:.2f}")
            
def test_camera_connection(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret
    return False

# if not test_camera_connection(PHONE1_URL):
#     print(f"ERROR: Cannot connect to IP camera at {PHONE1_URL}")
#     print("Please verify:")
#     print("1. Your phone is on the same WiFi network")
#     print("2. The IP address is correct")
#     print("3. The camera app is running")
#     print("4. No firewall is blocking the connection")
#     exit(1)

if __name__ == "__main__":
    main()