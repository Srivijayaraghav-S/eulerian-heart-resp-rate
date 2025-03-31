import cv2
import numpy as np

# Load DNN model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    face_rects = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Ensure coordinates are within frame bounds
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face_rects.append((startX, startY, endX-startX, endY-startY))
    
    return face_rects

def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    return process_video(cap)

def process_video(cap, duration=None):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if duration:
        total_frames = duration * fps
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_frames = []
    face_rects = ()
    frame_count = 0
    
    while cap.isOpened() and frame_count < total_frames:
        ret, img = cap.read()
        if not ret:
            break
        
        # Standardize frame size
        img = cv2.resize(img, (256, 256))  # Force consistent size

        # Detect face using DNN (only once if stationary subject)
        if frame_count == 0 or frame_count % fps == 0:
            face_rects = detect_faces_dnn(img)
        
        if len(face_rects) > 0:
            (x, y, w, h) = face_rects[0]  # Use first detected face
            roi_frame = img[y:y+h, x:x+w]
            if roi_frame.size > 0:  # Ensure valid ROI
                roi_frame = cv2.resize(roi_frame, (256, 256))  # Standard size
                frame = roi_frame.astype("float32") / 255.0
                video_frames.append(frame)

        frame_count += 1

    cap.release()
    return video_frames, len(video_frames), fps