import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")


# Read in and simultaneously preprocess video from a file
def read_video(path):
    cap = cv2.VideoCapture(path)
    return process_video(cap)


# Read in and preprocess video from webcam
def read_webcam(cap, duration=10):
    return process_video(cap, duration)


# Common function to process video frames
def process_video(cap, duration=None):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if duration:
        total_frames = duration * fps
    else:
        total_frames = float('inf')

    video_frames = []
    face_rects = ()

    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_frame = img

        # Detect face in the first frame
        if frame_count == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Select ROI (Region of Interest)
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1. / 255)
                video_frames.append(frame)

        frame_count += 1

    frame_ct = len(video_frames)
    cap.release()

    return video_frames, frame_ct, fps
