from argparse import ArgumentParser
import clip
import torch
from util import cvtransform
import cv2
import time
from packaging import version
import numpy as np
import collab_cam
import eulerian_main

assert version.parse(torch.__version__) >= version.parse("1.7.1"), "pytorch version must be >= 1.7.1"

# Detection parameters
MIN_FALL_CONFIDENCE = 0.45
MIN_SIT_CONFIDENCE = 0.35
REQUIRED_FALL_DURATION = 2.0
FRAME_RATE = 30
DEBOUNCE_TIME = 1.0

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
targets = [
    "a person lying flat",
    "a person sitting",
    "a person standing upright",
    "no human in the frame",
    "a person bending over at the waist"
]
preprocess = cvtransform(preprocess)
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in targets]).to(device)

class PostureState:
    STANDING = 0
    SITTING = 1
    LYING = 2
    BENDING = 3
    UNKNOWN = 4

def draw_text_box(img, text, pos=(0, 0), font_scale=1.5, font_thickness=2,
                 text_color=(255, 255, 255), text_color_bg=(0, 0, 0), margin=2):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x - margin, y - margin), (x + text_w + margin, y + text_h + margin),
                  text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    return text_size

def get_posture(results):
    values, indices = zip(*results)
    max_idx = indices[0]
    max_conf = values[0].item()
    if max_conf < 0.25:
        return PostureState.UNKNOWN
    if max_idx == 0 and max_conf >= MIN_FALL_CONFIDENCE:
        return PostureState.LYING
    elif max_idx == 1 and max_conf >= MIN_SIT_CONFIDENCE:
        return PostureState.SITTING
    elif max_idx == 2:
        return PostureState.STANDING
    elif max_idx == 4:
        return PostureState.BENDING
    return PostureState.UNKNOWN

def infer(src, text_features):
    image_input = preprocess(src).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(targets))
        results = list(zip(values, indices))
        return results
    
def draw_fall_alert(frame):
    """Draw a full-screen fall detection alert"""
    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Calculate text size and position
    text = "FALL DETECTED!"
    font_scale = 5.0
    thickness = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text dimensions
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate center position
    x = (frame.shape[1] - text_width) // 2
    y = (frame.shape[0] + text_height) // 2
    
    # Draw outline text (for better visibility)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness*2, cv2.LINE_AA)
    # Draw main text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

# Shared text_features for infer() call
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    
def fall_detection_stream(camera_device=0, set_fall_detected_callback=None):
    current_state = PostureState.UNKNOWN
    state_start_time = time.time()
    last_state_change = time.time()
    is_fall_detected = False

    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {camera_device}")
        return  # or exit()

    print("Running fall detection stream...")
    last_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        frame_count += 1
        current_time = time.time()
        # Compute frame rate (if needed for debugging)
        _ = 1.0 / (current_time - last_time)
        last_time = current_time

        # Convert BGR frame to RGB for processing with CLIP.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = infer(image, text_features)
        new_posture = get_posture(results)

        # Update posture state based on debounce interval.
        if new_posture != current_state:
            if (current_time - last_state_change) > DEBOUNCE_TIME:
                current_state = new_posture
                state_start_time = current_time
                last_state_change = current_time
        else:
            if current_state == PostureState.LYING and (current_time - state_start_time) >= REQUIRED_FALL_DURATION:
                is_fall_detected = True

        if current_state != PostureState.LYING:
            is_fall_detected = False

        # Build the info panel (a fixed-height image appended below the frame).
        info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        for i, (value, index) in enumerate(results):
            text_info = f"{targets[index][:25]:<25} {100 * value.item():5.1f}%"
            color = (255, 255, 255)
            if index == 0 and value.item() >= MIN_FALL_CONFIDENCE:
                color = (0, 0, 255)
            elif index == 1 and value.item() >= MIN_SIT_CONFIDENCE:
                color = (0, 255, 255)
            cv2.putText(info_panel, text_info, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        state_names = ["STANDING", "SITTING", "LYING DOWN", "BENDING", "UNKNOWN"]
        state_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 165, 0), (128, 128, 128)]
        status_text = f"State: {state_names[current_state]}"
        (text_width, text_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x_pos = info_panel.shape[1] - text_width - 20  # Right padding
        y_pos = 40  # Top padding
        cv2.putText(info_panel, status_text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colors[current_state], 2)

        duration = current_time - state_start_time
        cv2.putText(info_panel, f"Duration: {duration:.1f}s",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # If a fall is detected, optionally modify the frame. For example, you could overlay a large alert.
        if is_fall_detected:
            frame = draw_fall_alert(frame)
            # (Optional: If you want the stream to stop on fall detection, use "break" here.)
            # break

        # Combine the original (or alert-modified) frame and the info panel vertically.
        display_frame = np.vstack([frame, info_panel])

        # Encode the combined frame as JPEG.
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue  # Skip if encoding failed.
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if is_fall_detected:
            print("Fall detected. Ending fall detection stream.")
            if set_fall_detected_callback:
                set_fall_detected_callback()
            break

    cap.release()


# MAIN block starts here
def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="camera_device", type=int, default=0,
                        help="Camera device number (default: 0)", required=False)
    args = parser.parse_args()

    current_state = PostureState.UNKNOWN
    state_start_time = time.time()
    last_state_change = time.time()
    is_fall_detected = False

    cap = cv2.VideoCapture(args.camera_device)
    # cap = cv2.VideoCapture("videos/fall_vid_2.MOV")
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera_device}")
        exit()

    print("Running... Press 'q' to quit...")
    last_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        frame_count += 1
        current_time = time.time()
        frame_rate = 1.0 / (current_time - last_time)
        last_time = current_time

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = infer(image, text_features)
        new_posture = get_posture(results)

        if new_posture != current_state:
            if (current_time - last_state_change) > DEBOUNCE_TIME:
                current_state = new_posture
                state_start_time = current_time
                last_state_change = current_time
        else:
            if (current_state == PostureState.LYING and
                (current_time - state_start_time) >= REQUIRED_FALL_DURATION):
                is_fall_detected = True

        if current_state != PostureState.LYING:
            is_fall_detected = False

        info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        for i, (value, index) in enumerate(results):
            text = f"{targets[index][:25]:<25} {100 * value.item():5.1f}%"
            color = (255, 255, 255)
            if index == 0 and value.item() >= MIN_FALL_CONFIDENCE:
                color = (0, 0, 255)
            elif index == 1 and value.item() >= MIN_SIT_CONFIDENCE:
                color = (0, 255, 255)
            cv2.putText(info_panel, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        state_names = ["STANDING", "SITTING", "LYING DOWN", "BENDING", "UNKNOWN"]
        state_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 165, 0), (128, 128, 128)]
        status_text = f"State: {state_names[current_state]}"
        (text_width, text_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x_pos = info_panel.shape[1] - text_width - 20  # Padding from right
        y_pos = 40  # Top padding
        # cv2.putText(info_panel, f"State: {state_names[current_state]}",
        #             (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #             state_colors[current_state], 2)
        cv2.putText(info_panel, status_text,
            (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            state_colors[current_state], 2)

        duration = current_time - state_start_time
        cv2.putText(info_panel, f"Duration: {duration:.1f}s",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)

        display_frame = np.vstack([frame, info_panel])

        if is_fall_detected:
            frame = draw_fall_alert(frame)
            cv2.imshow("Fall Detection", frame)
            cv2.waitKey(500)
            print("Fall detected. Switching to face tracking.")
            break

        cv2.imshow("Enhanced Fall Detection", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    collab_cam.main()
    # eulerian_main.main()

if __name__ == "__main__":
    main()