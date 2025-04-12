from argparse import ArgumentParser
import clip
import torch
from util import cvtransform
import cv2
import time
from packaging import version
import numpy as np
import collab_cam
import main

assert version.parse(torch.__version__) >= version.parse("1.7.1"), "pytorch version must be >= 1.7.1"

parser = ArgumentParser()
parser.add_argument("-d", "--device", dest="camera_device", type=int, default=0,
                    help="Camera device number (default: 0)", required=False)
args = parser.parse_args()

# Detection parameters
MIN_FALL_CONFIDENCE = 0.45  # Higher threshold for lying down
MIN_SIT_CONFIDENCE = 0.35   # Threshold for sitting
REQUIRED_FALL_DURATION = 2.0  # Seconds of sustained lying detection
FRAME_RATE = 30  # Estimated frames per second
DEBOUNCE_TIME = 1.0  # Seconds to wait after posture change

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Expanded target categories
targets = [
    "a person lying flat",  # More specific description
    "a person sitting",
    "a person standing upright",
    "no human in the frame",
    "a person bending over at the waist"
]

# Convert CLIP's preprocessing to OpenCV compatible format
preprocess = cvtransform(preprocess)

# Prepare the inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in targets]).to(device)

# State tracking
class PostureState:
    STANDING = 0
    SITTING = 1
    LYING = 2
    BENDING = 3
    UNKNOWN = 4

current_state = PostureState.UNKNOWN
state_start_time = time.time()
last_state_change = time.time()
is_fall_detected = False

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
    """Determine posture from inference results with confidence checks"""
    values, indices = zip(*results)
    max_idx = indices[0]
    max_conf = values[0].item()

    # Only return a posture if confidence is high enough
    if max_conf < 0.25:  # Very low confidence
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

# Encode text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Open camera
cap = cv2.VideoCapture(args.camera_device)
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

    # Convert to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = infer(image, text_features)
    new_posture = get_posture(results)

    # State machine logic
    if new_posture != current_state:
        # Only change state if new posture is sustained for debounce period
        if (current_time - last_state_change) > DEBOUNCE_TIME:
            current_state = new_posture
            state_start_time = current_time
            last_state_change = current_time
    else:
        # Check for fall condition (lying for required duration)
        if (current_state == PostureState.LYING and
            (current_time - state_start_time) >= REQUIRED_FALL_DURATION):
            is_fall_detected = True

    # Reset fall detection if posture changes
    if current_state != PostureState.LYING:
        is_fall_detected = False

    # Display predictions and state information
    info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)

    # Show all predictions
    for i, (value, index) in enumerate(results):
        text = f"{targets[index][:25]:<25} {100 * value.item():5.1f}%"
        color = (255, 255, 255)
        if index == 0 and value.item() >= MIN_FALL_CONFIDENCE:
            color = (0, 0, 255)  # Red for lying
        elif index == 1 and value.item() >= MIN_SIT_CONFIDENCE:
            color = (0, 255, 255)  # Yellow for sitting
        cv2.putText(info_panel, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Show current state
    state_names = ["STANDING", "SITTING", "LYING", "BENDING", "UNKNOWN"]
    state_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 165, 0), (128, 128, 128)]
    cv2.putText(info_panel, f"State: {state_names[current_state]}",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                state_colors[current_state], 2)

    # Show duration in current state
    duration = current_time - state_start_time
    cv2.putText(info_panel, f"Duration: {duration:.1f}s",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1)

    # Combine with main frame
    display_frame = np.vstack([frame, info_panel])

    # Display fall warning if detected
    if is_fall_detected:
        draw_text_box(display_frame, "FALL DETECTED!",
                     pos=(frame.shape[1]//2 - 200, frame.shape[0] - 50),
                     font_scale=1.3, text_color=(255, 255, 255),
                     text_color_bg=(0, 0, 255))
        collab_cam.main()
        main.main()
        break
    
    cv2.imshow("Enhanced Fall Detection", display_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()