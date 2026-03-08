"""
Enhanced Real-time Cat vs Dog Detector with False Positive Reduction

IMPROVEMENTS:
- Stricter object detection in ROI (sharpness, contrast, edges, color diversity)
- Wider neutral zone between CAT (0.25) and DOG (0.75) thresholds
- Multiple stability checks - requires 5 consecutive frames to confirm
- Color distribution analysis to filter empty/plain backgrounds
- Contour analysis to ensure object (not just patterns)
- Debug info showing detection status and confirmation progress

USAGE:
- Place cat or dog inside the center box
- Keep pet steady for ~5 frames to get confirmation
- Press 'q' to quit

The detector will show:
  - GREEN: Cat detected with high confidence
  - RED: Dog detected with high confidence
  - YELLOW: Unclear object, needs closer view
  - GRAY: No pet or insufficient detail
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
from collections import deque

IMG_SIZE = 224
PREDICT_EVERY_N_FRAMES = 2

# Stricter prediction thresholds
CAT_THRESHOLD = 0.25      # More strict for cat detection
DOG_THRESHOLD = 0.75      # More strict for dog detection

# Stability settings
STABLE_FRAMES_REQUIRED = 5  # Require more frames for confirmation
EMPTY_RESET_FRAMES = 3      # More frames before resetting to neutral

def main():
    print("=" * 60)
    print("🐱🐶 CAT / DOG DETECTOR")
    print("📸 Ilagay lang ang pet sa box")
    print("=" * 60)

    model = find_model()
    if model is None:
        print("❌ Walang model! Press Enter to exit...")
        input()
        sys.exit(1)

    cap = open_camera()
    if cap is None:
        print("❌ Walang camera! Press Enter to exit...")
        input()
        sys.exit(1)

    detect_loop(cap, model)

def find_model():
    model_paths = [
        "cat_dog_classifier_85percent.keras",
        "best_model_85.keras",
        "cat_dog_classifier.h5",
        "../cat_dog_classifier_85percent.keras",
        "../best_model_85.keras",
        "../cat_dog_classifier.h5",
    ]

    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            return load_model(path)

    return None

def open_camera():
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (0, "Default"),
    ]

    for backend, name in backends:
        try:
            if backend == 0:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0, backend)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera backend: {name}")
                    return cap

                cap.release()
        except Exception:
            continue

    return None

def preprocess_roi(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(roi, model):
    try:
        img = preprocess_roi(roi)
        pred = float(model.predict(img, verbose=0)[0][0])
        return pred
    except Exception:
        return None

def roi_has_enough_detail(roi):
    """
    Enhanced check para sigurado may object sa box (cat/dog, hindi lang background).
    Checks: sharpness, contrast, edge density, at color distribution.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Sharpness check (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Contrast check
    std = gray.std()

    # Edge density check
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.count_nonzero(edges) / edges.size

    # Color spread analysis - checks if there are diverse colors (sign of object)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    hue_std = h.std()
    sat_std = s.std()
    val_std = v.std()
    
    color_diversity = (hue_std + sat_std + val_std) / 3

    # Symmetry/Pattern check - too much symmetry might mean just background/pattern
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_thin = cv2.Canny(gray_blurred, 50, 150)
    
    # Check if image has multiple details (not just a flat surface)
    contours, _ = cv2.findContours(edges_thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_multiple_features = len(contours) >= 3

    # Stricter thresholds
    enough_sharpness = lap_var > 100      # Increased from 45
    enough_contrast = std > 25             # Increased from 18
    enough_edges = edge_ratio > 0.03       # Increased from 0.015
    good_color_diversity = color_diversity > 20  # New check
    has_features = has_multiple_features   # New check

    return (enough_sharpness and enough_contrast and enough_edges and 
            good_color_diversity and has_features)

def get_state_from_pred(pred):
    """
    Convert prediction to state with confidence.
    Uses stricter thresholds to reduce false positives.
    """
    if pred is None:
        return "NEUTRAL", "No pet detected", (180, 180, 180)

    # Wider neutral zone (0.25 to 0.75) to prevent uncertain predictions
    if CAT_THRESHOLD < pred < DOG_THRESHOLD:
        return "NEUTRAL", "Unclear - move pet closer", (200, 200, 100)

    if pred <= CAT_THRESHOLD:
        # Higher confidence for cats
        confidence = ((0.25 - pred) / 0.25) * 100
        confidence = min(100, max(0, confidence))
        return "CAT", f"CAT {confidence:.0f}%", (0, 255, 0)

    if pred >= DOG_THRESHOLD:
        # Higher confidence for dogs
        confidence = ((pred - 0.75) / 0.25) * 100
        confidence = min(100, max(0, confidence))
        return "DOG", f"DOG {confidence:.0f}%", (255, 0, 0)
    
    return "NEUTRAL", "Unable to classify", (180, 180, 180)

def detect_loop(cap, model):
    frame_count = 0
    recent_states = deque(maxlen=STABLE_FRAMES_REQUIRED)
    recent_details = deque(maxlen=3)  # Track detail quality

    final_state = "NEUTRAL"
    final_text = "Place cat or dog inside the box"
    final_color = (180, 180, 180)

    empty_counter = 0
    debug_info = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        box_size = 224
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]

        frame_count += 1
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            # Step 1: Enhanced check if ROI has enough detail and an actual object
            has_detail = roi_has_enough_detail(roi)
            recent_details.append(has_detail)
            
            if not has_detail:
                empty_counter += 1
                recent_states.clear()
                debug_info = "No object in box"

                if empty_counter >= EMPTY_RESET_FRAMES:
                    final_state = "NEUTRAL"
                    final_text = "Place cat or dog inside box"
                    final_color = (180, 180, 180)
            else:
                empty_counter = 0
                debug_info = "Object detected"

                # Step 2: Predict only if sufficient detail exists
                pred = predict_image(roi, model)
                state, text, color = get_state_from_pred(pred)

                # Step 3: Require multiple stable frames before confirming
                recent_states.append((state, text, color))

                if len(recent_states) == STABLE_FRAMES_REQUIRED:
                    states_only = [s[0] for s in recent_states]

                    # All recent states must be the same (CAT, DOG, or NEUTRAL)
                    if all(s == states_only[0] for s in states_only):
                        final_state, final_text, final_color = recent_states[-1]
                        debug_info = f"Confirmed {final_state}"
                    else:
                        final_state = "NEUTRAL"
                        final_text = "Inconsistent detection - hold steady"
                        final_color = (200, 200, 100)
                        debug_info = "Unstable"

        # Draw box
        cv2.rectangle(display, (x1, y1), (x2, y2), final_color, 3)

        # Dark overlay header for readability
        cv2.rectangle(display, (x1, y1 - 35), (x2, y1), (0, 0, 0), -1)

        cv2.putText(
            display,
            final_text,
            (x1 + 8, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            final_color,
            2
        )

        cv2.putText(
            display,
            "Keep pet steady inside the box",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Wait for stable confirmation",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        # Show debug info
        cv2.putText(
            display,
            f"Status: {debug_info} | Confirmed: {len(recent_states)}/{STABLE_FRAMES_REQUIRED}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 255),
            1
        )

        cv2.putText(
            display,
            "Press q to quit",
            (w - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1
        )

        cv2.imshow("Cat / Dog Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector closed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")