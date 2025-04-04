import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Track Current Unit (1, 2, 3)
current_unit = 1
prev_wrist_x = None

def detect_gesture(hand_landmarks):
    """Detect hand gesture based on finger positions."""
    if not hand_landmarks:
        return None, ""
    lm = hand_landmarks.landmark
    fingers = []
    # Thumb (X-coordinates comparison)
    fingers.append(1 if lm[4].x < lm[3].x else 0)
    # Other 4 fingers (Y-coordinates comparison)
    tips_ids = [8, 12, 16, 20]
    for tip in tips_ids:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    
    # Gesture Dictionary
    gesture_dict = {
        (0, 0, 0, 0, 0): "FIST",
        (0, 1, 0, 0, 0): "INDEX_UP",
        (0, 1, 1, 0, 0): "TWO_FINGERS",
        (0, 1, 1, 1, 0): "THREE_FINGERS",
        (0, 1, 1, 1, 1): "FOUR_FINGERS",
        (1, 1, 1, 1, 1): "PALM"
    }
    return gesture_dict.get(tuple(fingers), None)

def apply_filter(img, gesture, unit):
    """Apply corresponding image filter based on gesture and unit."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filters = {
    1: {
        "FIST": (cv2.bitwise_not(gray), "Negative"),
        "INDEX_UP": (cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1], "Thresholding"),
        "TWO_FINGERS": (cv2.equalizeHist(gray), "Histogram Equalization"),
        "THREE_FINGERS": (cv2.GaussianBlur(gray, (15, 15), 0), "Gaussian Blur"),
        "FOUR_FINGERS": (gray // 2, "Darkening"),
        "PALM": (gray * 2, "Lightening")
    },
    2: {
        "FIST": (cv2.equalizeHist(gray), "Histogram Equalization"),
        "INDEX_UP": (cv2.GaussianBlur(gray, (15, 15), 0), "Smoothing Filter"),
        "TWO_FINGERS": (cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])), "Sharpening"),
        "THREE_FINGERS": (cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)), "Laplacian Edge Detection"),
        "FOUR_FINGERS": (cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)), "Sobel Edge Detection"),
        "PALM": (gray, "Original Image")
    },
    3: {
        "FIST": (cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2)), "Downsampling"),
        "INDEX_UP": (cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2)), "Upsampling"),
        "TWO_FINGERS": (cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1], "Thresholding"),
        "THREE_FINGERS": (cv2.GaussianBlur(gray, (15, 15), 0), "JPEG Compression Simulation"),
        "FOUR_FINGERS": (gray // 2, "Haar Transform Simulation"),
        "PALM": (gray * 2, "Inverse Transform")
    }
}

    filtered_img, filter_name = filters[unit].get(gesture, (gray, "None"))

    # Resize filtered image to match frame size
    filtered_img = cv2.resize(filtered_img, (frame.shape[1], frame.shape[0]))

    return filtered_img, filter_name


def detect_swipe(lm):
    """Detect swipe left or right based on wrist movement."""
    global current_unit, prev_wrist_x
    wrist_x = lm[0].x
    if prev_wrist_x is not None:
        if wrist_x - prev_wrist_x > 0.15 and current_unit < 3:
            current_unit += 1  # Swipe Right
        elif prev_wrist_x - wrist_x > 0.15 and current_unit > 1:
            current_unit -= 1  # Swipe Left
    prev_wrist_x = wrist_x

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    gesture, filter_name = None, "None"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            detect_swipe(hand_landmarks.landmark)
    
    filtered, filter_name = apply_filter(frame, gesture, current_unit)
    stacked = np.hstack((cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), frame))
    
    # Display Unit, Gesture, and Filter Info
    cv2.putText(stacked, f"Unit: {current_unit}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(stacked, f"Gesture: {gesture if gesture else 'None'}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(stacked, f"Filter: {filter_name}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Gesture-Based Image Processing", stacked)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        if gesture and filter_name:
            # Check if the 'images' folder exists, and create it if not.
            if not os.path.exists("images"):
                os.makedirs("images")  # Create the directory

            filename = f"images/{gesture}_{filter_name}.jpg"
            cv2.imwrite(filename, filtered)
            print(f"Image saved as: {filename}")
        else:
            print("Gesture or Filter not detected, image not saved.")
    elif key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
