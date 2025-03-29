# Gesture-Based Image Processing

This project uses Mediapipe's hand tracking to apply various image filters based on hand gestures. It also allows switching between different filter units using swipe gestures.

## Features

-   **Real-time Hand Gesture Recognition:** Detects hand gestures like fist, index up, two fingers, three fingers, four fingers, and palm.
-   **Dynamic Filter Application:** Applies different image filters based on the recognized gesture.
-   **Filter Unit Switching:** Allows switching between three different sets of filters using left and right swipe gestures.
-   **Real-time Display:** Displays the original and filtered images side by side, along with information about the current unit, gesture, and filter applied.
-   **Image Saving:** Allows saving the filtered image by pressing 's'.
-   **Webcam Input:** Processes video from the webcam.

## Dependencies

-   Python 3.x
-   OpenCV (cv2)
-   Mediapipe
-   NumPy

Install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```

## How to Run

1.  Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2.  Run the Python script:

```bash
python main.py
```

3.  Use the following gestures to apply filters:

    -   **FIST:** Applies the first filter in the current unit.
    -   **INDEX_UP:** Applies the second filter in the current unit.
    -   **TWO_FINGERS:** Applies the third filter in the current unit.
    -   **THREE_FINGERS:** Applies the fourth filter in the current unit.
    -   **FOUR_FINGERS:** Applies the fifth filter in the current unit.
    -   **PALM:** Applies the sixth filter in the current unit.

4.  Use swipe gestures to switch between filter units:

    -   **Swipe Right:** Switches to the next filter unit (if available).
    -   **Swipe Left:** Switches to the previous filter unit (if available).

5.  Press 's' to save the filtered image.
6.  Press 'q' to quit the application.

## Filter Units

The application has three filter units, each with a different set of filters:

### Unit 1

-   **FIST:** Negative
-   **INDEX_UP:** Thresholding
-   **TWO_FINGERS:** Histogram Equalization
-   **THREE_FINGERS:** Gaussian Blur
-   **FOUR_FINGERS:** Darkening
-   **PALM:** Lightening

### Unit 2

-   **FIST:** Histogram Equalization
-   **INDEX_UP:** Smoothing Filter
-   **TWO_FINGERS:** Sharpening
-   **THREE_FINGERS:** Laplacian Edge Detection
-   **FOUR_FINGERS:** Sobel Edge Detection
-   **PALM:** Original Image

### Unit 3

-   **FIST:** Downsampling
-   **INDEX_UP:** Upsampling
-   **TWO_FINGERS:** Thresholding
-   **THREE_FINGERS:** JPEG Compression Simulation
-   **FOUR_FINGERS:** Haar Transform Simulation
-   **PALM:** Inverse Transform

## Example Usage

```python
# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# ... (rest of the code)

# Start webcam capture
cap = cv2.VideoCapture(0)
# ... (rest of the code)
```

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.

