# Emotion Detection Project

This project detects emotions from a webcam feed using a Deep Learning model.

## How to Run

1.  **Install Dependencies** (if not already installed):
    ```bash
    pip install tensorflow opencv-python flask matplotlib
    ```

2.  **Run the App**:
    ```bash
    python app.py
    ```
    Open your browser and go to `http://127.0.0.1:5000`.

## Improving Accuracy

The default model might have low accuracy. To improve it:

1.  **Train a new model**:
    Run the training script which uses a better architecture and data augmentation.
    ```bash
    python train_model.py
    ```
    This will save a new model file `emotion_detector_model_improved.h5`.

2.  **Restart the App**:
    The app will automatically detect the improved model and use it.

## Files

-   `app.py`: The main web application (Flask).
-   `train_model.py`: Script to train the model with better accuracy.
-   `emotion_dataset/`: Directory containing training and testing images.
-   `haarcascade_frontalface_default.xml`: Pre-trained face detection model.
