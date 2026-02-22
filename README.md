# 🎭 Real-Time Emotion Detection Web App

A powerful and interactive Real-Time Emotion Detection application built using **Flask**, **OpenCV**, and **Deep Learning (TensorFlow/Keras)**. This application accesses the user's webcam, detects faces, and predicts their current emotion in real time using a custom-trained Convolutional Neural Network (CNN).

## ✨ Features
- **Live Video Feed:** Processes webcam video directly in the browser.
- **Real-Time Emotion Recognition:** Detects 7 basic emotions: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*.
- **Visual Feedback:** Draws bounding boxes around detected faces, color-coded based on the detected emotion.
- **RESTful API Endpoint:** Exposes real-time emotion probabilities via a structured JSON API (`/emotion_data`).
- **Custom Trained CNN:** Built from scratch using Keras and trained on facial expression datasets.

## 📸 Screenshots

Here is a glimpse of the application in action:

![Main Dashboard](Images/Main%20.png)

### Emotion Detections

| Detections | Detections |
| :---: | :---: |
| ![Emotion 1](Images/1.jpg) | ![Emotion 2](Images/2.jpg) |
| ![Emotion 3](Images/3.jpg) | ![Emotion 4](Images/4.jpg) |

![Emotion 5](Images/5.jpg)

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Computer Vision:** OpenCV (`haarcascade_frontalface_default.xml` for face detection)
- **Deep Learning:** TensorFlow, Keras
- **Frontend:** HTML, CSS, JavaScript (via `templates/index.html` and `static/style.css`)

## 🚀 How to Run Locally

### 1. Ensure you are in the project folder

### 2. Install Required Dependencies:
Install all necessary Python packages:
```bash
pip install tensorflow opencv-python flask numpy scikit-learn matplotlib
```

### 3. Run the Application:
Start the Flask development server:
```bash
python app.py
```
Open your web browser and go to: `http://127.0.0.1:5000`

## 🧠 Model Training & Improving Accuracy

The project includes a robust training script (`train_model.py`) to build the emotion classifier from scratch using data augmentation, batch normalization, and dropout techniques to prevent overfitting.

To train a new or improved model:
1. Ensure your dataset is correctly placed in `emotion_dataset/train` and `emotion_dataset/test`.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. This process will dynamically save the best model weights based on validation accuracy.
4. Start the application, it will fallback automatically to the available models (`final_emotion_model.h5`, `emotion_detector_model_improved.h5`, or `emotion_detector_model.h5`).

## 📂 Project Structure

- `app.py`: The main Flask web server, rendering the UI and handling video streams.
- `train_model.py`: Script defining the CNN architecture and handling the training process.
- `Images/`: Directory containing project screenshots and sample outputs.
- `emotion_dataset/`: Directory containing the image datasets used for training/testing.
- `templates/` & `static/`: HTML templates and stylesheet assets for the frontend UI.
- `haarcascade_frontalface_default.xml`: Haar Cascade required for detecting faces via OpenCV.
- `*.h5`: The saved and compiled Keras Deep Learning models.
