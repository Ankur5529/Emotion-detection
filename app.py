from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import threading

app = Flask(__name__)

# Load Model
MODEL_PATH = "final_emotion_model.h5"  
# Fallback logic

if not os.path.exists(MODEL_PATH):
    if os.path.exists("emotion_detector_model_improved.h5"):
        MODEL_PATH = "emotion_detector_model_improved.h5"
    else:
        MODEL_PATH = "emotion_detector_model.h5"

print(f"Loading Model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
except:
    print("Warning: Could not load the preferred model. Make sure training is complete.")
    model = None

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Face Detection Haarcascade
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
if not os.path.exists(FACE_CASCADE_PATH):
    print(f"Error: Haarcascade file not found at {FACE_CASCADE_PATH}")

face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)

data_lock = threading.Lock()
current_emotion_state = {
    'emotion': 'Neutral',
    'probabilities': {label: 0.0 for label in class_labels}
}

# Video Processing
def generate_frames():
    cap = cv2.VideoCapture(0) 
    global current_emotion_state
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                pass
            else:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = np.expand_dims(roi, axis=0)
                        roi = np.expand_dims(roi, axis=-1)

                        label = 'Neutral'
                        if model is not None:
                            prediction = model.predict(roi, verbose=0)[0]
                            label_argmax = prediction.argmax()
                            label = class_labels[label_argmax]
                            
                            # Update global state
                            with data_lock:
                                current_emotion_state = {
                                    'emotion': label,
                                    'probabilities': {class_labels[i]: float(prediction[i]) for i in range(len(class_labels))}
                                }
                        
                        # Draw visual feedback on frame
                        color = (0, 255, 0) # Green for box
                        if label == 'Angry': color = (0, 0, 255)
                        elif label == 'Happy': color = (0, 255, 255) # Yellow
                        elif label == 'Sad': color = (255, 0, 0) # Blue
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release() 

#Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    with data_lock:
        return jsonify(current_emotion_state)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
