from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
import json
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Global model variable
model = None

# Load model with fix for unsupported layers
def load_model_with_fix(model_path, model_json_path=None):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except TypeError as e:
        print(f"Error loading model: {e}. Attempting to fix...")
        if model_json_path:
            with open(model_json_path, "r") as json_file:
                model_config = json.load(json_file)
            for layer in model_config["config"]["layers"]:
                if layer["class_name"] == "DepthwiseConv2D":
                    layer["config"].pop("groups", None)
            model = model_from_json(json.dumps(model_config))
            model.load_weights(model_path.replace(".h5", "_weights.h5"))
            print("Fixed and loaded model successfully.")
            return model
        else:
            print("No model JSON path provided for fixing.")
            return None

# Initialize or connect to the attendance database
def init_attendance_db(db_path="attendance.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            PRIMARY KEY (student_id, date)
        )
    ''')
    conn.commit()
    return conn, cursor

# Function to mark attendance in the database
def mark_attendance(cursor, conn, student_id, name):
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    cursor.execute('SELECT * FROM attendance WHERE student_id = ? AND date = ?', (student_id, current_date))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO attendance (student_id, name, date, time) VALUES (?, ?, ?, ?)',
                       (student_id, name, current_date, current_time))
        conn.commit()
        print(f"Attendance marked for {name} at {current_time}.")
    else:
        print(f"Attendance already marked for {name}.")

# Real-time face recognition
def generate_frames():
    global model
    conn, cursor = init_attendance_db()
    cap = cv2.VideoCapture(0)
    class_names = ["Aavani", "Amaljith", "Bharath Dev", "Manikandan", "Nafl"]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = img_to_array(face_resized) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]

            if confidence >= 0.6:
                label = f"{predicted_class}: {confidence:.2f}"
                color = (0, 255, 0)
                mark_attendance(cursor, conn, str(predicted_class_idx), predicted_class)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    conn.close()

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mark_attendance")
def mark_attendance_page():
    return render_template("mark_attendance.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/check_attendance")
def check_attendance():
    conn, cursor = init_attendance_db()
    cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()
    conn.close()
    return render_template("check_attendance.html", data=rows)

# Run Flask App
if __name__ == "__main__":
    model = load_model_with_fix("model1.h5", "model_config.json")
    app.run(debug=True)
