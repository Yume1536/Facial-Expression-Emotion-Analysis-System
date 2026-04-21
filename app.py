from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, g, Response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import sqlite3
import json
import sys

# -----------------------------
# Flask Configuration
# -----------------------------
app = Flask(__name__)
app.secret_key = "secretkey"
app.config['UPLOAD_FOLDER'] = "static/uploads/"
DATABASE = 'facial_emotion.db'

# -----------------------------
# SQLite Database Connection
# -----------------------------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Helper to save predictions
def save_prediction(username, emotion):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO predictions (username, emotion) VALUES (?, ?)", (username, emotion))
    db.commit()

# -----------------------------
# Load Model
# -----------------------------
model_path = "CNNModel_ck_5emo.h5"
model = load_model(model_path, compile=False)
emotion_labels = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise']

# -----------------------------
# Confusion Matrix Folder
# -----------------------------
CONF_MATRIX_FOLDER = "outputs/confusion_matrix"
os.makedirs(CONF_MATRIX_FOLDER, exist_ok=True)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return redirect('/login')

# -----------------------------
# Register
# -----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()

        if user:
            return render_template("register.html", error="⚠️ User already exists. Try again.")

        cursor.execute("INSERT INTO users (name, password) VALUES (?, ?)", (name, hashed_password))
        db.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')

    return render_template('register.html')

# -----------------------------
# Login
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['username']
        password = request.form['password']

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['username'] = name
            return redirect('/dashboard')
        else:
            return render_template("login.html", error="⚠️ Invalid credentials. Try again.")

    return render_template('login.html')

# -----------------------------
# Dashboard
# -----------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/login')

    # Get latest confusion matrix
    conf_matrix_file = None
    files = os.listdir(CONF_MATRIX_FOLDER)
    if files:
        conf_matrix_file = files[-1]

    # Get prediction history
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT emotion, timestamp FROM predictions WHERE username = ? ORDER BY id DESC LIMIT 5", (session['username'],))
    history = cursor.fetchall()

    return render_template('dashboard.html',
                           username=session['username'],
                           conf_matrix_file=conf_matrix_file,
                           history=history)

# -----------------------------
# Train Model
# -----------------------------
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Full command as a single string
        cmd = "python trainer.py -d fer -m CNNModel -em Angry,Happy,Surprise -ep 25 -tr 0.7 -bs 16 -sa 1 -sm 1 -scm 1 -sth 1 -tg 4"

        # Run the command in a shell and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,          # Required for single-string commands
            cwd=os.getcwd()       # Ensure it runs in project directory
        )

        output = ""
        for line in process.stdout:
            output += line

        # Optionally, check for a confusion matrix image
        conf_matrix_file = None
        possible_files = os.listdir(CONF_MATRIX_FOLDER)
        if possible_files:
            conf_matrix_file = possible_files[-1]  # latest file

        return render_template(
            "dashboard.html",
            train_output=output,      # or test_output for test route
            conf_matrix_file="conf_matrix.png"  # image name only
        )
        

    except Exception as e:
        return render_template("dashboard.html", train_output=f"Error running trainer: {str(e)}")



# Serve confusion matrix images
@app.route('/confusion_matrix/<filename>')
def confusion_matrix(filename):
    return send_from_directory(CONF_MATRIX_FOLDER, filename)

# -----------------------------
# Test Model
# -----------------------------
@app.route('/test', methods=['POST'])
def test_model():
    try:
        python_exe = sys.executable
        test_script = os.path.join(os.path.dirname(__file__), "test_model.py")

        process = subprocess.Popen([python_exe, test_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = "".join(process.stdout.readlines())

        conf_dir = os.path.join("static", "confusion_matrix")
        os.makedirs(conf_dir, exist_ok=True)

        files = os.listdir(conf_dir)
        conf_matrix_file = files[-1] if files else None

        return render_template("dashboard.html", test_output=output, conf_matrix_file=conf_matrix_file)
    except Exception as e:
        return render_template("dashboard.html", test_output=f"Error running test: {str(e)}")

# -----------------------------
# Upload Image Prediction
# -----------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect('/login')
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "Invalid image. Please upload a proper image."

            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            emotion = emotion_labels[np.argmax(pred)] if pred.size > 0 else "Unknown"

            # Save prediction to DB
            save_prediction(session['username'], emotion)

            return render_template('result.html', emotion=emotion, image_file=filename)
    return render_template('upload.html')

# -----------------------------
# Live Webcam Prediction
# -----------------------------
camera = None

@app.route('/live')
def live_webcam():
    if 'username' not in session:
        return redirect('/login')
    return render_template('live.html')


def gen_frames():
    global camera
    # Create a new camera instance every time live page is opened
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        pred = model.predict(face)
        emotion = emotion_labels[np.argmax(pred)]

        # Display result on video
        cv2.putText(frame, emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()  # release automatically when loop ends


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
    return "Camera stopped", 200

# -----------------------------
# Logout
# -----------------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

