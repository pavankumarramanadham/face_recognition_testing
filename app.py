import cv2
import os
import numpy as np
import dlib
import face_recognition
from flask import Flask, request, render_template, Response, jsonify
from datetime import date, datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
import base64
from io import BytesIO
from PIL import Image
import atexit

# Flask app initialization
app = Flask(__name__)

# SQLAlchemy setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Attendance model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(100), nullable=False)

# Initialize database
with app.app_context():
    db.create_all()

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load dlib face detector (No shape predictor required)
face_detector = dlib.get_frontal_face_detector()

# Ensure necessary directories exist
os.makedirs('static/faces', exist_ok=True)

# Load known face encodings
def load_known_faces():
    known_encodings = {}
    registered_users = os.listdir('static/faces')
    
    for user in registered_users:
        user_folder = f'static/faces/{user}'
        encodings = []
        
        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                encodings.append(face_encodings[0])
        
        if encodings:
            known_encodings[user] = np.mean(encodings, axis=0)
    
    return known_encodings

known_faces = load_known_faces()

# Identify a face using dlib face recognition
def identify_face(face_encoding):
    global known_faces
    if known_faces:
        names = list(known_faces.keys())
        encodings = list(known_faces.values())
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if matches else None
        
        if best_match_index is not None and matches[best_match_index]:
            return names[best_match_index]  # Return identified name
    
    return None  # Face not recognized

# Flask route to process new user registration
@app.route('/process_new_user', methods=['POST'])
def process_new_user():
    data = request.get_json()
    username, userid, images = data['username'], data['userid'], data['images']
    user_folder = f'static/faces/{username}_{userid}'
    os.makedirs(user_folder, exist_ok=True)
    
    for i, img_data in enumerate(images):
        with Image.open(BytesIO(base64.b64decode(img_data.split(',')[1]))) as img:
            img = img.resize((320, 240))
            img.save(f"{user_folder}/{username}_{i}.jpg")
    
    global known_faces
    known_faces = load_known_faces()  # Reload known faces
    return jsonify({'message': f'User {username} added successfully!'})

# Video streaming setup
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html', totalreg=len(os.listdir('static/faces')), datetoday2=datetoday2)

# Process attendance with face recognition
@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    with Image.open(BytesIO(base64.b64decode(image_data))) as img:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    face_encodings = face_recognition.face_encodings(frame)
    if face_encodings:
        identified_person = identify_face(face_encodings[0])
        if identified_person:
            add_attendance(identified_person)
            return jsonify({'message': f'Attendance marked for {identified_person}'})
        else:
            return jsonify({'message': 'Face not recognized. Please register first.'})
    
    return jsonify({'message': 'No face detected, try again'})

# Add attendance (prevents duplicate entries within 10 minutes)
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    
    last_entry = Attendance.query.filter_by(roll=userid, date=datetoday).order_by(Attendance.id.desc()).first()
    if last_entry:
        last_time = datetime.strptime(last_entry.time, "%H:%M:%S")
        if (datetime.now() - last_time).total_seconds() < 600:
            return
    
    db.session.add(Attendance(name=username, roll=userid, time=current_time, date=datetoday))
    db.session.commit()

# Shutdown and release camera
@app.route('/shutdown')
def shutdown():
    release_camera()
    return "Camera released"

def release_camera():
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()

atexit.register(release_camera)  # Ensure camera release on exit

if __name__ == '__main__':
    app.run(debug=True)
