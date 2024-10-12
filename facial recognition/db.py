from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import face_recognition
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for file_name in os.listdir('known_faces'):
        if file_name.endswith('.jpg'):
            image = face_recognition.load_image_file(f'known_faces/{file_name}')
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file_name)[0])
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

def mark_attendance(name):
    df = pd.read_csv('attendance.csv')
    if name not in df['Name'].values:
        new_row = {'Name': name, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = df.append(new_row, ignore_index=True)
        df.to_csv('attendance.csv', index=False)

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                mark_attendance(name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    df = pd.read_csv('attendance.csv')
    return render_template('attendance.html', tables=[df.to_html(classes='data', header="true")])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if not name:
            flash('Name is required!')
            return redirect(url_for('register'))

        file = request.files['image']
        if not file:
            flash('Image is required!')
            return redirect(url_for('register'))

        file.save(os.path.join('known_faces', f'{name}.jpg'))
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()
        flash('User registered successfully!')
        return redirect(url_for('index'))
    return render_template('register.html')

if __name__ == '__main__':
    if not os.path.exists('attendance.csv'):
        df = pd.DataFrame(columns=['Name', 'Time'])
        df.to_csv('attendance.csv', index=False)
    app.run(debug=True)
