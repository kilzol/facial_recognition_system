import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)


imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

nimgs = 10

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,User_ID,In_Time,Out_Time')

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []
        

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    user_id = df['User_ID']
    intimes = df['In_Time']
    outtimes = df['Out_Time']
    l = len(df)
    return names, user_id, intimes, outtimes, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['User_ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},"0"')
            

def out_times(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')


    df['Out_Time'] = df['Out_Time'].astype(str)
    if int(userid) in list(df['User_ID']):
        row_index= df.loc[df['Name'] == username].index[0]
        if df.loc[row_index, 'Out_Time'] != "0":
          print("Out_Time already marked")
          return
        df.loc[row_index, 'Out_Time'] = current_time
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
        print(df)




@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    names, user_id, intimes,outtimes, l = extract_attendance()
    ret = True
    cap = cv2.VideoCapture(0)
#    while ret:
    ret, frame = cap.read()
    if len(extract_faces(frame)) > 0:
        (x, y, w, h) = extract_faces(frame)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
        cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
        face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        add_attendance(identified_person)
            

#        cv2.imshow('Attendance', imgBackground)
#    if cv2.waitKey(1) == 27:
#       break
#    return render_template('attendance.html')
    cap.release()
    cv2.destroyAllWindows()
    names, user_id, intimes,outtimes, l = extract_attendance()
    return render_template('attendance.html',names=names,user_id = user_id,intimes=intimes,outtimes=outtimes,l=l)




@app.route('/outtime', methods=['GET', 'POST'])
def outtime():
    names, user_id, intimes,outtimes, l = extract_attendance()
    ret = True
    cap = cv2.VideoCapture(0)
#    while ret:
    ret, frame = cap.read()
    if len(extract_faces(frame)) > 0:
        (x, y, w, h) = extract_faces(frame)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
        cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
        face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        out_times(identified_person)
            

#        cv2.imshow('Attendance', imgBackground)
#    if cv2.waitKey(1) == 27:
#       break
#    return render_template('attendance.html')
    cap.release()
    cv2.destroyAllWindows()
    names, user_id, intimes,outtimes, l = extract_attendance()
    return render_template('attendance.html',names=names,user_id = user_id,outtimes=outtimes,intimes=intimes,l=l)






@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        user_id = request.form['user_id']
        userimagefolder = 'static/faces/'+name+'_'+str(user_id)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    nameo = name+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+nameo, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
    return render_template('register.html')


    
if __name__ == '__main__':
    app.run(debug=True)