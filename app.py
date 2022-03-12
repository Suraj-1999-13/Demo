from flask import Flask,render_template,Response
import cv2
import mediapipe as mp
import time
import numpy as np

app = Flask(__name__)
cap=cv2.VideoCapture(0)
no_img = cv2.imread('C:/Users/DELL/Desktop/Project/NO_HAND.jpg')

def generate_frames():
    mpHands=mp.solutions.hands
    hands=mpHands.Hands(max_num_hands=1)
    mpDraw=mp.solutions.drawing_utils
    padd_amount = 15

    while True:
        success, img=cap.read()

        if not success:
            break
        else:
            #img = cv2.flip(img, 1)
            roi=img[210:101,202:260].copy()
            imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result=hands.process(imgRGB)

        if result.multi_hand_landmarks:
            landmarks = [] 
            for handLms in result.multi_hand_landmarks:
                
                for id, landmark in enumerate(handLms.landmark):
                    height, width, _ = img.shape
                    landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                         (landmark.z * width)))
                mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)
                x_coordinates = np.array(landmarks)[:,0]
                y_coordinates = np.array(landmarks)[:,1]
                x1  = int(np.min(x_coordinates) - padd_amount)
                y1  = int(np.min(y_coordinates) - padd_amount)
                x2  = int(np.max(x_coordinates) + padd_amount)
                y2  = int(np.max(y_coordinates) + padd_amount)
                cv2.rectangle(img, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
                roi=img[y1:y2,x1:x2].copy()
            
 
            ret,buffer=cv2.imencode('.jpg',img)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def frames():
    ptime=0
    ctime=0
    while True:
        success, frame = cap.read()
        if success:
            try:
                #frame = cv2.flip(frame, 1)
                ctime=time.time()
                fps=1/(ctime-ptime)
                ptime=ctime
                cv2.putText(frame,str(int(fps)),(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
                ret, buffer = cv2.imencode('.jpg',frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/roi')
def roi():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)