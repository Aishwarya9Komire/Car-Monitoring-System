
# coding: utf-8

# In[ ]:
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import time
import datetime
import cv2
import dlib
import numpy as np
import imutils
import winsound
import argparse

duration = 2000
freq = 500
c=0
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15
MINI = 3
eye_avg_ratio = 0.0
COUNTER = 0
ALARM_ON = True
eye_avg_ratio = 0.0
yawns = 0
yawn_status = False

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def eye_ratio(eye):

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])


	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear


print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

print("[INFO] starting video stream thread....")
vs = VideoStream(src=0).start()
time.sleep(1.0)

frame = vs.read()
frame = imutils.resize(frame,width=450)
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_SIMPLEX
input_frame = img

while True:
    frame = vs.read()   
    image_landmarks, lip_distance = mouth_open(frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img, 0)
    
    prev_yawn_status = yawn_status
    
    if lip_distance > 25:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        winsound.Beep(freq, duration)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
    

    for face in faces:
        
        face_data = face_utils.shape_to_np(predictor(img,face))
        left_eye = face_data[36:42]
        right_eye = face_data[42:48]
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        
        cv2.drawContours(frame,[leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame,[rightEyeHull], -1, (0, 255,0), 1)

        leftEAR = eye_ratio(left_eye)
        rightEAR = eye_ratio(right_eye)

        eye_avg_ratio = (leftEAR + rightEAR)/2.0       

        if eye_avg_ratio < EYE_AR_THRESH:
            COUNTER = COUNTER + 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                               
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(freq,duration)
                
        if eye_avg_ratio < 0.25:
            c= c+1
            if c >= MINI:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(freq, duration)

                
        if eye_avg_ratio < 0.2:
            c= c+1
            if c >= MINI:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(freq, duration)                
         
        else:
            COUNTER = 0
            c=0
            ALARM_ON = True
        cv2.putText(frame, "EAR : {:.2f}".format(eye_avg_ratio),
                        (300,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
            

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Drowsy Detection', frame )
    if(COUNTER == 51):
        cv2.waitKey(1000)
        normal_count = 0
    else:
        wait = cv2.waitKey(1)
        if wait==ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break 

    
    if cv2.waitKey(1) == 13:
        break
        
vs.release()
cv2.destroyAllWindows() 
vs.stop()
