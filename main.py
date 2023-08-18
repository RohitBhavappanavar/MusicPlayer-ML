from flask import *
from pygame import mixer
import time
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
model = model_from_json(open("fer.json", "r").read())
# load weights
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
i = 0
j = 0
k = 0
happy = ['Chandralekha A Gentleman 128 Kbps.mp3','Chikni Chameli Agneepath 128 Kbps.mp3','happy.mp3.mp3']
fear=['Iktara Wake Up Sid 128 Kbps.mp3','Iktara Wake Up Sid 128 Kbps.mp3']
disgust=['Pressure Martin Garrix 128 Kbps.mp3','Willow Evermore (deluxe Version) 128 Kbps.mp3']
angry=['Ishq Bulaava Hasee Toh Phasee 128 Kbps (1).mp3','Shayad - Love Aaj Kal 128 Kbps.mp3']
sad=['Aas Pass Hai Khuda Anjaana Anjaani 128 Kbps.mp3','Suna Hai Sanak 128 Kbps.mp3']
surprise=['Tu Hi Hai Dear Zindagi 128 Kbps.mp3','Tu Jaane Na Ajab Prem Ki Ghazab Kahani 128 Kbps.mp3']
neutral = ["Soch Na Sake (male) Airlift 128 Kbps.mp3", "The Breakup Song Ae Dil Hai Mushkil 128 Kbps.mp3","neutral.mp3.mp3"]
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        if predicted_emotion == 'happy':
            i = i + 1
            j = k = 0
            if i > 10:
                print(i)

                for son in l:
                    mixer.init()
                    mixer.music.load('happy/'+str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'neutral':
            print(j)
            j = j + 1
            i = k = 0
            if j > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load('neutral/'+str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'surprise':
            print(k)
            k = k + 1
            j = i = 0
            if k > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load("surprise/"+str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'sad':
            print(k)
            k = k + 1
            j = i = 0
            if k > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load("sad/"+str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'angry':
            print(k)
            k = k + 1
            j = i = 0
            if k > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load("static/angry/", +str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'disgust':
            print(k)
            k = k + 1
            j = i = 0
            if k > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load("static/disgust/", +str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        if predicted_emotion == 'fear':
            print(k)
            k = k + 1
            j = i = 0
            if k > 10:
                print(j)

                for son in l:
                    mixer.init()
                    mixer.music.load("static/fear/", +str(son))
                    mixer.music.set_volume(0.9)
                    mixer.music.play()
                    time.sleep(40)
                    mixer.music.stop()

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
