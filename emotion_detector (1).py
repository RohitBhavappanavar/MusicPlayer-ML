from flask import *
from pygame import mixer
import time
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array,load_img
#from keras.utils import img_to_array,load_img

import speech_recognition as sr
import pyttsx3

happy = ['Nashe_Si_Chadh_Gayi.mp3','Param_Sundari_Mimi.mp3','Badtameez_Dil_Full_Song.mp3','Mehabooba_Kannada____KGF_Chapter_2.mp3','Chandralekha A Gentleman 128 Kbps.mp3','Chikni Chameli Agneepath 128 Kbps.mp3','happy.mp3.mp3']
fear=['Sulthan_Song_KGF_Chapter_2.mp3','Iktara Wake Up Sid 128 Kbps.mp3','Iktara Wake Up Sid 128 Kbps.mp3','Jug_Jug_Jeeve_Shiddat.mp3','He_Sharade_.mp3']
disgust=['Pressure Martin Garrix 128 Kbps.mp3','Masakali___Delhi_6.mp3','Willow Evermore (deluxe Version) 128 Kbps.mp3','Maate_Poojaka-Patriotic_Song.mp3']
angry=['Ishq Bulaava Hasee Toh Phasee 128 Kbps (1).mp3','_Sooraj_Dooba_Hain__.mp3','Raju_Kannada_Medium___Kodeyondara.mp3','Kala_Chashma___Baar_Baar_Dekho.mp3','Shayad - Love Aaj Kal 128 Kbps.mp3']
sad=['Aas Pass Hai Khuda Anjaana Anjaani 128 Kbps.mp3','Alemaariye.mp3','Gandi_Baat.mp3','Neenendare_Nannolage___Junglee.mp3','Suna Hai Sanak 128 Kbps.mp3','HERO_-Nenapina_Hudugiye.mp3']
surprise=['saree_ke_fall_sa_R_Rajkumar.mp3','BAHADDUR_-_NEENE_NEENE.mp3','Tu Hi Hai Dear Zindagi 128 Kbps.mp3','One_Two_Three_Four_Chennai_Express.mp3','Tu Jaane Na Ajab Prem Ki Ghazab Kahani 128 Kbps.mp3']
neutral = ['Pushpa__Srivalli.mp3','KHAIRIYAT__BONUS_TRACK.mp3','Aayat___Bajirao_Mastani.mp3',"Soch Na Sake (male) Airlift 128 Kbps.mp3",'Mynaa___Modala_Maleyanthe.mp3','Hamdard_Ek_Villain.mp3',"The Breakup Song Ae Dil Hai Mushkil 128 Kbps.mp3","neutral.mp3.mp3"]
cnt = 0
# Initialize the recognizer
r = sr.Recognizer()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/start')
def start():
    # load model
    model = model_from_json(open("fer.json", "r").read())
    # load weights
    model.load_weights('fer.h5')

    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i = 0
    j = 0
    k=0
    cnt =0
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
            img_pixels = img_to_array(roi_gray)
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
                    track=happy[cnt]
                    return render_template('song.html',track=track,e="h",exp="Happy",cnt=cnt,songs=happy)
            if predicted_emotion == 'neutral':
                print(j)
                j = j + 1
                i=k = 0
                if j > 10:
                    print(j)
                    track = neutral[cnt]
                    return render_template('song.html',track=track,e="n",exp="Neutral",cnt=cnt,songs=neutral)
            if predicted_emotion == 'fear':
                print(k)
                k = k + 1
                j=i = 0
                if k > 10:
                    print(j)
                    track=fear[cnt]
                    return render_template('song.html',track=track,e="f",exp="Fear",cnt=cnt,songs=fear)
            if predicted_emotion == 'sad':
                i = i + 1
                j = k = 0
                if i > 10:
                    print(i)
                    track=sad[cnt]
                    return render_template('song.html',track=track,e="sad",exp="Sad",cnt=cnt,songs=sad)
            if predicted_emotion == 'surprise':
                print(j)
                j = j + 1
                i=k = 0
                if j > 10:
                    print(j)
                    track = surprise[cnt]
                    return render_template('song.html',track=track,e="s",exp="Surprise",cnt=cnt,songs=surprise)
            if predicted_emotion == 'disgust':
                print(k)
                k = k + 1
                j=i = 0
                if k > 10:
                    print(j)
                    track=disgust[cnt]
                    return render_template('song.html',track=track,e="d",exp="Disgust",cnt=cnt,songs=disgust)
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.route('/control/<e>/<c>')
def controls(e,c):
    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.5)
            audio2 = r.listen(source2)
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print(MyText)
            if MyText == 'next':
                cnt = int(c)+1
                if cnt > 2:
                    cnt=2

                if e == "h":
                    track = happy[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=happy)
                if e == "n":
                    track = neutral[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=neutral)
                if e == "f":
                    track = fear[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=fear)
                if e == "sad":
                    track = sad[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=sad)
                if e == "s":
                    track = surprise[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=surprise)
                if e == "d":
                    track = disgust[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=disgust)
            if MyText == "previous" :
                cnt = int(c)-1
                if cnt < 0:
                    cnt = 0
                if e == "h":
                    track = happy[cnt]
                    return render_template("song.html", track=track, cnt=cnt,e=e,songs=happy)
                if e == "n":
                    track = neutral[cnt]
                    return render_template("song.html", track=track, cnt=cnt,e=e,songs=neutral)
                if e == "f":
                    track = fear[cnt]
                    return render_template("song.html", track=track, cnt=cnt,e=e,songs=fear)
                if e == "sad":
                    track = sad[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=sad)
                if e == "s":
                    track = surprise[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=surprise)
                if e == "d":
                    track = disgust[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=disgust)
            if MyText == 'stop':
                return render_template('index.html')
            else:
                cnt = int(c)
                if e == "h":
                    track = happy[int(c)]
                    return render_template("song.html", track=track, cnt=int(c), e=e,songs=happy)

                if e == "n":
                    track = neutral[int(c)]
                    return render_template("song.html", track=track, cnt=int(c), e=e,songs=neutral)

                if e == "f":
                    track = fear[int(c)]
                    return render_template("song.html", track=track, cnt=int(c), e=e,songs=fear)
                if e == "sad":
                    track = sad[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=sad)
                if e == "s":
                    track = surprise[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=surprise)
                if e == "d":
                    track = disgust[cnt]
                    return render_template("song.html",track=track,cnt=cnt,e=e,songs=disgust)







    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occured")
        cnt = int(c)
        if e == "h":
            track = happy[int(c)]
            return render_template("song.html", track=track,e=e, cnt=int(c))
        if e == "n":
            track = neutral[int(c)]
            return render_template("song.html", track=track,e=e, cnt=int(c))
        if e == "f":
            track = fear[int(c)]
            return render_template("song.html", track=track,e=e, cnt=int(c))
        if e == "sad":
            track = sad[cnt]
            return render_template("song.html", track=track, cnt=cnt, e=e)
        if e == "s":
            track = surprise[cnt]
            return render_template("song.html", track=track, cnt=cnt, e=e)
        if e == "d":
            track = disgust[cnt]
            return render_template("song.html", track=track, cnt=cnt, e=e)





if __name__ == '__main__':
    app.run()