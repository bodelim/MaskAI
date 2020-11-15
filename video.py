#-*- coding:utf-8 -*-



from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from playsound import playsound
import time
import requests
import threading
from gtts import gTTS
import speech_recognition as sr





url = "https://maker.ifttt.com/trigger/MaskNotice/with/key/c4CPrrPzQibURECdA5f2JC"
data = {'value1': '1층로비', 'value2': '구역에서 마스크 미착용학생이 감지되었습니다.', 'value3': '+8201074517300'}


nomaskAmount = 0
checkNoAmount = 0
maskokAmount = 0
soundpower = False
noticecoolPower = False
noticecoolTime = '0'
noticeturn = 0

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')

cap = cv2.VideoCapture(0)
ret, img = cap.read()


def speak():
    #tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    #tts.save(filename)
    playsound(filename)


    
def noticeStart():
    global noticecoolTime
    global noticeturn
    global noticecoolPower
    if noticeturn == 0:
        print(noticeturn)
        noticeturn = 1
        print("마스크 착용독려 음성이 전송될 예정입니다.")
        #playsound("voice.mp3")
        thread_tts = threading.Thread(target=speak)
        thread_tts.start()
        noticecoolPower = False
        print('[MaskAi] 마스크 미착용감 알림 전송 준비중입니다.')
        #re = requests.post(url, data=data)
        print('[MaskAi] 선생님께 마스크 미착용감지 메세지가 전송되었습니다.')
        noticecoolTime = str(time.time()).split(".")[0]
        noticecoolPower = False
       
    if noticeturn == 1:
        noticecoolPower = False
        #if int(str(time.time()).split(".")[0])-int(noticecoolTime) < 3:
         #   threading.Timer(1, noticeStart).start()
        if int(str(time.time()).split(".")[0])-int(noticecoolTime) >= 10:
            noticecoolPower = False
            print("마스크 착용독려 음성이 전송될 예정입니다.")
            print(noticeturn)
            #playsound("voice.mp3")
            thread_tts = threading.Thread(target=speak)
            thread_tts.start()
            noticecoolPower = False
            print('[MaskAi] 마스크 미착용감지 알림 전송 준비중입니다.')
            #re = requests.post(url, data=data)
            print('[MaskAi] 선생님께 마스크 미착용감지 메세지가 전송되었습니다.')
            noticecoolTime = str(time.time()).split(".")[0]
            noticecoolPower = False
            
     
def maskRender():
    global nomaskAmount
    global nomaskcoolPower
    global nomaskcoolTime
    global maskokAmount
    global noticecoolPower
    global noticecoolTime
    global out
    
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(500, 500), mean=(104., 177., 123.))
        facenet.setInput(blob)

        #결과추론단계
        dets = facenet.forward()


        result_img = img.copy()

        for i in range(dets.shape[2]):
            #결과의 정확성
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            #얼굴 커트
            face = img[y1:y2, x1:x2]

            try:
                face_input = cv2.resize(face, dsize=(230, 230))
                #RGB -> BGR 변환
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0)

                mask, nomask = model.predict(face_input).squeeze()
            except Exception as e:
                print('')
            if mask > nomask:
                color = (0, 255, 0)#초록색                                                                   
                label = 'Mask Ok %d%%' % (mask * 100)
                maskokAmount = maskokAmount + 1
                #print('마스크정상착용 카운트:'+str(maskokAmount))
            else:
                color = (0, 0, 255)#빨간색
                label = 'NO MASK!!!! %d%%' % (nomask * 100)
                nomaskAmount = nomaskAmount + 1
                #print("메세지쿨타임시간: "+ str(noticecoolTime))
                #print("메세지쿨타임전원: "+ str(noticecoolPower))
                #print('마스크미착용 카운트: '+str(nomaskAmount))
                #마스크 미착용 고발
                if noticecoolPower == False:
                    noticecoolPower = True
                    
                    noticeStart()
                    #re = requests.post(url, data=data)

        #if noticecoolPower == True:
         #       if int(str(time.time()).split(".")[0])-int(noticecoolTime) >= 5:
          #          noticecoolPower = False




            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)
            #cv2.putText(result_img, text='TestVersion', org= (0,+20), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (255,255,255), thickness= 2)

        #out.write(result_img)
        cv2.imshow('result', result_img)
        if cv2.waitKey(1) == ord('q'):
            break
thread_render = threading.Thread(target=maskRender)
thread_render.start()

our.release()
cap.release()
