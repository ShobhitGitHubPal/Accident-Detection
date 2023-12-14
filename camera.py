# c='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4'
# d='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/Amazing  car accident😱😱 viral short video##ytviralcar.mp4'
# ##e='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/BEST IDIOTS IN TRUCKS & CARS FAILS 2023 - CUTTING TREE FAILS - TOTAL IDIOTS AT WORK 2023.mp4'
# ##f='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/Dashboard camera shows head-on collision on Highway 178.mp4'
# #d='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videoplayback (online-video-cutter.com).mp4'








import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
# from main import video_feed

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    # print('kfrhirygiigoiho5h',a)
    a='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/CCTV Car Crash Detection Sample.mp4'
    # b='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/maitighar.mp4'
    # g='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/koteshore.mp4'
    
   
    video = cv2.VideoCapture(a) # for camera use video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            # cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (500, 300), font, 1, (250, 250, 0), 2)
            
        # ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#
        # frame = buffer.tobytes()
        # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')#
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        dete=cv2.imshow('Video', frame) 
        if dete:
            return dete 
        
        
        # video.release()
        # cv2.destroyAllWindows()
                
if __name__ == '__main__':
    startapplication()



