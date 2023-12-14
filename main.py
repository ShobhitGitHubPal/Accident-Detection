# # from camera import startapplication
# from flask import Flask, render_template, request, Response,redirect,send_file
# import os
# from werkzeug.utils import secure_filename
# app = Flask(__name__)

# import cv2
# from detection import AccidentDetectionModel
# import numpy as np
# import os
# # from main import video_feed

# model = AccidentDetectionModel("model.json", 'model_weights.h5')
# font = cv2.FONT_HERSHEY_SIMPLEX
# @app.route('/detect', methods=['POST','GET'])
# def startapplication():
#     # print('kfrhirygiigoiho5h',a)
#     a='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/CCTV Car Crash Detection Sample.mp4'
#     # b='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/maitighar.mp4'
#     # g='C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/videos/koteshore.mp4'
    
   
#     video = cv2.VideoCapture(a) # for camera use video = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video.read()
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         roi = cv2.resize(gray_frame, (250, 250))

#         pred, prob = model.predict_accident(roi[np.newaxis, :, :])
#         if(pred == "Accident"):
#             prob = (round(prob[0][0]*100, 2))
            
#             # to beep when alert:
#             # if(prob > 90):
#             #     os.system("say beep")

#             # cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
#             cv2.putText(frame, pred+" "+str(prob), (500, 300), font, 1, (250, 250, 0), 2)
            
#         # ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#
#         # frame = buffer.tobytes()
#         # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')#
        
#         if cv2.waitKey(33) & 0xFF == ord('q'):
#             return
#         dete=cv2.imshow('Video', frame) 
#         if dete:
#             return dete 




# video_path = ''  # Define the path to the uploaded video file

# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(startapplication(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Define a route that starts the camera application
# @app.route('/', methods=['POST','GET'])  #
# def Upload():
#     global video_path
#     if request.method == "POST":
#         file= request.files["filename"]

#         if file:
#             filename = secure_filename(file.filename)
#             video_path = os.path.join('C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/testuploads', filename)
#             file.save(video_path)
#             # file.save(os.path.join('C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/testuploads', filename))
#             print('uploaded')
#             # return redirect('/video_feed')
#             startapplication(os.path.join('C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/testuploads', filename))
            
#     # return 'khu/wgu//fugfu'
#     return render_template('index.html')
#     # startapplication()


# if __name__ == '__main__':
#     app.run(debug=True)







from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

app = Flask(__name__)

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
video_path = ''
processed_frame = None

def startapplication(video_path):
    global processed_frame
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)
            # cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            # cv2.putText(frame, pred + " " + str(prob), (450, 450), font, 1, (250, 250, 0), 2)

            # Assuming 'frame' is your image or video frame
            height, width, _ = frame.shape

            # Position the text at the center
            text = f"{pred} {prob}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            # Draw text on the frame
            cv2.putText(frame, text, (text_x, text_y), font, 1, (250, 250, 0), 2)

        processed_frame = frame

    video.release()

@app.route('/video_feed')
def video_feed():
    global processed_frame

    def generate():
        while True:
            if processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST', 'GET'])
def upload():
    global video_path

    if request.method == "POST":
        file = request.files["filename"]

        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join('C:/Users/LENOVO/Desktop/hhh/Accident-Detection-System/testuploads', filename)
            file.save(video_path)
            print('uploaded')
            startapplication(video_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=5050)