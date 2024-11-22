import cv2
import random
trained_face_data = cv2.CascadeClassifier('Smile Detection/haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('Smile Detection/haarcascade_smile.xml')
trained_eye_data = cv2.CascadeClassifier('Smile Detection/haarcascade_eye.xml')


# img = cv2.imread('IMG_9906.JPG')
webcam = cv2.VideoCapture(0)

while True:
    successframe, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_frame = frame[y:y+h, x:x+w]
        grayscaled_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        smile_coordinates = trained_smile_data.detectMultiScale(grayscaled_face,scaleFactor=1.5,minNeighbors=20)
        eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_face,scaleFactor=1.5,minNeighbors=10)
        for (x_,y_,w_,h_) in smile_coordinates:
            cv2.rectangle(face_frame, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
        if len(smile_coordinates) > 0:
            cv2.putText(frame, "Smiling", (x, y+h+40), fontScale=1,color=(0,0,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        for (x_,y_,w_,h_) in eye_coordinates:
            cv2.rectangle(face_frame, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
        if len(face_frame) > 0:
            cv2.imshow('TestFaceRecog', face_frame)
    
        cv2.line(frame, )
    cv2.imshow('TestSmileRecog', frame)

    key = cv2.waitKey(1)
    if key == 113:
        break

# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)
# for (x,y,w,h) in face_coordinates:
#     cv2.rectangle(img , (x, y), (x+w, y+h), (random.randint(0,256), random.randint(0,256), random.randint(0,256)), 2)
# cv2.imshow('TestFaceRecog', img)
# cv2.waitKey()
# print("Code Completed")
 