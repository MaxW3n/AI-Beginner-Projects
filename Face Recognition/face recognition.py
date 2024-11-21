import cv2
import random
trained_face_data = cv2.CascadeClassifier('Face Recognition/haarcascade_frontalface_default.xml')

# img = cv2.imread('IMG_9906.JPG')
webcam = cv2.VideoCapture(0)

while True:
    successframe, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, minNeighbors=5)
    edge = cv2.Canny(frame, 50, 100)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(x, ", ", y)
    cv2.imshow('TestFaceRecog', frame)
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
 