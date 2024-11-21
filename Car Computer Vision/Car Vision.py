import cv2
trained_car_data = cv2.CascadeClassifier('Car Computer Vision/cars.xml')
trained_pedestrian_data = cv2.CascadeClassifier('Car Computer Vision/haarcascade_fullbody.xml')
car_img = cv2.imread("Car Computer Vision/typical_view.jpg")
dashcam = cv2.VideoCapture('Car Computer Vision/Angry Pedestrian Blocks Cyclist As He Races Through Zebra Crossing.mp4')

while True:
    successframe, frame = dashcam.read()
    grayscale_vers = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates = trained_car_data.detectMultiScale(grayscale_vers, scaleFactor=1.1)
    pedes_coordinates = trained_pedestrian_data.detectMultiScale(grayscale_vers, scaleFactor=1.1)
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x,y,w,h) in pedes_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('TestcarRecog', frame)
    key = cv2.waitKey(1)
    if key == 113 or not successframe:
        break

