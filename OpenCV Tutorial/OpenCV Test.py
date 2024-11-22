import cv2
import numpy as np

img = cv2.imread('OpenCV Tutorial/tumblr_n2lq5lwa8H1sikueao1_1280.jpg')
# Making Blank Canvas
blank = np.zeros((500, 500, 3), dtype='uint8')
# Resizing Image
def rescale(frame, scale):
    # img.shape[0] is height and img.shape[1] is width
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    # INTER_AREA is for shrinking, INTER_LINEAR is for upscaling, INTER_CUBIC is for high quality upscale
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
# Putting Stuff on Blank Canvas
blank[0:250, 0:250] = 0,0,255
cv2.rectangle(blank, (0,250),(250,500),(255,0,0), thickness=10)
cv2.rectangle(blank, (0,250),(250,500),(0,255,0), thickness=cv2.FILLED)
cv2.circle(blank, (blank.shape[1]//4, blank.shape[0]//4), int(blank.shape[1]//4), (255,0,0), thickness=10)
cv2.line(blank, (250,0),(250, 500),(255,0,0), thickness=10)
cv2.putText(blank, "HI", (250,500),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=5.0, color=(255,255,255), thickness=5)
# Photo Adjustments
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (9,9), cv2.BORDER_DEFAULT)
# Array slicing: img[yvalue_1:yvalue_2,xvalue_1:xvalue_2]
cropped = img[500:800, 500:800]
# Edge Shenanigans
canny = cv2.Canny(img, 125, 175)
dialated = cv2.dilate(canny, (9,9), iterations=4)
eroded = cv2.erode(dialated, (9,9), iterations=4)
# Displaying
cv2.imshow("Unedited", img)
cv2.imshow("Scaled", rescale(img, 0.5))
cv2.imshow('Blank', blank)
cv2.imshow("Gray",gray)
cv2.imshow("Blur", blur)
cv2.imshow("Edges", eroded)
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
