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
# Transforming Image
def transform(frame, x, y):
    transmat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(frame, transmat, (frame.shape[1], frame.shape[0]))
# Rotating Image
def rotate(frame, angle, point=None):
    (height, width) = frame.shape[:2]
    if point == None:
        point = (width//2, height//2)
    rotmat = cv2.getRotationMatrix2D(point, angle, 1.0)
    return cv2.warpAffine(frame, rotmat, (width, height))
# Flipping: 0 = Vertical, 1 = Horizontal, -1 = Both
flip = cv2.flip(img, -1)
# Putting Stuff on Blank Canvas
blank[0:250, 0:250] = 0,0,255
cv2.rectangle(blank, (0,250),(250,500),(255,0,0), thickness=10)
cv2.rectangle(blank, (0,250),(250,500),(0,255,0), thickness=cv2.FILLED)
cv2.circle(blank, (blank.shape[1]//4, blank.shape[0]//4), int(blank.shape[1]//4), (255,0,0), thickness=10)
cv2.line(blank, (250,0),(250, 500),(255,0,0), thickness=10)
cv2.putText(blank, "HI", (250,500),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=5.0, color=(255,255,255), thickness=5)
# Photo Adjustments + Color space conversion
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.GaussianBlur(img, (9,9), cv2.BORDER_DEFAULT)
# Array slicing: img[yvalue_1:yvalue_2,xvalue_1:xvalue_2]
cropped = img[500:800, 500:800]
# Edge Shenanigans
canny = cv2.Canny(img, 125, 175)
dialated = cv2.dilate(canny, (9,9), iterations=4)
eroded = cv2.erode(dialated, (9,9), iterations=4)
# Contour list is a list of all of the edge coordinates; heirarchies are all of the shapes within one another
# RETR_LIST lists all edges, RETR_TREE lists only the heirarchy edges, RETR_EXTERNAL only lists external
# CHAIN_APPRROX_NONE lists all points, CHAIN_APPROX_SIMPLE lists only the end points
blank_ = np.zeros(img.shape, dtype='uint8')
contours, heirarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank_, contours, -1, (0, 0, 255), 1) # -1 tells OpenCV to draw all contours
# Thresholding
ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
# Displaying
cv2.imshow("Unedited", img)
cv2.imshow("Scaled", rescale(img, 0.5))
cv2.imshow('Blank', blank)
cv2.imshow("Gray",gray)
cv2.imshow("Blur", blur)
cv2.imshow("Cropped", cropped)
cv2.imshow("Shifted IMG", transform(img, 50, 50))
cv2.imshow("Rotated IMG", rotate(img, 25))
cv2.imshow("Flipped", flip)
cv2.imshow("Edges", eroded) # White Outline
cv2.imshow("Contours", blank_) # Red Outline
cv2.imshow("Threshold", thresh) # Horizon
cv2.imshow("HSV", HSV)
cv2.imshow("LAB", LAB)
cv2.imshow("RGB", RGB) # OpenCV reads BGR so this will be inverted
print(len(contours))
cv2.waitKey(0)
