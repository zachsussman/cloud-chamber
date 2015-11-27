import cv2
import numpy as np
from skeleton import skeletonize
from contours import join_contours

GREEN = (0, 255, 0)

cap = cv2.VideoCapture("/Users/sussmanz/cloud_test.mp4")
cap.open("/Users/sussmanz/cloud_test.mp4")

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def getLineFromContour(contour):
    center, sides, angle = cv2.minAreaRect(contour)
    x, y = center
    l, m = sides
    angle = np.radians(angle)
    if l > m:
        a = (int(x - l*np.cos(angle)), int(y - m*np.sin(angle)))
        b = (int(x + l*np.cos(angle)), int(y + m*np.sin(angle)))
    else:
        a = (int(x - m*np.sin(angle)), int(y + m*np.cos(angle)))
        b = (int(x + m*np.sin(angle)), int(y - m*np.cos(angle)))
    return a, b

def detectTracks(img):
    points = cv2.findNonZero(img)

    i, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 1:
        contours = join_contours(contours)

    
    
    return map(getLineFromContour, contours)
##    return contours


fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setVarThreshold(30)


while(cap.isOpened()):
    ret, frame = cap.read()

    gray = grayscale(frame)
    mask = fgbg.apply(gray)
    mask = cv2.blur(mask, (15, 15))
##    mask = skeletonize(mask)
##    mask = ~mask
    ret, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, (4,4))
    

##    mask = cv2.Canny(mask, 75, 125)

##    lines = cv2.HoughLinesP(mask, 1, 3.14/180, 100, 10, 10)
    
##    rects = detectTracks(mask)
    lines = detectTracks(mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if lines != None:
        for l in lines:
            cv2.line(mask, l[0], l[1], GREEN, 10)


##    if rects != None:
##        for l in rects:
##            cv2.line(mask, (l[0],l[1]), (l[0],l[3]), (0,255,0), 10)
##            cv2.line(mask, (l[0],l[3]), (l[2],l[3]), (0,255,0), 10)
##            cv2.line(mask, (l[2],l[1]), (l[0],l[1]), (0,255,0), 10)
##            cv2.line(mask, (l[2],l[3]), (l[2],l[1]), (0,255,0), 10)

##    cv2.line(mask, (0,40), (300,400), (0,255,0))

    gray = cv2.resize(gray, (400, 300))
    mask = cv2.resize(mask, (400,300))

    
    cv2.imshow('original', gray)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
