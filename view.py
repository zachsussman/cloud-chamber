import cv2

cap = cv2.VideoCapture("/Users/sussmanz/cloud_test.mp4")
cap.open("/Users/sussmanz/cloud_test.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 19130)

frame = cap.read()[1]
frame = cv2.resize(frame, (400,300))


cv2.imshow('frame', frame)

while cv2.waitKey(1) & 0xFF != ord('q'):
    pass

cap.release()
cv2.destroyAllWindows()
