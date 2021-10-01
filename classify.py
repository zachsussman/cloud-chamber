import cv2
import MySQLdb as mdb

connection = mdb.connect("localhost", "cloud", "cloud", "test")

cap = cv2.VideoCapture("/Users/sussmanz/cloud_test.mp4")
cap.open("/Users/sussmanz/cloud_test.mp4")

results = []


with connection:
    cursor = connection.cursor()
    cursor.execute("select * from events;")
    while True:
        eid, angle, length, intensity, timestamp, etype, lifespan, divergence = cursor.fetchone()
        cap.set(cv2.CAP_PROP_POS_FRAMES, timestamp + 3)

        frame = cap.read()[1]
        frame = cv2.resize(frame, (400,300))

        cv2.imshow('frame', frame)

        etype = cv2.waitKey(0) & 0xFF

        if etype == "q":
            break
        
        results += [(eid, angle, length, intensity, timestamp, etype, lifespan, divergence)]
for r in results:
    cursor.execute("insert into classified values " + str(r) + ";")

cap.release()
cv2.destroyAllWindows()

