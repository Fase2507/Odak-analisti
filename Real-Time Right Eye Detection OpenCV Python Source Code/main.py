import cv2
import numpy as np
def empty(a):
    pass
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)

cv2.createTrackbar("Hue min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue max","Trackbars",179,179,empty)
cv2.createTrackbar("Sat min","Trackbars",0,255,empty)
cv2.createTrackbar("Sat max","Trackbars",255,255,empty)
cv2.createTrackbar("Val min","Trackbars",0,255,empty)
cv2.createTrackbar("Val max","Trackbars",255,255,empty)

while True:
    _, img = cap.read()
    #burası_eklenti.
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min=cv2.getTrackbarPos("Hue min","Trackbars")
    h_max=cv2.getTrackbarPos("Hue max","Trackbars")
    s_min=cv2.getTrackbarPos("Sat min","Trackbars")
    s_max=cv2.getTrackbarPos("Sat max","Trackbars")
    v_max=cv2.getTrackbarPos("Val max","Trackbars")
    v_min=cv2.getTrackbarPos("Val min","Trackbars")
    print(h_min,h_max,s_min,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)

    imgResult=cv2.bitwise_and(img,img,mask=mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    cv2.imshow("original",img)
    cv2.imshow("result",imgResult)
    cv2.imshow("Msk",mask)
    cv2.waitKey(1)

# Load the cascade
#face_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
# To capture video from webcam.
#cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
#while True:
    # Read the frame
    #_, img = cap.read()
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Detect the faces
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    #for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Display
    #cv2.imshow('img', img)
    # Stop if escape key is pressed
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
        #break
# Release the VideoCapture object
cap.release()