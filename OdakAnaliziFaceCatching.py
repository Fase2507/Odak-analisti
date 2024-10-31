import cv2
import numpy as np
###STACK###
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

###########
def empty(a):
    pass
#CASCADES

face_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3,360)
cap.set(4,480)
cap.set(12,250)

####TrackBar#####
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)

cv2.createTrackbar("Hue min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue max","Trackbars",179,179,empty)
cv2.createTrackbar("Sat min","Trackbars",0,255,empty)
cv2.createTrackbar("Sat max","Trackbars",255,255,empty)
cv2.createTrackbar("Val min","Trackbars",0,255,empty)
cv2.createTrackbar("Val max","Trackbars",255,255,empty)
###

while True:
    success, img = cap.read()
    ###HUE-SAT-VAL####
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat max", "Trackbars")
    v_max = cv2.getTrackbarPos("Val max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val min", "Trackbars")
    # print(h_min,h_max,s_min,v_min,v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    imgResult = cv2.bitwise_and(img, img, mask=mask)
    ##################
    faces = faceCascade.detectMultiScale(img, 1.2, 4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    imgCropped = img[50:350, 80:180]

    for (x, y, w, h) in Faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #cv2.imshow("Result", img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    imgStack = stackImages(0.8, ([img, imgResult, mask,imgCropped]))

    # cv2.imshow("original", img)
    # cv2.imshow("result", imgResult)
    # cv2.imshow("Msk", mask)
    cv2.imshow("ALL", imgStack)

cap.release()
print(h_min,h_max,s_min,v_min,v_max)
#0 61 0 75 93
#0 179 0 94 145!! onemli eşik degeri.
#0 179 0 75 145!! Negatif eşik odak yok!!

cv2.waitKey(0)




