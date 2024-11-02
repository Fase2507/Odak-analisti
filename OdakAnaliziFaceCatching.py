
import numpy as np
import time
import cv2

# Stack function
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.hstack(imgArray[x]) for x in range(rows)]
        ver = np.vstack(hor)
    else:
        hor = np.hstack([cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR) if len(imgArray[x].shape) == 2 else imgArray[x] for x in range(rows)])
        ver = hor
    return ver

def empty(a):
    pass

# Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
global imgEye, imgEyeHSV

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 360)
cap.set(4, 480)
cap.set(12, 250)

# Trackbars for HSV values
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)

cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val min", "Trackbars", 36, 255, empty)
cv2.createTrackbar("Val max", "Trackbars", 255, 255, empty)

focus_loss_start_time = None
total_elapsed_time = 0  # Toplam süre

while True:
    success, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Trackbar values
    h_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    for (x, y, w, h) in Faces:
        imgEye = img[y:y + h, x:x + w]
        imgEyeHSV = cv2.inRange(imgEye, lower, upper)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    white_area = cv2.countNonZero(imgEyeHSV)
    total_area = imgEyeHSV.shape[0] * imgEyeHSV.shape[1]
    white_ratio = white_area / total_area

    # Eğer beyaz oranı %95'ten fazlaysa süreyi başlat
    if white_ratio > 0.95:
        if focus_loss_start_time is None:
            focus_loss_start_time = time.time()
        else:
            elapsed_time = time.time() - focus_loss_start_time
            if elapsed_time >= 5:  # 5 saniye geçtiğinde toplam süreyi güncelle
                total_elapsed_time += elapsed_time
                focus_loss_start_time = time.time()  # Başlangıç zamanını güncelle
    else:
        focus_loss_start_time = None  # Beyaz alan oranı %95'in altına düştüğünde başlangıç zamanını sıfırla

    cv2.putText(img, f"Odak kaybi: {int(total_elapsed_time)} sn", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    imgStack = stackImages(0.8, [[img, imgResult, mask], [imgEye, imgEyeHSV, img]])
    cv2.imshow("ALL", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
