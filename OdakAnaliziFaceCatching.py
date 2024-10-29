import cv2

faceCascade= cv2.CascadeClassifier("ImgSrc/haarcascade_frontalface_default.xml")
#img= cv2.imread('ImgSrc/Ronaldo.jpg')
#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(0)
cap.set(3,360)
cap.set(4,480)
cap.set(12,250)


while True:
    success, img = cap.read()
    cv2.imshow("Video",img)
    faces = faceCascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break






cv2.waitKey(0)