import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if faceCascade.empty():
    print("Error loading cascade classifier.")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera.")
    exit(1)

while True:
    Dip, img = cap.read()
    if not Dip:
        print("Error reading frame from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Dip Roy", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
