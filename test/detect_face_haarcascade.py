# plot photo with detected faces using opencv cascade classifier
import cv2
# load the photograph
img = cv2.imread('../images/test1.jpg')
img = cv2.resize(img, (800, 600))
# load the pre-trained model
classifier = cv2.CascadeClassifier(
    '../models/haarcascade_frontalface_default.xml')
# perform face detection
# try test with scaleFactor = 1.05 and minNeighbors = 8, default scaleFactor = 1.1 and minNeighbors = 3
bboxes = classifier.detectMultiScale(img, 1.05, 8)
# print bounding box for each detected face
for box in bboxes:
    # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)
# show the image
cv2.imshow('face detection', img)
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()
