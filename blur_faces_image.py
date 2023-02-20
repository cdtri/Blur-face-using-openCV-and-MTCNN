import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2


def pixelate_image(image, grid_size):
    # Chia anh thanh block x block o vuong
    (h, w) = image.shape[:2]
    xGridLines = np.linspace(0, w, grid_size + 1, dtype="int")
    yGridLines = np.linspace(0, h, grid_size + 1, dtype="int")

    # Lap qua tung o vuong
    for i in range(1, len(xGridLines)):
        for j in range(1, len(yGridLines)):

            # Lay toa do cua o vuong hien tai
            cell_startX = xGridLines[j - 1]
            cell_startY = yGridLines[i - 1]
            cell_endX = xGridLines[j]
            cell_endY = yGridLines[i]

            # Trich vung anh theo toa do ben tren
            cell = image[cell_startY:cell_endY, cell_startX:cell_endX]

            # Tinh trung binh cong vung anh va ve vao o vuong hien tai
            (B, G, R) = [int(x) for x in cv2.mean(cell)[:3]]
            cv2.rectangle(image, (cell_startX, cell_startY), (cell_endX, cell_endY),
                          (B, G, R), -1)

    return image


if __name__ == '__main__':
    filename = 'images/test2.jpg'
    # load image from file
    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(img)

    # Create figure and axes
    fig, ax = plt.subplots()

    for face in faces:
        print(face)
        x, y, w, h = face['box']
        roi = img[y:y+h, x:x+w]
        # roi = pixelate_image(roi, grid_size=9)
        roi = cv2.GaussianBlur(roi, (9, 9), 0)
        # impose this blurred image on original image to get final image
        img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # Display the image
    ax.imshow(img)

    plt.show()
