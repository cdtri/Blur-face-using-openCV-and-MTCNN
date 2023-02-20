import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np


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


detector = MTCNN()
video = cv2.VideoCapture("videos/video_test.mp4")

if (video.isOpened() == False):
    print("Error reading video file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('output/video_output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'), 29, size)
frame_num = 0
while (True):
    ret, frame = video.read()
    frame_num += 1
    print(frame_num)
    if ret == True:

        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                roi = frame[y:y2, x:x2]
                roi = pixelate_image(roi, grid_size=9)
                frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


video.release()
result.release()

cv2.destroyAllWindows()

print("The video was successfully saved")
