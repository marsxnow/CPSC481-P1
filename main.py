from deepface import DeepFace
import cv2, os, time, uuid
import matplotlib.pyplot as plt

image_path = os.path.join('data', 'images')
number_images = 30

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collection Image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(image_path,f'{str(uuid.uuid4())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
