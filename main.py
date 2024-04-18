from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1 = "data/yoshirandy.jpg"
img2 = "data/randy.jpg"

img1 = cv2.imread(img1)
img2 = cv2.imread(img2)

plt.imshow(img1[:, :, ::-1])
# plt.show()
plt.imshow(img2[:, :, ::-1])
# plt.show()


result = DeepFace.verify(img1, img2)
print(result)
