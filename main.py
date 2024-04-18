# from deepface import DeepFace
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, time, uuid, json

from tensorflow.python.ops.image_ops_impl import image_gradients

image_path = os.path.join('data', 'images')
number_images = 30

# cap = cv2.VideoCapture(0)
# for imgnum in range(number_images):
#     print('Collection Image {}'.format(imgnum))
#     ret, frame = cap.read()
#     imgname = os.path.join(image_path,f'{str(uuid.uuid4())}.jpg')
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

images =  tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)

def load_image(path):
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)

images.as_numpy_iterator().next()
print(type(images))

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()
fig, ax =plt.subplots(ncols= 4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)

plt.show()
