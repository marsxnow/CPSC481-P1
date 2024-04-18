# from deepface import DeepFace
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, time, uuid, json

from tensorflow.python.ops.image_ops_impl import image_gradients

# path where images will be stored
image_path = os.path.join('data', 'images')
#number of images to collect when code commented below is run
number_images = 30

'''
this code is what collects the images from the webcam
remember to give permission to the webcam if you want to add more photos to the dataset
uncomment to collect images
'''
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
'''
------------------------------------------------------------------------------------------
'''

# query system for available GPUs that are compatible with tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # set memory growth to true to allow the GPU to allocate memory as needed
    tf.config.experimental.set_memory_growth(gpu, True)

# confirm that the configuration has been set
tf.config.list_physical_devices('GPU')

# load the images from the directory
images =  tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)


def load_image(path):
    '''
    function to load image from path and return the image
    '''
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img

# transform the images to tensors
images = images.map(load_image)

images.as_numpy_iterator().next()
# print(type(images))

image_generator = images.batch(4).as_numpy_iterator()
# plot the images to confirm that they have been loaded
plot_images = image_generator.next()
fig, ax =plt.subplots(ncols= 4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)

plt.show()
