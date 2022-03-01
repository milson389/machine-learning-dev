import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL
import time
import functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



def image_to_tensor(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)

    img = tf.image.resize(img, [512, 512])
    img = img[tf.newaxis, :]
    return img



kucing = image_to_tensor('datasets/cat.jpg')
pohon = image_to_tensor('datasets/tree.jpg')

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(kucing), tf.constant(pohon))[0]
plt.imshow(tensor_to_image(stylized_image))
plt.show()