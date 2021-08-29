import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
from fast_helper import *
import pathlib
hello = tf.constant('Hello, TensorFlow!')
print(hello)
data_dir = pathlib.Path(tf.keras.utils.get_file(origin='https://teststyletransfer.s3.ap-northeast-1.amazonaws.com/media/picasso2-1000x500_XPfehOO.jpg'))
img = list(data_dir.glob('*/*.jpg'))
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# plt.imshow(img[0].numpy().astype('uint8'))
PIL.Image.open(str(img))