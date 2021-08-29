import os
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
# import fast_helper
from .fast_helper import *
from datetime import datetime

def transfer(content_path, style_path):
    import os
    IMAGE_DIR = 'images'
    os.system(f'mkdir {IMAGE_DIR}')
    os.system(f'wget -q -O ./images/content.jpg {content_path}')
    os.system(f'wget -q -O ./images/style.jpg {style_path}')
    content_path = f'{IMAGE_DIR}/content.jpg'
    style_path = f'{IMAGE_DIR}/style.jpg'
    # print(os.system('ls -l'))
    content_image, style_image = load_images(content_path, style_path)
    # print('asadasd')
    # show_images_with_objects([content_image, style_image], 
    #                         titles=[f'content image: {content_path}',
    #                                 f'style image: {style_path}'])
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = hub_module(tf.image.convert_image_dtype(content_image, tf.float32), 
                                tf.image.convert_image_dtype(style_image, tf.float32))[0]
    # convert the tensor to image
    generated_image = tensor_to_image(stylized_image)
    generated_path = f'myapp/media/generated{datetime.now()}.jpg'
    generated_image.save(generated_path)
    print(generated_path)
    # show_images_with_objects([stylized_image], titles=['generated image'])
    return generated_path
# transfer()