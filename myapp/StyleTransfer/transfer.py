import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

from keras import backend as K

from imageio import mimsave
from IPython.display import display as display_fn
from IPython.display import Image, clear_output

import helper
from helper import *


content_layers = ['conv2d_88']

style_layers = ['conv2d','conv2d_1','conv2d_2','conv2d_3','conv2d_4']
                
content_and_style_layers = content_layers + style_layers

NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)


def inception_model(layer_names):

  # Load InceptionV3 with the imagenet weights and **without** the fully-connected layer at the top of the network
  inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

  # Freeze the weights of the model's layers (make them not trainable)
  inception.trainable = False
  
  # Create a list of layer objects that are specified by layer_names
  output_layers = [inception.get_layer(name).output for name in layer_names]

  # Create the model that outputs the content and style layers
  model = tf.keras.Model(inputs=inception.input, outputs=output_layers)
    
  # return the model
  return model
inception = inception_model(content_and_style_layers)

def get_style_loss(features, targets):
    
  # Calculate the style loss
  style_loss = tf.reduce_mean(tf.square(features-targets))
    
  return style_loss

def get_content_loss(features, targets):
  # get the sum of the squared error multiplied by a scaling factor
  content_loss = 0.5*tf.reduce_sum(tf.square(features-targets))
    
  return content_loss

def gram_matrix(input_tensor):
  # calculate the gram matrix of the input tensor
  gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 

  # get the height and width of the input tensor
  input_shape = tf.shape(input_tensor) 
  height = input_shape[1] 
  width = input_shape[2] 

  # get the number of locations (height times width), and cast it as a tf.float32
  num_locations = tf.cast(height * width, tf.float32)

  # scale the gram matrix by dividing by the number of locations
  scaled_gram = gram / num_locations
    
  return scaled_gram

def get_style_image_features(image):  
  # preprocess the image using the given preprocessing function
  preprocessed_style_image = preprocess_image(image)

  # get the outputs from the inception model that you created using inception_model()
  outputs = inception(preprocessed_style_image)

  # Get just the style feature layers (exclude the content layer)
  style_outputs = outputs[NUM_CONTENT_LAYERS:]

  # for each style layer, calculate the gram matrix for that layer and store these results in a list
  gram_style_features = [gram_matrix(name) for name in style_outputs]
  
  return gram_style_features

def get_content_image_features(image):
  # preprocess the image
  preprocessed_content_image = preprocess_image(image)
    
  # get the outputs from the inception model
  outputs = inception(preprocessed_content_image)

  # get the content layer of the outputs
  content_outputs = outputs[:NUM_CONTENT_LAYERS]

  return content_outputs

def get_style_content_loss(style_targets, style_outputs, content_targets, 
                           content_outputs, style_weight, content_weight):
    
  # Sum of the style losses
  style_loss = tf.add_n([ get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])
  
  # Sum up the content losses
  content_loss = tf.add_n([get_content_loss(content_output, content_target)
                           for content_output, content_target in zip(content_outputs, content_targets)])

  # scale the style loss by multiplying by the style weight and dividing by the number of style layers
  style_loss = style_weight/ NUM_STYLE_LAYERS * style_loss

  # scale the content loss by multiplying by the content weight and dividing by the number of content layers
  content_loss = content_weight / NUM_CONTENT_LAYERS * content_loss
    
  # sum up the style and content losses
  total_loss =  style_loss +  content_loss
  # return the total loss
  return total_loss

def calculate_gradients(image, style_targets, content_targets, 
                        style_weight, content_weight):

  with tf.GradientTape() as tape:
      
    # get the style image features
    style_features = get_style_image_features(image)
      
    # get the content image features
    content_features = get_content_image_features(image)
      
    # get the style and content loss
    loss = get_style_content_loss(style_targets, style_features, content_targets, 
                           content_features, style_weight, content_weight)

  # calculate gradients of loss with respect to the image
  gradients = tape.gradient(loss, image)

  return gradients

def update_image_with_style(image, style_targets, content_targets, style_weight, 
                            content_weight, optimizer):

  # Calculate gradients using the function that you just defined.
  gradients = calculate_gradients(image, style_targets, content_targets, 
                        style_weight, content_weight)

  # apply the gradients to the given image
  optimizer.apply_gradients([(gradients, image)])

  image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))

def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4, 
                       optimizer='adam', epochs=1, steps_per_epoch=1):
  images = []
  step = 0

  # get the style image features 
  style_targets = get_style_image_features(style_image)
    
  # get the content image features
  content_targets = get_content_image_features(content_image)

  # initialize the generated image for updates
  generated_image = tf.cast(content_image, dtype=tf.float32)
  generated_image = tf.Variable(generated_image) 
  
  # collect the image updates starting from the content image
  images.append(content_image)
  
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
    
      # Update the image with the style using the function that you defined
      update_image_with_style(generated_image, style_targets, content_targets, style_weight, 
                            content_weight, optimizer)
    
      print(".", end='')
      if (m + 1) % 10 == 0:
        images.append(generated_image)
    
    # display the current stylized image
    clear_output(wait=True)
    display_image = tensor_to_image(generated_image)
    print(display_fn(display_image))

    # mimsave('myapp/media/generated.jpg',generated_image)
    # append to the image collection for visualization later
    images.append(generated_image)
    print("Train step: {}".format(step))
  
  # convert to uint8 (expected dtype for images with pixels in the range [0,255])
  generated_image = tf.cast(generated_image, dtype=tf.uint8)
  
  return generated_image, images


# define style and content weight
def main():
    # inception = inception_model(content_and_style_layers)
    tmp_layer_list = [layer.output for layer in inception.layers]
    content_path = 'myapp/media/dog1.jpg'
    style_path = 'myapp/media/style.jpg'
    content_image, style_image = load_images(content_path, style_path)
    
    # show_images_with_objects([content_image, style_image], 
    #                         titles=[f'content image: {content_path}',
    #                                 f'style image: {style_path}'])                    
    style_weight =  1
    content_weight = 1e-32 

    # define optimizer. learning rate decreases per epoch.
    adam = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
        )
    )

    # start the neural style transfer
    stylized_image, display_images = fit_style_transfer(style_image=style_image, content_image=content_image, 
                                                        style_weight=style_weight, content_weight=content_weight,
                                                        optimizer=adam, epochs=1, steps_per_epoch=10)
    
    
    pil_img = tf.keras.preprocessing.image.array_to_img(tf.squeeze(stylized_image))
    print(type(pil_img))
    pil_img.save('myapp/media/generated.jpg')
if __name__ == "__main__":
    main()                                                