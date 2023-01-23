from __future__ import print_function

import time
import sys
import glob
from PIL import Image
import numpy as np
import scipy.io as sio
import os
from data_preprocess import *
import h5py
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from scipy.optimize import fmin_l_bfgs_b

height = 512
width = 512
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
path_VGG19 = ""

def prepare_data(content_path, style_path):
    content_ = imread(content_path,height,width)
    style_ = imread(style_path,height,width)
    content_array = expand_dimensions(content_)
    style_array = expand_dimensions(style_)
    content_image = backend.variable(content_array)
    style_image = backend.variable(style_array)
    combination_image = backend.placeholder((1, height, width, 3))
    input_tensor = backend.concatenate([content_image,
                                    style_image,
                                    combination_image], axis = 0)
    return input_tensor,combination_image

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

def eval_loss_and_grads(x,f_outputs):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values



def conv_relu(prev_layer, n_layer, layer_name):

    VGG19 = sio.loadmat(path_VGG19)
    VGG19_layers = VGG19['layers'][0]
    # get weights for this layer:
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    # create a conv2d layer
    conv2d = tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b    
    # add a ReLU function and return
    return tf.nn.relu(conv2d)

def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def build_(img_style,img_content):
    

    # Setup network
    with tf.compat.v1.Session() as sess:
        a, h, w, d     = img_content.shape
        net = {}
        net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
        net['conv1_1']  = conv_relu(net['input'], 0, 'conv1_1')
        net['conv1_2']  = conv_relu(net['conv1_1'], 2, 'conv1_2')
        net['avgpool1'] = _avgpool(net['conv1_2'])
        net['conv2_1']  = conv_relu(net['avgpool1'], 5, 'conv2_1')
        net['conv2_2']  = conv_relu(net['conv2_1'], 7, 'conv2_2')
        net['avgpool2'] = _avgpool(net['conv2_2'])
        net['conv3_1']  = conv_relu(net['avgpool2'], 10, 'conv3_1')
        net['conv3_2']  = conv_relu(net['conv3_1'], 12, 'conv3_2')
        net['conv3_3']  = conv_relu(net['conv3_2'], 14, 'conv3_3')
        net['conv3_4']  = conv_relu(net['conv3_3'], 16, 'conv3_4')
        net['avgpool3'] = _avgpool(net['conv3_4'])
        net['conv4_1']  = conv_relu(net['avgpool3'], 19, 'conv4_1')
        net['conv4_2']  = conv_relu(net['conv4_1'], 21, 'conv4_2')     
        net['conv4_3']  = conv_relu(net['conv4_2'], 23, 'conv4_3')
        net['conv4_4']  = conv_relu(net['conv4_3'], 25, 'conv4_4')
        net['avgpool4'] = _avgpool(net['conv4_4'])
        net['conv5_1']  = conv_relu(net['avgpool4'], 28, 'conv5_1')
        net['conv5_2']  = conv_relu(net['conv5_1'], 30, 'conv5_2')
        net['conv5_3']  = conv_relu(net['conv5_2'], 32, 'conv5_3')
        net['conv5_4']  = conv_relu(net['conv5_3'], 34, 'conv5_4')
        net['avgpool5'] = _avgpool(net['conv5_4'])

 
class Calc_loss(object):
    '''
    To calulate the loss and gradients 
    '''

    def __init__(self,f_output):
        self.loss_value = None
        self.grads_values = None
        self.f_output = f_output

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x,self.f_output)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def build_network(content_path,style_path,count):

    input_tensor,combination_image = prepare_data(content_path,style_path)
    model = VGG16(input_tensor=input_tensor, weights='imagenet',
              include_top=False)
    model_layers = dict([(layer.name, layer.output) for layer in model.layers])
    loss = backend.variable(0.)
    features = model_layers['block2_conv2']
    content_image_features = features[0, :, :, :]
    combination_features = features[2, :, :, :]

    loss = loss+ content_weight * content_loss(content_image_features,
                                      combination_features)

    feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

    for layer_name in feature_layers:
        features = model_layers[layer_name]
        style_features = features[1, :, :, :]
        combination_features = features[2, :, :, :]
        sl = style_loss(style_features, combination_features)
        loss = loss +  (style_weight / len(feature_layers)) * sl
    
    loss = loss +  total_variation_weight * total_variation_loss(combination_image)

    grads = backend.gradients(loss, combination_image)

    outputs = [loss]
    outputs += grads
    f_outputs = backend.function([combination_image], outputs)
    loss_eval = Calc_loss(f_outputs)

    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    # To set number of iterations
    num_iterations = 1

    for i in range(num_iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(loss_eval.loss, x.flatten(),
                                        fprime=loss_eval.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    adjust_value0 = 103.939
    adjust_value1 = 116.779
    adjust_value2 = 123.68
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += adjust_value0
    x[:, :, 1] += adjust_value1
    x[:, :, 2] += adjust_value2
    x = np.clip(x, 0, 255).astype('uint8')
    #Give path for the output folder accordingly
    output = os.path.join('output/',str(count)+'.jpg')
    imageio.imsave(output, x)

if __name__ == "__main__":
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    print(content_path, style_path)
    count  = 1
    for content_image,style_image in zip(glob.iglob(f'{content_path}/*'),glob.iglob(f'{style_path}/*')):
        build_network(content_image, style_image,count)
        count+=1
            


