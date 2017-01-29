from datetime import datetime
import math
import time

from PIL import Image
import scipy.misc
import numpy as np
import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/projects/cifar10/temp/cifar10_train',
                           """Directory where to read model checkpoints.""")

ImageH = 24
ImageW = 24

def  evaluate_images(images):
  logit = cifar10.inference(images)
  load_trained_model(logit)

def load_trained_model(logits):
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('global step: %s', global_step)
    else:
      print('No checkpoint file found')
      return

    predict = tf.argmax(logits,1)
    print(predict.eval(), '\n')

def img_read(filename, images = None):
  if not tf.gfile.Exists(filename):
    tf.logging.fatal('File does not exists %s', filename)
  ### Seem this is not working, I don't know why. Use scipy to read file instead
  #image_data = tf.gfile.FastGFile(filename, 'rb').read()
  image_data = scipy.misc.imread(filename)
  tfimage = tf.cast(image_data, tf.float32)
  ### resize image to fit the trained model
  resized_image = tf.image.resize_image_with_crop_or_pad(tfimage, ImageH, ImageW)
  ### trained model expected a batch instead of one image
  resized_image = tf.expand_dims(resized_image, 0)

  if images == None:
    images = resized_image
  else:
    images = tf.concat(0, [images, resized_image])

  #tfimage = tf.Session().run(resized_image).astype(np.uint8)
  #Image.fromarray(tfimage).show()
  return images

def img_show(images):
  with tf.Session() as sess:
    img_num = sess.run(tf.shape(images))[0]
    for i in range(img_num):
      tfimage = tf.slice(images, [i, 0, 0, 0], [1, ImageH, ImageW, 3])
      tfimage = tf.reshape(tfimage, [ImageH, ImageW, 3])
      image_data = sess.run(tfimage).astype(np.uint8)
      Image.fromarray(image_data).show()


FLAGS.batch_size = 3
filename = 'airplane1.png'
images = img_read(filename)
filename = 'airplane2.png'
images = img_read(filename, images)
filename = 'dog4.png'
images = img_read(filename, images)

img_show(images)
evaluate_images(images)