from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class CPPNParameters():
  def __init__(self):
    self.batch_size = 1
    self.layer_size = 128
    self.x_dim = 1024
    self.y_dim = 1024
    self.z_dim = 8
    self.c_dim = 1
    self.scale = 1


class CPPN():
    def __init__(self, params):
        self.params = params

        num_points = self.params.x_dim * self.params.y_dim
        self.z = tf.placeholder(tf.float32,
                [self.params.batch_size, self.params.z_dim])
        self.x = tf.placeholder(tf.float32,
                [self.params.batch_size, num_points, 1])
        self.y = tf.placeholder(tf.float32,
                [self.params.batch_size, num_points, 1])
        self.r = tf.placeholder(tf.float32,
                [self.params.batch_size, num_points, 1])

        self.session = tf.Session()
        self.coordinates = self.get_coordinates()
        self.image_op = self.get_image_op()
        self.init()

    def init(self):
        self.session.run(tf.initialize_all_variables())

    def get_coordinates(self):
        """Generates a set of coordinates for the every pixel in the image.
        """
        num_points = self.params.x_dim * self.params.y_dim
        x_dim = self.params.x_dim
        y_dim = self.params.y_dim

        x_range = (np.arange(x_dim) - (x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = (np.arange(y_dim) - (y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), self.params.batch_size).reshape(
                self.params.batch_size, num_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.params.batch_size).reshape(
                self.params.batch_size, num_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.params.batch_size).reshape(
                self.params.batch_size, num_points, 1)
        return x_mat, y_mat, r_mat

    def get_image_op(self):
        """Constructs the neural network and returns the output image data op.
        """
        input1 = slim.fully_connected(
                self.x,
                self.params.layer_size,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=None)
        input2 = slim.fully_connected(
                self.y,
                self.params.layer_size,
                activation_fn=None, 
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=None)
        input3 = slim.fully_connected(
                self.r,
                self.params.layer_size,
                activation_fn=None, 
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=None)
        input4 = self.params.scale * slim.fully_connected(
                self.z,
                self.params.layer_size,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=tf.random_normal_initializer(stddev=1.0))
        network = tf.tanh(input1 + input2 + input3 + input4)

        network = slim.fully_connected(
                network,
                self.params.layer_size,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=tf.random_normal_initializer(stddev=1.0))
        network = slim.fully_connected(
                network,
                self.params.layer_size,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=tf.random_normal_initializer(stddev=1.0))
        network = slim.fully_connected(
                network,
                self.params.layer_size,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=tf.random_normal_initializer(stddev=1.0))

        output = slim.fully_connected(
                network,
                self.params.c_dim,
                activation_fn=tf.sigmoid,
                weights_initializer=tf.random_normal_initializer(stddev=1.0),
                biases_initializer=tf.random_normal_initializer(stddev=1.0))
       
        image = tf.reshape(output,
                [self.params.batch_size, self.params.y_dim,
                 self.params.x_dim, self.params.c_dim])
        return image 
   
    def get_image(self, z):
        """Creates a tensor of image data from a the neural network.
        """
        x_mat, y_mat, r_mat = self.coordinates
        image = self.session.run(self.image_op, feed_dict = {
                self.x: x_mat,
                self.y: y_mat,
                self.r: r_mat,
                self.z: z})
        return image

    def tensor_to_image(self, tensor):
        """Converts the output of the neural network into a PIL Image.
        """
        image_data = np.array(1 - tensor)

        if self.params.c_dim == 1:
            image_data = np.array(255.0 * image_data.reshape(
                    (self.params.x_dim, self.params.y_dim)),
                    dtype=np.uint8)
        else:
            image_data = np.array(255.0 * image_data.reshape(
                    (self.params.x_dim, self.params.y_dim, self.params.c_dim)),
                    dtype=np.uint8)

        return Image.fromarray(image_data)

    def save_gif(self, filename, num_steps):
        """Saves many generated images to the given filename.

        It chooses two randomly locations in the latent space, and generates
        images by stepping linearly between them.
        """
        z1 = np.random.uniform(-1.0, 1.0, size=(
                self.params.batch_size, self.params.z_dim))
        z2 = np.random.uniform(-1.0, 1.0, size=(
                self.params.batch_size, self.params.z_dim))

        delta = (z2 - z1) / (num_steps + 1)
        images = []
        for i in xrange(num_steps + 2):
            z = z1 + i * delta
            image_data = self.get_image(z)        
            images.append(self.tensor_to_image(image_data))

        images = images + images[1::-1]
        images[0].save(filename, save_all=True, append_images=images[1:])

    def save_image(self, filename):
        """Saves a generated image to the given filename.
        """
        z = np.random.uniform(-1.0, 1.0, size=(
                self.params.batch_size, self.params.z_dim))
        image_data = self.get_image(z)
        self.tensor_to_image(image_data).save(filename)
