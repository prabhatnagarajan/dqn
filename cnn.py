import tensorflow as tf
import math

def linear_weight_var(shape):
   stdv = 1.0/math.sqrt(shape[0])
   initial = tf.random_uniform(shape, minval = -stdv, maxval = stdv)
   return tf.Variable(initial)

def linear_bias_var(shape, input_size):
   stdv = 1.0/math.sqrt(input_size)
   initial = tf.random_uniform(shape, minval = -stdv, maxval = stdv)
   return tf.Variable(initial)

def conv_weight_var(shape, n_in, kW, kH):
	stdv = 1.0/math.sqrt(kW * kH * n_in)
	initial = tf.random_uniform(shape, minval = -stdv, maxval = stdv)
	return tf.Variable(initial)

def conv_bias_var(shape, n_in, kW, kH):
	stdv = 1.0/math.sqrt(kW * kH * n_in)
	initial = tf.random_uniform(shape, minval = -stdv, maxval = stdv)
	return tf.Variable(initial)

def make_weight_var(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def make_bias_var(shape):
	initial=tf.constant(0.001, shape=shape)
	return tf.Variable(initial)

def print_shapes(tensors):
	for tensor in tensors:
		print tensor.get_shape()

def clip_error(err):
	return tf.select(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)	
#Implements the Convolutional Neural Network
class NatureCNN():

	def __init__(self, learning_rate, momentum, sq_momentum, hist_len, num_legal_actions):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.sq_momentum = sq_momentum
		self.hist_len = hist_len
		#input data
		self.state = tf.placeholder(tf.float32, [None, 84, 84, self.hist_len])
		'''
		We convolve 32 filters of 8x8 with stride 4 with the input image
		and applies a rectifier nonlinearity. Then stores output of first
		hidden layer in conv_layer_1
		'''
		#Convolutional Layer 1 - outputs batches x 21 x 21 x 32
		self.weights_conv1 = conv_weight_var([8, 8, 4, 32], 4, 8, 8)
		self.bias_conv1 = conv_bias_var([32], 4, 8, 8)
		conv_layer_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.state, self.weights_conv1, strides=[1, 4, 4, 1], padding="VALID"), self.bias_conv1))

		#Convolutional Layer 2 - outputs batches x 11 x 11 x 64
		self.weights_conv2 = conv_weight_var([4, 4, 32, 64], 32, 4, 4)
		self.bias_conv2 = conv_bias_var([64], 32, 4, 4)
		conv_layer_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_1, self.weights_conv2, strides=[1, 2, 2, 1], padding="VALID"), self.bias_conv2))

		#Convolutional Layer 3 - outputs batches x 11 x 11 x 64
		self.weights_conv3 = conv_weight_var([3, 3, 64, 64], 64, 3, 3)
		self.bias_conv3 = conv_bias_var([64], 64, 3, 3)
		conv_layer_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_2, self.weights_conv3, strides=[1, 1, 1, 1], padding="VALID"), self.bias_conv3))

		#Final fully connected hidden layer
		conv3_output = tf.reshape(conv_layer_3, [-1, 64 * 7 * 7])
		#not a linear layer (a relu layer), but the weights are made using linear_weight_var (consistent with lua's nn lib)
		self.weights_fc1 = linear_weight_var([64 * 7 * 7, 512])
		self.bias_fc1 = linear_bias_var([512], 64 * 7 * 7)
		fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv3_output, self.weights_fc1), self.bias_fc1))

		#Output Layer
		self.weights_output = linear_weight_var([512, num_legal_actions])
		self.bias_output = linear_bias_var([num_legal_actions], 512)

		#Q values
		self.q = tf.nn.bias_add(tf.matmul(fc1, self.weights_output), self.bias_output)

		#target
		self.target = tf.placeholder(tf.float32, shape=[None])
		
		#List of action indices in minimal action set array
		self.actions = tf.placeholder(tf.uint8, shape=[None])

		#Compute Q Values of all 32 states
		batch_Q = tf.reduce_sum(tf.mul(self.q, tf.one_hot(self.actions, num_legal_actions)), reduction_indices=1) 

		self.diff = self.target - batch_Q

		#Loss function
		self.loss = tf.reduce_mean(clip_error(self.diff))	

		#Train with RMS Prop
		#TODO perhaps remove epsilon and allow default
		self.train_agent = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, epsilon=0.01).minimize(self.loss)		
