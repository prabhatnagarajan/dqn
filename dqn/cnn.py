import tensorflow as tf
import math
from pdb import set_trace

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
	return tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)	
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
		batch_Q = tf.reduce_sum(tf.multiply(self.q, tf.one_hot(self.actions, num_legal_actions)), reduction_indices=1) 

		self.diff = self.target - batch_Q

		#Loss function
		self.loss = tf.reduce_mean(clip_error(self.diff))		

		# Deepmind RMSProp
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)	

		self.g_val_weights_conv1 = tf.Variable(tf.zeros(self.weights_conv1.get_shape()))
		self.g_val_bias_conv1 = tf.Variable(tf.zeros(self.bias_conv1.get_shape()))
		self.g_val_weights_conv2 = tf.Variable(tf.zeros(self.weights_conv2.get_shape()))
		self.g_val_bias_conv2 = tf.Variable(tf.zeros(self.bias_conv2.get_shape()))
		self.g_val_weights_conv3 = tf.Variable(tf.zeros(self.weights_conv3.get_shape()))
		self.g_val_bias_conv3 = tf.Variable(tf.zeros(self.bias_conv3.get_shape()))
		self.g_val_weights_fc1 = tf.Variable(tf.zeros(self.weights_fc1.get_shape()))
		self.g_val_bias_fc1 = tf.Variable(tf.zeros(self.bias_fc1.get_shape()))
		self.g_val_weights_output = tf.Variable(tf.zeros(self.weights_output.get_shape()))
		self.g_val_bias_output = tf.Variable(tf.zeros(self.bias_output.get_shape()))

		self.g2_val_weights_conv1 = tf.Variable(tf.zeros(self.weights_conv1.get_shape()))
		self.g2_val_bias_conv1 = tf.Variable(tf.zeros(self.bias_conv1.get_shape()))
		self.g2_val_weights_conv2 = tf.Variable(tf.zeros(self.weights_conv2.get_shape()))
		self.g2_val_bias_conv2 = tf.Variable(tf.zeros(self.bias_conv2.get_shape()))
		self.g2_val_weights_conv3 = tf.Variable(tf.zeros(self.weights_conv3.get_shape()))
		self.g2_val_bias_conv3 = tf.Variable(tf.zeros(self.bias_conv3.get_shape()))
		self.g2_val_weights_fc1 = tf.Variable(tf.zeros(self.weights_fc1.get_shape()))
		self.g2_val_bias_fc1 = tf.Variable(tf.zeros(self.bias_fc1.get_shape()))
		self.g2_val_weights_output = tf.Variable(tf.zeros(self.weights_output.get_shape()))
		self.g2_val_bias_output = tf.Variable(tf.zeros(self.bias_output.get_shape()))

		#following deepmind's notation
		self.g = [self.g_val_weights_conv1, self.g_val_bias_conv1, self.g_val_weights_conv2, self.g_val_bias_conv2,
					self.g_val_weights_conv3, self.g_val_bias_conv3, self.g_val_weights_fc1, self.g_val_bias_fc1, 
					self.g_val_weights_output, self.g_val_bias_output]
		self.g2 = [self.g2_val_weights_conv1, self.g2_val_bias_conv1, self.g2_val_weights_conv2, self.g2_val_bias_conv2,
					self.g2_val_weights_conv3, self.g2_val_bias_conv3, self.g2_val_weights_fc1, self.g2_val_bias_fc1, 
					self.g2_val_weights_output, self.g2_val_bias_output]
		
		#grads_and_vars = optimizer.compute_gradients(self.loss, )
		variables = [self.weights_conv1,
			self.bias_conv1,
			self.weights_conv2,
			self.bias_conv2,
			self.weights_conv3,
			self.bias_conv3,
			self.weights_fc1,
			self.bias_fc1,
			self.weights_output,
			self.bias_output
		]

		grads = tf.gradients(self.loss, variables)
		gradients = []
		for grad in grads:
			gradients.append(grad)

		update_g = [g_val.assign(0.95 * g_val + 0.05 * grad) for g_val, grad in zip(self.g, gradients)]
		
		update_g2 = [g2_val.assign(0.95 * g2_val + 0.05 * tf.square(grad)) for g2_val, grad in zip(self.g2, gradients)]

		rms = [tf.sqrt(g2_val - tf.square(g_val) + 0.01) for g_val, g2_val in zip(self.g, self.g2)]

		rms_update = [gradient/rms_val for gradient, rms_val in zip(gradients, rms)]

		#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/control_flow_ops.py
		g_updates = update_g + update_g2

		training = optimizer.apply_gradients(zip(rms_update, variables))
		self.train_rms_prop = tf.group(training, tf.group(*(g_updates)))