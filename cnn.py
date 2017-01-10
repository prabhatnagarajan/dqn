import tensorflow as tf

def make_weight_var(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def make_bias_var(shape):
	initial=tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#Implements the Convolutional Neural Network
class CNN():

	def __init__(self, learning_rate, momentum, sq_momentum, hist_len):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.sq_momentum = sq_momentum
		self.input = tf.placeholder(tf.float32, shape=(1024, 1024))
		self.hist_len = hist_len
		#input data
		self.state = tf.placeholder(tf.float32, [None, 84, 84, self.hist_len])
		'''
		We convolve 32 filters of 8x8 with stride 4 with the input image
		and applies a rectifier nonlinearity. Then stores output of first
		hidden layer in conv_layer_1
		'''
		#Convolutional Layer 1
		self.weights_conv1 = make_weight_var([8, 8, 4, 32])
		self.bias_conv1 = make_bias_var([32])
		conv_layer_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.state, self.weights_conv1, strides=[1, 4, 4, 1], padding="SAME"), self.bias_conv1))

		#Convolutional Layer 2
		self.weights_conv2 = make_weight_var([4, 4, 32, 64])
		self.bias_conv2 = make_bias_var([64])
		conv_layer_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_1, self.weights_conv2, strides=[1, 2, 2, 1], padding="SAME"), self.bias_conv2))

		#Convolutional Layer 3
		self.weights_conv3 = make_weight_var([3, 3, 64, 64])
		self.bias_conv3 = make_bias_var([64])
		conv_layer_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_2, self.weights_conv3, strides=[1, 1, 1, 1], padding="SAME"), self.bias_conv3))

		#Final fully connected hidden layer
		self.weights_layer4 = make_weight_var([512])
		self.bias_layer4 = make_bias_var([512])
		layer_4 = tf.nn.relu(tf.nn.bias_add(tf.multiply(conv_layer_3, weights_layer4), self.bias_layer4))

		#Output Layer
		#TODO change 18 to be "number of valid actions"
		#18 is most number legal actions
		self.weights_output = make_weight_var([18])
		self.bias_output = make_bias_var([18])
		output = tf.nn.bias_add(tf.multiply(layer_4, self.weights_output), self.bias_output)

	def train(self, state):
		print "do nothing"
		#self.train_step = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum).minimize(self.loss_func)
