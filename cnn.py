import tensorflow as tf

def make_weight_var(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def make_bias_var(shape):
	initial=tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def clip_error(err):
	return tf.select(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)

#Implements the Convolutional Neural Network
class CNN():

	def __init__(self, learning_rate, momentum, sq_momentum, hist_len, num_legal_actions):
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
		#Convolutional Layer 1 - outputs batches x 21 x 21 x 32
		self.weights_conv1 = make_weight_var([8, 8, 4, 32])
		self.bias_conv1 = make_bias_var([32])
		conv_layer_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.state, self.weights_conv1, strides=[1, 4, 4, 1], padding="SAME"), self.bias_conv1))

		#Convolutional Layer 2 - outputs batches x 11 x 11 x 64
		self.weights_conv2 = make_weight_var([4, 4, 32, 64])
		self.bias_conv2 = make_bias_var([64])
		conv_layer_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_1, self.weights_conv2, strides=[1, 2, 2, 1], padding="SAME"), self.bias_conv2))

		#Convolutional Layer 3 - outputs batches x 11 x 11 x 64
		self.weights_conv3 = make_weight_var([3, 3, 64, 64])
		self.bias_conv3 = make_bias_var([64])
		conv_layer_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_layer_2, self.weights_conv3, strides=[1, 1, 1, 1], padding="SAME"), self.bias_conv3))

		#Final fully connected hidden layer
		conv3_output = tf.reshape(conv_layer_3, [-1, 11 * 11 * 64])
		self.weights_fc1 = make_weight_var([11 * 11 * 64, 512])
		self.bias_fc1 = make_bias_var([512])
		fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv3_output, self.weights_fc1), self.bias_fc1))

		#Output Layer
		self.weights_output = make_weight_var([512, num_legal_actions])
		self.bias_output = make_bias_var([num_legal_actions])

		#Q values
		self.q = tf.nn.bias_add(tf.matmul(fc1, self.weights_output), self.bias_output)

		#target
		self.target = tf.placeholder(tf.float32, shape=[None])
		
		#List of legal actions
		self.actions = tf.placeholder(tf.uint8, shape=[None])

		#Compute Q Values of all 32 states
		batch_Q = tf.reduce_sum(tf.mul(self.q, tf.one_hot(self.actions, num_legal_actions)), reduction_indices=1) 

		self.diff = self.target - batch_Q

		#Loss function
		self.loss = tf.reduce_mean(clip_error(self.diff))	

		#Train with RMS Prop
		#TODO perhaps remove epsilon and allow default
		self.train_agent = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, epsilon=0.01).minimize(self.loss)
