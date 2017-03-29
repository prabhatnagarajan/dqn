#!/usr/bin/env python
import random
from random import randrange
from random import uniform
from cnn import NatureCNN
import os
import tensorflow as tf
import numpy as np
from time import time
from constants import *

class DQN:
	def __init__(self, ale, session, epsilon, learning_rate, momentum, sq_momentum, hist_len, min_num_actions, 
		tgt_update_freq, discount, rom):
		self.ale = ale
		self.session = session
		self.minimal_action_set = ale.getMinimalActionSet().tolist()
		self.epsilon = epsilon
		self.num_updates = 0
		self.discount = discount
		self.tgt_update_freq = tgt_update_freq
		self.chkpt_freq = CHECKPOINT_FREQUENCY
		self.prediction_net = NatureCNN(learning_rate, momentum, sq_momentum, hist_len, min_num_actions)
		self.target_net = NatureCNN(learning_rate, momentum, sq_momentum, hist_len, min_num_actions)
		self.best_net = NatureCNN(learning_rate, momentum, sq_momentum, hist_len, min_num_actions)
		self.rom = rom		
		self.reset_target_network = [
		#Copy Weights
		self.target_net.weights_conv1.assign(self.prediction_net.weights_conv1),
		self.target_net.weights_conv2.assign(self.prediction_net.weights_conv2),
		self.target_net.weights_conv3.assign(self.prediction_net.weights_conv3),
		self.target_net.weights_fc1.assign(self.prediction_net.weights_fc1),
		self.target_net.weights_output.assign(self.prediction_net.weights_output),
		#Copy Bias
		self.target_net.bias_conv1.assign(self.prediction_net.bias_conv1),
		self.target_net.bias_conv2.assign(self.prediction_net.bias_conv2),
		self.target_net.bias_conv3.assign(self.prediction_net.bias_conv3),
		self.target_net.bias_fc1.assign(self.prediction_net.bias_fc1),
		self.target_net.bias_output.assign(self.prediction_net.bias_output)]

		self.update_best_network = [
		#Copy Weights
		self.best_net.weights_conv1.assign(self.prediction_net.weights_conv1),
		self.best_net.weights_conv2.assign(self.prediction_net.weights_conv2),
		self.best_net.weights_conv3.assign(self.prediction_net.weights_conv3),
		self.best_net.weights_fc1.assign(self.prediction_net.weights_fc1),
		self.best_net.weights_output.assign(self.prediction_net.weights_output),
		#Copy Bias
		self.best_net.bias_conv1.assign(self.prediction_net.bias_conv1),
		self.best_net.bias_conv2.assign(self.prediction_net.bias_conv2),
		self.best_net.bias_conv3.assign(self.prediction_net.bias_conv3),
		self.best_net.bias_fc1.assign(self.prediction_net.bias_fc1),
		self.best_net.bias_output.assign(self.prediction_net.bias_output)]
		

		self.checkpoint_directory = CHECKPOINT_DIR + "/" + rom

		self.session.run(tf.global_variables_initializer())

		#Copy the target network to begin with
		self.copy_network()

		self.counter = 0
		var_list = 	[
			#weights prediction
			self.prediction_net.weights_conv1,
			self.prediction_net.weights_conv2,
			self.prediction_net.weights_conv3,
			self.prediction_net.weights_fc1,
			self.prediction_net.weights_output,
			#bias prediction
			self.prediction_net.bias_conv1,
			self.prediction_net.bias_conv2,
			self.prediction_net.bias_conv3,
			self.prediction_net.bias_fc1,
			self.prediction_net.bias_output,
			#weights target
			self.target_net.weights_conv1,
			self.target_net.weights_conv2,
			self.target_net.weights_conv3,
			self.target_net.weights_fc1,
			self.target_net.weights_output,
			#bias target
			self.target_net.bias_conv1,
			self.target_net.bias_conv2,
			self.target_net.bias_conv3,
			self.target_net.bias_fc1,
			self.target_net.bias_output
			]
		#add rms prop variables
		var_list.extend(self.prediction_net.g)
		var_list.extend(self.prediction_net.g2)
		self.saver = tf.train.Saver(var_list, max_to_keep=3)

		checkpoint = tf.train.get_checkpoint_state(self.checkpoint_directory)
		if checkpoint and checkpoint.model_checkpoint_path:
			print "Loading the saved weights..."
			print "Latest Checkpoint is ..."
			print tf.train.latest_checkpoint(self.checkpoint_directory)
			#self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			self.saver.restore(self.session, tf.train.latest_checkpoint(self.checkpoint_directory))
			self.counter = int(checkpoint.model_checkpoint_path.split('-')[-1])
			print "Load complete."
		else:
			print "No saved weights. Beginning with random weights."

	def eGreedy_action(self, state, epsilon):
		rand = uniform(0,1)
		if (rand < epsilon):
			return self.minimal_action_set[randrange(len(self.minimal_action_set))]
		else:
			#Choose greedy action
			mod_state = np.array([state], dtype=np.float32)
			q_vals = self.prediction_net.q.eval(
	        		feed_dict = {self.prediction_net.state: mod_state})[0]
			return self.minimal_action_set[np.argmax(q_vals)]

	def get_action(self, state):
		return self.eGreedy_action(state, self.epsilon)

	def set_epsilon(self, epsilon):
		self.epsilon = epsilon

	def compute_labels(self, sample, minibatch_size):
	    label = np.zeros(minibatch_size)
	    for i in range(minibatch_size):
	        if (sample[i].game_over):
	            label[i] = sample[i].reward
	        else:
	        	q_vals = self.target_net.q.eval(
	        		feed_dict = {self.target_net.state:[sample[i].new_state]})
	    		label[i] = sample[i].reward + self.discount * q_vals.max()
	    return label

	def copy_network(self):
		self.session.run(self.reset_target_network)
	
	def sample_minibatch(self, replay_memory, minibatch_size):
		return random.sample(replay_memory, minibatch_size)

	def train(self, replay_memory, minibatch_size):
	    #sample a minibatch of transitions
	    tf.get_default_graph().finalize()
	    sample = self.sample_minibatch(replay_memory, minibatch_size)
	    #set label
	    labels = self.compute_labels(sample, minibatch_size)
	    state = [x.state for x in sample]
	    actions_taken = [x.action for x in sample]
	    action_indices = [self.minimal_action_set.index(x) for x in actions_taken]
	    feed_dict={
		    self.prediction_net.state : state,
		    self.prediction_net.actions : action_indices,
		    self.prediction_net.target : labels.tolist()
	    }

	    #Perform the gradient descent step
	    self.prediction_net.train_rms_prop.run(feed_dict=feed_dict)

	    #increment update counter
	    self.num_updates = self.num_updates + 1

	    if self.num_updates % self.chkpt_freq == 0:
	    	print "Saving Weights"
	    	self.saver.save(self.session, os.path.join(self.checkpoint_directory, self.rom), global_step = self.counter + self.num_updates)
	    	print "Saved."
