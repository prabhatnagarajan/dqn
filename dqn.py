#!/usr/bin/env python
from random import randrange
from random import uniform
class DQN:
	def __init__(self, ale, capacity, minibatch_size, epsilon):
		self.ale = ale
		self.capacity = capacity
		self.legal_actions = ale.getLegalActionSet()
		self.epsilon = epsilon

	def get_action(self, state):
		rand = uniform(0,1)
		if (rand < self.epsilon):
			return self.legal_actions[randrange(len(self.legal_actions))]
		else:
			#TODO: select max action
			return self.legal_actions[randrange(len(self.legal_actions))]


