#!/usr/bin/env python
from random import randrange
class DQN:
	def __init__(self, ale, capacity, minibatch_size):
		self.ale = ale
		self.capacity = capacity
		self.legal_actions = ale.getLegalActionSet()

	def get_action(self, state):
		a = self.legal_actions[randrange(len(self.legal_actions))]
		return a



