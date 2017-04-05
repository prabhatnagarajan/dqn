import sys
from dqn import DQN
from ale_python_interface import ALEInterface
from collections import deque
import preprocess as pp
import numpy as np
import tensorflow as tf
import os
from constants import *

def test(session, hist_len=4, discount=0.99, act_rpt=4, upd_freq=4, min_sq_grad=0.01, epsilon=TEST_EPSILON, 
    no_op_max=30, num_tests=30, learning_rate=0.00025, momentum=0.95, sq_momentum=0.95):
    #Create ALE object
    if len(sys.argv) < 2:
      print 'Usage:', sys.argv[0], 'rom_file'
      sys.exit()

    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)
    #Changes repeat action probability from default of 0.25
    ale.setFloat('repeat_action_probability', 0.0)
    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    USE_SDL = False
    if USE_SDL:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        ale.setBool('sound', True)
      ale.setBool('display_screen', True)

    # Load the ROM file
    ale.loadROM(sys.argv[1])

    # create DQN agent
    # learning_rate and momentum are unused parameters (but needed)
    agent = DQN(ale, session, epsilon, learning_rate, momentum, sq_momentum, hist_len, len(ale.getMinimalActionSet()), None, discount, rom_name(sys.argv[1]))
    
    #Store the most recent two images
    preprocess_stack = deque([], 2)

    num_episodes = 0
    while num_episodes < num_tests:
        #initialize sequence with initial image
        seq = list()
        perform_no_ops(ale, no_op_max, preprocess_stack, seq)
        total_reward = 0
        while not ale.game_over():
            state = get_state(seq, hist_len)
            action = agent.get_action_best_network(state, epsilon)
            #skip frames by repeating action
            reward = 0
            for i in range(act_rpt):
                reward = reward + ale.act(action)
                preprocess_stack.append(ale.getScreenRGB())
            seq.append(pp.preprocess(preprocess_stack[0], preprocess_stack[1]))
            total_reward += reward
        print('Episode ended with score: %d' % (total_reward))
        num_episodes = num_episodes + 1
        ale.reset_game()

def get_state(seq, hist_len):
    if len(seq) < hist_len + 1:
        num_copy = hist_len - len(seq)
        state = ([seq[0]] * num_copy) + (seq)
    else:
        state = seq[-hist_len:]
    return np.moveaxis(state, 0, -1)

def rom_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def perform_no_ops(ale, no_op_max, preprocess_stack, seq):
    #perform nullops
    for _ in range(np.random.randint(no_op_max + 1)):
        ale.act(0)
    #fill the preprocessing stack
    ale.act(0)
    preprocess_stack.append(ale.getScreenRGB())
    ale.act(0)
    preprocess_stack.append(ale.getScreenRGB())
    seq.append(pp.preprocess(preprocess_stack[0], preprocess_stack[0]))

if __name__ == '__main__':
    with tf.Session() as session:
        test(session)
