import sys
import random
from random import randrange
from dqn import DQN
from ale_python_interface import ALEInterface
from collections import deque
import preprocess as pp
import numpy as np
import cnn
import tensorflow as tf
#TODO Remove unused imports

def test(session, hist_len=4, discount=0.99, act_rpt=4, upd_freq=4, min_sq_grad=0.01, epsilon=0.05, 
    noop_max=30, num_tests=30):
    #Create ALE object
    if len(sys.argv) < 2:
      print 'Usage:', sys.argv[0], 'rom_file'
      sys.exit()

    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)

    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    USE_SDL = True
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
    agent = DQN(ale, session,  1000000, epsilon, None, None, None, hist_len, len(ale.getLegalActionSet()), None, discount)

    num_episodes = 0
    #TODO Change the episode ranges to be a function of frames
    while num_episodes < num_tests:
        img = ale.getScreenRGB()
        #initialize sequence with initial image
        seq = list()
        seq.append(img)
        proc_seq = list()
        proc_seq.append(pp.preprocess(seq))
        total_reward = 0
        state = img
        while not ale.game_over():
            action = agent.get_action(state)
            #skip frames by repeating action
            for i in range(act_rpt):
                reward = reward + ale.act(action)
                if ale.game_over():
                    break
            total_reward += reward
        #print('Episode %d ended with score: %d' % (episode, total_reward))
        print('Episode ended with score: %d' % (total_reward))
        num_episodes = num_episodes + 1
        ale.reset_game()

if __name__ == '__main__':
    with tf.Session() as session:
        test(session)