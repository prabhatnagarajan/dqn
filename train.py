import sys
import os
import random
from random import randrange
from dqn import DQN
from ale_python_interface import ALEInterface
from collections import deque
from collections import namedtuple
import preprocess as pp
import numpy as np
import cnn
import tensorflow as tf
from time import time

#SETUP
Experience = namedtuple('Experience', 'state action reward new_state game_over')

def train(session, minibatch_size=32, replay_capacity=1000000, hist_len=4, tgt_update_freq=10000,
    discount=0.99, act_rpt=4, upd_freq=4, learning_rate=0.00025, grad_mom=0.95,
    sgrad_mom=0.95, min_sq_grad=0.01, init_epsilon=1.0, fin_epsilon=0.1, 
    fin_exp=1000000, replay_start_size=50000, noop_max=30, epsilon_file="epsilon.npy", memory_file="memory.npy", train_save_frequency=1000):
    #Create ALE object
    if len(sys.argv) < 2:
      print 'Usage:', sys.argv[0], 'rom_file'
      sys.exit()

    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)
    #Changes repeat action probability from default of 0.25
    ale.setFloat('repeat_action_probability', 0.0)
    #Automates Frame Skipping - changes from default value of 1
    ale.setInt('frame_skip', act_rpt)

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

    #initialize epsilon
    epsilon = init_epsilon
    epsilon_delta = (init_epsilon - fin_epsilon)/fin_exp

    # create DQN agent
    agent = DQN(ale, session,  1000000, epsilon, learning_rate, grad_mom, sgrad_mom, hist_len, len(ale.getLegalActionSet()), tgt_update_freq, discount)

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)

    #Load any saved memory
    if os.path.isfile(memory_file):
        memory = np.load(memory_file)
        for item in memory:
            replay_memory.append(Experience._make(item))
    #Load a saved Epsilon
    if o.spath.isfile(epsilon_file):
        epsilon = float(np.load(epsilon_file))

    num_frames = 0
    #TODO Change the episode ranges to be a function of frames
    episode_count = 1
    while ale.getFrameNumber() < 30000000:
        img = ale.getScreenRGB()
        #initialize sequence with initial image
        seq = list()
        seq.append(img)
        proc_seq = list()
        proc_seq.append(pp.preprocess(seq))
        total_reward = 0
        while not ale.game_over():
            state = get_state(proc_seq, hist_len)
            action = agent.get_action(state)
            #reward = 0
            
            #skip frames by repeating action
            #for i in range(act_rpt):
            #    reward = reward + ale.act(action)
            #    if ale.game_over():
            #        break
            reward = ale.act(action)

            total_reward += reward
            #cap reward
            reward = cap_reward(reward)
            # game state is just the pixels of the screen
            img = ale.getScreenRGB()
            #set s(t+1) = s_t, a_t, x_t+1
            seq.append(action)
            seq.append(img)
            #preprocess s_t+1
            proc_seq.append(pp.preprocess(seq))
            #store transition (phi(t), a_t, r_t, phi(t+1)) in replay_memory
            exp = get_experience(proc_seq, action, reward, hist_len, ale)
            replay_memory.append(exp)
            #then we can do learning
            if (num_frames > replay_start_size):
                epsilon = max(epsilon - epsilon_delta, fin_epsilon)
                agent.set_epsilon(epsilon)
                if num_frames % upd_freq == 0:
                    agent.train(replay_memory, minibatch_size) 
        print('Episode '+ str(episode_count) +' ended with score: %d' % (total_reward))
        print "Number of frames is " + str(ale.getFrameNumber())
        ale.reset_game()
        episode_count = episode_count + 1
        if episode_count % train_save_frequency:
            np.save(replay_memory, memory_file)
            np.save(epsilon, epsilon_file)
    print "Number " + str(num_frames)

#Returns hist_len most preprocessed frames and memory
def get_experience(proc_seq, action, reward, hist_len, ale):
    tplus = len(proc_seq) - 1
    exp_state = list()
    exp_new_state = list()
    '''
    If we don't have enough images to produce a history
    '''
    if len(proc_seq) < hist_len + 1:
        num_copy = hist_len - (len(proc_seq) - 1)
        for i in range(num_copy):
            exp_state.append(proc_seq[0])
        for i in range(len(proc_seq) - 1):
            exp_state.append(proc_seq[i])

        num_copy = hist_len - len(proc_seq)
        for i in range(num_copy):
            exp_new_state.append(proc_seq[0])
        for i in range(len(proc_seq)):
            exp_new_state.append(proc_seq[i])
    else:
        for i in range(len(proc_seq) - 1 - hist_len, len(proc_seq) - 1):
            exp_state.append(proc_seq[i])
        for i in range(len(proc_seq) - hist_len, len(proc_seq)):
            exp_new_state.append(proc_seq[i])
    exp = Experience(state=np.stack(np.array(exp_state),axis=2), action=action, reward=reward, new_state=np.stack(np.array(exp_new_state),axis=2), game_over=ale.game_over())
    return exp

def get_state(proc_seq, hist_len):
    if len(proc_seq) < hist_len + 1:
        num_copy = hist_len - len(proc_seq)
        state = ([proc_seq[0]] * num_copy) + (proc_seq)
        if not (np.shape(state) == (4, 84, 84)):
            print np.shape(state)
    else:
        state = proc_seq[-hist_len:]
        if not (np.shape(state) == (4, 84, 84)):
            print np.shape(state)
    #makes it 84 x 84 x hist_len (4, 84, 84) -> (84, 84, 4)
    return np.moveaxis(state, 0, -1)
    return state
    
def cap_reward(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

if __name__ == '__main__':
    with tf.Session() as session:
        train(session)