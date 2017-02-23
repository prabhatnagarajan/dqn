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

Experience = namedtuple('Experience', 'state action reward new_state game_over')

def train(session, minibatch_size=32, replay_capacity=1000000, hist_len=4, tgt_update_freq=10000,
    discount=0.99, act_rpt=4, upd_freq=4, learning_rate=0.00025, grad_mom=0.95,
    sgrad_mom=0.95, min_sq_grad=0.01, init_epsilon=1.0, fin_epsilon=0.1, 
    fin_exp=1000000, replay_start_size=50000, no_op_max=30, epsilon_file="epsilon.npy", memory_file="memory.npy", train_save_frequency=2000):
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

    #initialize epsilon
    epsilon = init_epsilon
    epsilon_delta = (init_epsilon - fin_epsilon)/fin_exp
    #Load saved Epsilon from file
    if os.path.isfile(epsilon_file):
        epsilon = float(np.load(epsilon_file)[0])

    # create DQN agent
    agent = DQN(ale, session,  1000000, epsilon, learning_rate, grad_mom, sgrad_mom, hist_len, len(ale.getMinimalActionSet()), tgt_update_freq, discount)

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)

    #Store the most recent two images
    preprocess_stack = deque([], 2)

    num_frames = 0
    episode_num = 1
    while num_frames < 30000000:
        #initialize sequence with initial image
        seq = list()

        #proc_seq.append(pp.preprocess(seq))
        perform_no_ops(ale, no_op_max, preprocess_stack, seq)
        total_reward = 0
        lives = ale.lives()
        episode_done = False
        while not episode_done:
            #state = get_state(proc_seq, hist_len)
            state = get_state(seq, hist_len)
            action = agent.get_action(state)
            reward = 0
            
            #skip frames by repeating action
            for i in range(act_rpt):
                reward = reward + ale.act(action)
                #add the images on stack 
                preprocess_stack.append(ale.getScreenRGB())

            total_reward += reward
            #cap reward
            reward = np.clip(reward, -1, 1)

            #game state is just the pixels of the screen
            #Order shouldn't matter between images
            img = pp.preprocess(preprocess_stack[0], preprocess_stack[1])
            
            #set s(t+1) = s_t, a_t, x_t+1
            seq.append(img)

            #store transition (phi(t), a_t, r_t, phi(t+1)) in replay_memory
            #Does getExperience assume a certain input format for processed sequence?
            episode_done = ale.game_over() or (ale.lives() < lives)
            exp = get_experience(seq, action, reward, hist_len, episode_done)
            replay_memory.append(exp)
            #Training
            if (num_frames > replay_start_size):
                epsilon = max(epsilon - epsilon_delta, fin_epsilon)
                agent.set_epsilon(epsilon)
                if num_frames % upd_freq == 0:
                    agent.train(replay_memory, minibatch_size) 
            num_frames = num_frames + 1
            #we end episode if life is lost or game is over
        print('Episode '+ str(episode_num) +' ended with score: %d' % (total_reward))
        print "Number of frames is " + str(num_frames)
        ale.reset_game()
        episode_num = episode_num + 1
        #Save epsilon value to a file
        if episode_num % train_save_frequency == 0:
            #np.save(replay_memory, memory_file)
            np.save(epsilon_file, [epsilon])
    print "Number " + str(num_frames)

#Returns hist_len most preprocessed frames and memory
def get_experience(seq, action, reward, hist_len, episode_done):
    exp_state = list()
    exp_new_state = list()
    '''
    If we don't have enough images to produce a history
    '''
    if len(seq) < hist_len + 1:
        num_copy = hist_len - (len(seq) - 1)
        for i in range(num_copy):
            exp_state.append(seq[0])
        for i in range(len(seq) - 1):
            exp_state.append(seq[i])

        num_copy = hist_len - len(seq)
        for i in range(num_copy):
            exp_new_state.append(seq[0])
        for i in range(len(seq)):
            exp_new_state.append(seq[i])
    else:
        exp_state = seq[-hist_len - 1 : -1]
        exp_new_state = seq[-hist_len:]
    exp = Experience(state=np.moveaxis(exp_state, 0, -1), action=action, reward=reward, new_state=np.moveaxis(exp_new_state, 0, -1), game_over=episode_done)
    return exp

def get_state(seq, hist_len):
    if len(seq) < hist_len + 1:
        num_copy = hist_len - len(seq)
        state = ([seq[0]] * num_copy) + (seq)
    else:
        state = seq[-hist_len:]
    #makes it 84 x 84 x hist_len (4, 84, 84) -> (84, 84, 4)
    return np.moveaxis(state, 0, -1)

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
        train(session)