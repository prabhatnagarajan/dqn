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
from constants import *

Experience = namedtuple('Experience', 'state action reward new_state game_over')
reward_history = []
reward_counts = []
episode_counts = []

def train(session, minibatch_size=MINIBATCH_SIZE, replay_capacity=REPLAY_CAPACITY, hist_len=HIST_LEN, tgt_update_freq=TGT_UPDATE_FREQ,
    discount=DISCOUNT, act_rpt=ACT_REPEAT, upd_freq=UPDATE_FREQ, learning_rate=LEARNING_RATE, grad_mom=GRADIENT_MOMENTUM,
    sgrad_mom=SQUARED_GRADIENT_MOMENTUM, min_sq_grad=MIN_SQUARED_GRADIENT, init_epsilon=INITIAL_EPSILON, fin_epsilon=FINAL_EPSILON, 
    fin_exp=FINAL_EXPLORATION_FRAME, replay_start_size=REPLAY_START_SIZE, no_op_max=NO_OP_MAX, epsilon_file=EPSILON_FILE, 
    num_frames_file=NUM_FRAMES_FILE, memory_file=MEMORY_FILE, reward_hist_file=REWARD_HIST_FILE, 
    train_save_frequency=TRAIN_SAVE_FREQUENCY):
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

    print "Minimal Action set is:"
    print ale.getMinimalActionSet()
    # create DQN agent
    agent = DQN(ale, session, epsilon, learning_rate, grad_mom, sgrad_mom, hist_len, len(ale.getMinimalActionSet()), tgt_update_freq, discount, rom_name(sys.argv[1]))

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)

    #Store the most recent two images
    preprocess_stack = deque([], 2)

    num_frames = 0


    if os.path.isfile(epsilon_file) and os.path.isfile(memory_file) and os.path.isfile(num_frames_file) and os.path.isfile(reward_hist_file):
        epsilon, num_frames, replay_memory, reward_history= load(epsilon_file, num_frames_file, memory_file, replay_capacity, reward_hist_file)

    print "Initial epsilon value is " + str(epsilon)
    print "Replay Memory size is " + str(len(replay_memory))
    print "Num Frames passed is " + str(num_frames)

    episode_num = 1
    while num_frames < TRAINING_FRAMES:
        seq = list()

        perform_no_ops(ale, no_op_max, preprocess_stack, seq)
        total_reward = 0
        lives = ale.lives()
        episode_done = False
        while not episode_done:
            state = get_state(seq, hist_len)
            action = agent.get_action(state)
            reward = 0
            
            #skip frames by repeating action
            for i in range(act_rpt):
                reward = reward + ale.act(action)
                #add the images on stack 
                preprocess_stack.append(ale.getScreenRGB())

            total_reward += reward
            reward = np.clip(reward, -1, 1)

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
                if num_frames % EVAL_FREQ == 0:
                    validate(ale, agent, no_op_max, hist_len)
            num_frames = num_frames + 1
            '''
            Inconsistency in Deepmind code versus Paper. In code they update target
            network every tgt_update_freq actions. In the the paper they say to do
            it every tgt_update_freq parameter updates.
            '''
            if num_frames % tgt_update_freq == 1:
                print "Copying Network"
                agent.copy_network()
                print "Done Copying"
            #Save epsilon value to a file
            if num_frames % train_save_frequency == 0:
                save(epsilon_file, num_frames_file, memory_file, reward_hist_file, epsilon, num_frames, replay_memory, reward_history)

            #we end episode if life is lost or game is over
        print('Episode '+ str(episode_num) +' ended with score: %d' % (total_reward))
        print "Number of frames is " + str(num_frames)
        ale.reset_game()
        episode_num = episode_num + 1

    print "Number " + str(num_frames)

def validate(ale, agent, no_op_max, hist_len):
    ale.reset_game()
    seq = list()
    preprocess_stack = deque([], 2)
    perform_no_ops(ale, no_op_max, preprocess_stack, seq)
    total_reward = 0
    num_rewards = 0
    num_episodes = 0
    episode_reward = 0
    eval_time = time()
    for _ in range(EVAL_STEPS):
        state = get_state(seq, hist_len)
        action = agent.eGreedy_action(state, TEST_EPSILON)
        reward = ale.act(action)
        episode_reward += reward
        if not (reward == 0):
            num_rewards += 1
        if ale.game_over():
            total_reward += episode_reward
            episode_reward = 0
            num_episodes += 1
            ale.reset_game()
            seq = list()
            perform_no_ops(ale, no_op_max, preprocess_stack, seq)
    total_reward = total_reward/max(1, num_episodes)
    if len(reward_history) == 0 or total_reward > max(reward_history):
        agent.update_best_scoring_network()
    reward_history.append(total_reward)
    reward_counts.append(num_rewards)
    episode_counts.append(num_episodes)

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
    for _ in range(np.random.randint(1, no_op_max + 1)):
        ale.act(0)
        preprocess_stack.append(ale.getScreenRGB())
    if len(preprocess_stack) < 2:
        ale.act(0)
        preprocess_stack.append(ale.getScreenRGB())
    seq.append(pp.preprocess(preprocess_stack[0], preprocess_stack[1]))

def save(epsilon_file, num_frames_file, memory_file, reward_hist_file, epsilon, num_frames, replay_memory, reward_history):
    print "Saving info"
    np.save(epsilon_file, [epsilon])
    np.save(num_frames_file, [num_frames])
    np.save(memory_file, replay_memory)
    np.save(reward_hist_file, reward_history)
    print "Saved info"

def load(epsilon_file, num_frames_file, memory_file, replay_capacity, reward_hist_file):
    "Loading Saved Training Information"
    epsilon = float(np.load(epsilon_file)[0])
    num_frames = int(np.load(num_frames_file)[0])
    memory = np.load(memory_file)
    memory = memory.tolist()
    replay_memory = deque([Experience._make(exp) for exp in memory], replay_capacity)
    reward_history = np.load(reward_hist_file)
    reward_history = reward_history.tolist()
    print "Loaded Training Information"
    return (epsilon, num_frames, replay_memory, reward_history)

def rom_name(path):
    return os.path.splitext(os.path.basename(path))[0]

if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=NUM_THREADS)) as session:
        train(session)