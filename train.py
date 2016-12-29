import sys
from random import randrange
from dqn import DQN
from ale_python_interface import ALEInterface
from collections import deque
import preprocess as pp

def train(minibatch_size=32, replay_capacity=1000, hist_len=4, tgt_update_freq=10000,
    discount=0.99, act_rpt=4, upd_freq=4, learning_rate=0.00025, grad_mom=0.95,
    sgrad_mom=0.95, min_sq_grad=0.01, init_epsilon=1.0, fin_epsilon=0.1, 
    fin_exp=1000000, replay_size=50000, noop_max=30):
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
    agent = DQN(ale, 1000000, 32)

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)

    # Initialize action-value function Q with random weights h
    #----Handled with Tensorflow
    # Initialize target action-value function Q^ with weights h2 5 h
    #----Handled with Tensorflow

    num_frames = 0
    for episode in range(2):
        img = ale.getScreenRGB()
        #initialize sequence with initial image
        seq = list()
        seq.append(img)
        proc_seq = list()
        proc_seq = pp.preprocess(seq)
        total_reward = 0
        while not ale.game_over():
            # game state is just the pixels of the screen
            state = ale.getScreenRGB()
            num_frames = num_frames + 1
            # the agent maps states to actions
            action = agent.get_action(state)
            # Apply an action and get the resulting reward
            reward = ale.act(action)
            total_reward += reward
        print('Episode %d ended with score: %d' % (episode, total_reward))
        ale.reset_game()
train()