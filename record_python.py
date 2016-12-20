#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import os
import thread
from keylistener import *
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 3:
  print('Usage: %s rom_file' % sys.argv[0])
  sys.exit()

ale = ALEInterface()

#event = CEvent()

# Get & Set the desired settings
ale.setInt(b'random_seed', 123)
ale.setBool('sound', False)
ale.setBool('display_screen', True)
record_path = sys.argv[2]
ale.setString('record_screen_dir', record_path)
ale.setString('record_sound_filename', (record_path + '/sound.wav'))
ale.setInt('fragsize', 64)
cmd = 'mkdir '
cmd += record_path 
os.system(cmd)
rom_file = str.encode(sys.argv[1])
ale.loadROM(rom_file)
legal_actions = ale.getLegalActionSet()

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

def play():
  # Play 10 episodes
  for episode in range(1):
    #thread.start_new_thread( handle(), ("Thread-1", 2, ) )
    total_reward = 0
    while not ale.game_over():
      a = legal_actions[randrange(len(legal_actions))]
      # Apply an action and get the resulting reward
      reward = ale.act(a);
      total_reward += reward
    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
  print 'Recording complete.'
thread.start_new_thread( handle(), ("Thread-1", 2, ) )
thread.start_new_thread( play(), ("Thread-2", 4, ) )
