# dqn
This repository contains a python implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf) using Python and Tensorflow. This is done in the Arcade Learning Environment, as in the original paper.

## Status
The status of this repository is 'Working'. There are a few details that are different from the Deepmind implementation that are work in progress, but the implementation is correct.

## Prerequisites

- Python 2.7
- Arcade Learning Environment
- Avconv
- SciPy
- TensorFlow 0.12.0
- opencv-python-3.2.0.6

## API
 - TODO

## References
I used a number of references in building this. I apologize for any references I may have excluded.
- [Original Deepmind Code](https://sites.google.com/a/deepmind.com/dqn/)
- [alewrap](https://github.com/deepmind/alewrap)
- [deep_q_rl](https://github.com/spragunr/deep_q_rl)
- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [DeepMind-Atari-Deep-Q-Learner](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)
- [simple_dqn](https://github.com/tambetm/simple_dqn)
- [Discussion](https://github.com/dennybritz/reinforcement-learning/issues/30)

## Results
These results are obtained after training for a few million frames on The Texas Advanced Computing Center's Maverick System, on an NVIDIA Tesla K40 GPU.

![breakout](gifs/breakout.gif)

## License

MIT License.
