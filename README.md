# dqn_atari
This repository will contain a python implementation of DQN used for atari games. This is written in Python using Tensorflow



![model](assets/model.png)

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values


## Prerequisites

- Python 2.7
- Arcade Learning Environment
- ffmpeg
- Avconv
- SciPy
- TensorFlow 0.12.0

## License

MIT License.
