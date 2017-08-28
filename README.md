# Implementation of ACER (Actor-Critic with Experience replay)

Contains the tensorflow and sonnet implementation for SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY by Ziyu Wang, Victor Bapst et al from Deepmind  (https://arxiv.org/abs/1611.01224).

The current version is tested only for MuJoCo gym environments,.

### Major dependencies

- Tensorflow v1.3 (https://www.tensorflow.org/install/)
- Sonnet (https://deepmind.github.io/sonnet/)
- Python 2.7

#### Running

```
python train.py --model_dir ./tmp_model/ --env InvertedPendulum-v1 --eval_every_sec 60 --num_agents 4
```

See `python train.py --help` for a full list of options. 


You can monitor training progress in Tensorboard:

```
tensorboard --logdir=/tmp_model/
```

#### Components

- [`train.py`](train.py) contains the main method to start training.
- [`agent.py`](agent.py) contains the code for the agent threads and actual ACER algortihm
- [`advantage_net.py`](advantage_net.py) contains code for building the stochasitic dueling net
- [`policy_net.py`](policy_net.py) contains code for building the policy network
- [`memory.py`](memory.py) contains the memory class for experience replay

#### References

- ACER, ICLR 2017 (https://arxiv.org/abs/1611.01224)
- Denny Britz a3c implementation (https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c)
- TF (https://www.tensorflow.org)
- Sonnet (https://deepmind.github.io/sonnet/)
