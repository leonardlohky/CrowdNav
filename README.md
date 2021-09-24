## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.

First, create a config file for the particular policy to be trained. E.g. if training an LSTM-RL policy, create a config named lstm_rl.py under crowd_nav/configs/icra_benchmark.
The base config file should at least contain the following lines:
```
from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config

class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'sarl' # change policy name here


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
```

After creating the config file for the policy, train it
```
# default params are for RGL policy
python train.py --policy rgl

# to train other policies, e.g. LSTM-RL
python train.py --policy lstm_rl --config configs/icra_benchmark/lstm_rl.py --output_dir data_lstmRL/output
```
2. Test policies with 500 test cases.
```
python test.py --policy rgl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy rgl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```



## Video Demo
[<img src="https://i.imgur.com/SnqVhit.png" width="70%">](https://youtu.be/U3quW30Eu3A)


## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```bibtex
@inproceedings{chen2020relational,
    title={Relational Graph Learning for Crowd Navigation},
    author={Changan Chen and Sha Hu and Payam Nikdel and Greg Mori and Manolis Savva},
    year={2020},
    booktitle={IROS}
}
@inproceedings{chen2019crowdnav,
    title={Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
    author={Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
    year={2019},
    booktitle={ICRA}
}
```
