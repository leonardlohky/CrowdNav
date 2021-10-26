"""
Never Modify this file! Always copy the settings you want to change to your local file.
"""


import numpy as np


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.time_limit = 30
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    env.train_size = np.iinfo(np.uint32).max - 2000
    env.randomize_attributes = False
    env.robot_sensor_range = 5

    reward = Config()
    reward.success_reward = 1
    reward.collision_penalty = -0.25
    reward.discomfort_dist = 0.2
    reward.discomfort_dist_front = 0.25 # discomfort distance for the front half of the robot
    reward.discomfort_dist_back = 0.25 # discomfort distance for the back half of the robot
    reward.discomfort_penalty_factor = 0.5

    sim = Config()
    sim.train_val_scenario = 'circle_crossing'
    sim.test_scenario = 'circle_crossing'
    sim.square_width = 20
    sim.circle_radius = 4
    sim.human_num = 5
    sim.nonstop_human = False
    sim.centralized_planning = True
    sim.group_human = False

    humans = Config()
    humans.visible = True
    humans.policy = 'orca'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'
    humans.FOV = 2. # FOV = this values * PI

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False
    humans.goal_change_chance = 0.25
    
    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = False
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False
    
    noise = Config()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1
    
    robot = Config()
    robot.visible = False
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = 'coordinates'
    robot.FOV = 2. # FOV = this values * PI

    
    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1


class BasePolicyConfig(object):
    rl = Config()
    rl.gamma = 0.9

    om = Config()
    om.cell_num = 4
    om.cell_size = 1
    om.om_channel_size = 3

    action_space = Config()
    action_space.kinematics = 'holonomic'
    action_space.speed_samples = 5
    action_space.rotation_samples = 16
    action_space.sampling = 'exponential'
    action_space.query_env = False
    action_space.rotation_constraint = np.pi / 3

    cadrl = Config()
    cadrl.mlp_dims = [150, 100, 100, 1]
    cadrl.multiagent_training = False

    lstm_rl = Config()
    lstm_rl.global_state_dim = 50
    lstm_rl.mlp1_dims = [150, 100, 100, 50]
    lstm_rl.mlp2_dims = [150, 100, 100, 1]
    lstm_rl.multiagent_training = True
    lstm_rl.with_om = False
    lstm_rl.with_interaction_module = True

    gru_rl = Config()
    gru_rl.global_state_dim = 50
    gru_rl.mlp1_dims = [150, 100, 100, 50]
    gru_rl.mlp2_dims = [150, 100, 100, 1]
    gru_rl.multiagent_training = True
    gru_rl.with_om = False
    gru_rl.with_interaction_module = True
    
    srl = Config()
    srl.mlp1_dims = [150, 100, 100, 50]
    srl.mlp2_dims = [150, 100, 100, 1]
    srl.multiagent_training = True
    srl.with_om = True

    sarl = Config()
    sarl.mlp1_dims = [150, 100]
    sarl.mlp2_dims = [100, 50]
    sarl.attention_dims = [100, 100, 1]
    sarl.mlp3_dims = [150, 100, 100, 1]
    sarl.multiagent_training = True
    sarl.with_om = True
    sarl.with_global_state = True

    gcn = Config()
    gcn.multiagent_training = True
    gcn.num_layer = 2
    gcn.X_dim = 32
    gcn.wr_dims = [64, gcn.X_dim]
    gcn.wh_dims = [64, gcn.X_dim]
    gcn.final_state_dim = gcn.X_dim
    gcn.gcn2_w1_dim = gcn.X_dim
    gcn.planning_dims = [150, 100, 100, 1]
    gcn.similarity_function = 'embedded_gaussian'
    gcn.layerwise_graph = True
    gcn.skip_connection = False

    gnn = Config()
    gnn.multiagent_training = True
    gnn.node_dim = 32
    gnn.wr_dims = [64, gnn.node_dim]
    gnn.wh_dims = [64, gnn.node_dim]
    gnn.edge_dim = 32
    gnn.planning_dims = [150, 100, 100, 1]

    # SRNN config
    SRNN = Config()
    # RNN size
    SRNN.human_node_rnn_size = 128  # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 256  # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_input_size = 3  # Dimension of the node features
    SRNN.human_human_edge_input_size = 2  # Dimension of the edge features
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    SRNN.attention_size = 64  # Attention size
    
    # ppo
    ppo = Config()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.0  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter
    
    def __init__(self, debug=False):
        pass


class BaseTrainConfig(object):
    trainer = Config()
    trainer.batch_size = 100
    trainer.optimizer = 'Adam'

    imitation_learning = Config()
    imitation_learning.il_episodes = 2000
    imitation_learning.il_policy = 'orca'
    imitation_learning.il_epochs = 50
    imitation_learning.il_learning_rate = 0.001
    imitation_learning.safety_space = 0.15

    train = Config()
    train.rl_train_epochs = 1
    train.rl_learning_rate = 0.001
    train.num_processes = 12 # how many training CPU processes to use
    train.cuda = True  # use CUDA for training
    # number of batches to train at the end of training episode il_episodes
    train.train_batches = 100
    # training episodes in outer loop
    train.train_episodes = 10000
    train.num_env_steps = 10e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    # number of episodes sampled in one training episode
    train.sample_episodes = 1
    train.target_update_interval = 1000
    train.evaluation_interval = 1000
    # the memory pool can roughly store 2K episodes, total size = episodes * 50
    train.capacity = 100000
    train.epsilon_start = 0.5
    train.epsilon_end = 0.1
    train.epsilon_decay = 4000
    train.checkpoint_interval = 1000

    train.train_with_pretend_batch = False

    def __init__(self, debug=False):
        if debug:
            self.imitation_learning.il_episodes = 10
            self.imitation_learning.il_epochs = 5
            self.train.train_episodes = 1
            self.train.checkpoint_interval = self.train.train_episodes
            self.train.evaluation_interval = 1
            self.train.target_update_interval = 1
