from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'bilstm_rl'

        self.bilstm_rl.global_state_dim = 50
        self.bilstm_rl.mlp1_dims = [150, 100, 100, 50]
        self.bilstm_rl.mlp2_dims = [150, 100, 100, 1]
        self.bilstm_rl.multiagent_training = True
        self.bilstm_rl.with_om = False
        self.bilstm_rl.with_interaction_module = True


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
