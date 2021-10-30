from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.gru_rl import GruRL
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.tree_search_rl import TreeSearchRL
from crowd_nav.policy.bilstm_rl import biLstmRL


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['bilstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
policy_factory['gru_rl'] = GruRL
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['tree_search_rl'] = TreeSearchRL
policy_factory['bilstm_rl'] = biLstmRL