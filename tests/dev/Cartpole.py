from src import GymAgent
from tests.utils import make_model_for_env

env_id = 'LunarLander-v2'
model = make_model_for_env(env_id)
a = GymAgent(env_id, noise=None, model=model)
