import typing as tp
import gymnasium as gym
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

# The name of the environment
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

# defining the types of the states and actions
State = int
Action = int
# The type of the key of our values table (Q(s,a))
ValuesKey = tt.Tuple[State, Action]

