import typing as tt
import gymnasium as gym
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

# The name of the environment
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
# The learning rate, which will be used in the value update
ALPHA = 0.2
TEST_EPISODES = 20

# defining the types of the states and actions
State = int
Action = int
# The type of the key of our values table (Q(s,a))
ValuesKey = tt.Tuple[State, Action]


class Agent:
    def __init__(self):
        # creating the environment
        self.env = gym.make(ENV_NAME)
        # getting the initial state and some unnecessary info from the environment
        self.state, _ = self.env.reset()
        # The value table for each state and action pair holding the Q(s,a) values  
        self.values: tt.Dict[ValuesKey, float] = defaultdict(float)

    # This method just gives us the next transition obtained from the environment
    def sample_env (self) -> tt.Tuple[State, Action, float, State]:
        # the current state
        old_state = self.state
        # The action 
        action = self.env.action_space.sample()
        # Taking a step in the environment
        new_state, reward, is_done, is_truncated, _  = self.env.step(action)
        # resetting the environment if the episode has ended or truncated
        if is_done or is_truncated:
            self.state, _ = self.env.reset()
        # updating the current state with the new state
        else:
            self.state = new_state
        return old_state, action, float(reward), new_state
    
    # this method finds the best action to take for the given state
    # it actually gives us the maximum Q value for the given state, and the action for which we got the maximum value
    def best_action_value(self, state: State) -> tt.Tuple[float, Action]:
        best_value, best_action = None, None
        # iterating over all the possible actions
        for action in range(self.env.action_space.n):
            # getting the action value from our table for the current state action pair
            value = self.values[(state, action)]
            # updating the best_value and best_action if the value is greater than the best_value
            if best_value is None or value > best_value:
                best_value = value
                best_action = action
        return best_value, best_action

    def value_update(self, state: State, action: Action, reward: float, new_state: State):
        # the old value
        old_value = self.values[(state, action)]
        # the best/max action value
        best_value, _ = self.best_action_value(new_state)
        # the bellman approximation
        b = reward + GAMMA *  best_value
        # the Bellman update using the blending technique
        self.values[(state, action)] = (1 - ALPHA) * old_value + ALPHA * b 
    

    # this method plays one full episode
    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        # getting the initial state and some other info from the given environment
        state, _ = env.reset()
        while True:
            # getting the best action to take from the given state
            _, action = self.best_action_value(state)
            # doing next transition in the environment
            new_state, reward, is_done, is_truncated, _  = env.step(action)
            # accumulation the total reward
            total_reward += reward
            if is_done or is_truncated:
                break
            # updating the current state
            state = new_state
        return total_reward
    
# The training loop
if __name__ == "__main__":
    # our agent
    agent = Agent()
    