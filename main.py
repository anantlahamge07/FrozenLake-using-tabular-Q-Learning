import gymnasium as gym
from frozenlake import Agent
from frozenlake import ENV_NAME
from frozenlake import TEST_EPISODES
from torch.utils.tensorboard.writer import SummaryWriter


class Main():


    # The training loop
    if __name__ == "__main__":
        # the test environment
        test_env = gym.make(ENV_NAME)
        # our agent
        agent = Agent()
        writer = SummaryWriter(comment = "-q-learning")
        best_reward = 0.0
        iter_no = 0
        while True:
            iter_no += 1
            # playing one step in the environment
            old_state, action, reward, new_state = agent.sample_env()
            # performing the value update using the obtained data
            agent.value_update(old_state, action, reward, new_state)
            # now we will test our current policy by playing several test episodes
            test_reward = 0.0
            for _ in range(TEST_EPISODES):
                test_reward += agent.play_episode(test_env)
            test_reward /= TEST_EPISODES
            # writing the data to the tensorboard
            writer.add_scalar("reward", test_reward, iter_no)
            # updating the best_reward 
            if test_reward > best_reward:
                print(f"best reward updated {best_reward} -> {test_reward}")
                best_reward = test_reward
            # our convergence condition
            if test_reward > 0.8:
                print(f"solved in {iter_no} iterations!")
                break
        # closing the SummaryWriter
        writer.close()
            

            