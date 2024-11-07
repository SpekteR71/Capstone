import numpy as np
from DRL_agent import DRLAgent
from DRL_env import TwoWheelerEnv
import random

def main():
    env = TwoWheelerEnv()  # Instantiate the environment class
    state_dim = 11
    action_dim = 3
    agent = DRLAgent(state_dim, action_dim)
    
    # Load the pre-trained models
    agent.load_actor("actor.h5")
    agent.load_critic("critic.h5")
    
    episodes = 10  # Number of test episodes
    steps_per_episode = 20000  # Maximum number of steps per episode

    with open("test_rewards.txt", "w") as f:
        for episode in range(episodes):
            state = env.reset()  
            destination_x = random.uniform(1, 6)  # Randomize x in range 2-5
            destination_y = random.uniform(-3, 3)  # Randomize y in range -3 to 3
            episode_reward = 0
            done = False
            final_x, final_y = None, None
            
            for t in range(1, steps_per_episode + 1):
                # Use the exploit method to select the action
                action = agent.exploit(np.array(state))
                
                # Take a step in the environment
                next_state, reward, done = env.step(state, action, destination_x, destination_y)
                state = next_state
                episode_reward += reward
                
                print(f"Episode: {episode + 1}, Step: {t}, Reward: {reward}, Done: {done}")

                if done:
                    final_x, final_y = state[0], state[1]  # Capture final x, y coordinates
                    break

            f.write(f"Episode {episode + 1}: Total Reward: {episode_reward}\n")
            if final_x is not None and final_y is not None:
                f.write(f"Final Position: x = {final_x}, y = {final_y}\n")

            print(f"Episode {episode + 1} finished with total reward: {episode_reward}")

if __name__ == "__main__":
    print("----------------------")
    print("ASBTW Testing Grounds")
    print("----------------------")
    main()
