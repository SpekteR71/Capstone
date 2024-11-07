import numpy as np
from DRL_agent import DRLAgent
from DRL_env import TwoWheelerEnv
import random

def main():
    env = TwoWheelerEnv()  # Instantiate the environment class
    state_dim = 11
    action_dim = 3
    agent = DRLAgent(state_dim, action_dim)
    episodes = 2800
    steps_per_episode = 15000
    checkpoint_interval = 25  # Save checkpoint every 10 episodes
    resume_from_checkpoint = False  # Set to True if you want to resume training from a checkpoint
    last_checkpoint_episode = 0  # Set this to the episode number you want to resume from

    if resume_from_checkpoint and last_checkpoint_episode > 0:
        agent.load_checkpoint(last_checkpoint_episode)

    with open('rewards.txt', 'a') as rewards_file:  
        for episode in range(last_checkpoint_episode, episodes):
            state = env.reset()  
            destination_x = random.uniform(1, 6)  # Randomize x in range 2-5
            destination_y = random.uniform(-3, 3)  # Randomize y in range -3 to 3
            start_x, start_y = state[0], state[1]  # Extract the starting coordinates
            
            # Log starting coordinates
            rewards_file.write(f"Episode {episode + 1} Start: ({round(start_x, 1)}, {round(start_y, 1)})\n")
            
            episode_reward = 0
            done = False
            if (episode + 1) % checkpoint_interval == 0:
                agent.save_checkpoint(episode + 1)
                last_checkpoint_episode = episode
                
            for t in range(1, steps_per_episode + 1):
                action = agent.select_action(np.array(state))
                next_state, reward, done = env.step(state, action, destination_x, destination_y)
                agent.replay_buffer.add((state, action, next_state, reward, float(done)))
                if t % 20 == 0:
                    agent.train(episode=episode)  # Pass the current episode number to the train function
                state = next_state
                episode_reward += reward
                
                print(f"Episode: {episode + 1}, Step: {t}, Reward: {reward}, Done: {done}")

                if done:
                    break

            # Calculate distance reached
            final_x, final_y = state[0], state[1]
            distance_reached = np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2)
            
            # Log total reward and distance reached to rewards.txt
            rewards_file.write(f"Episode {episode + 1} Reward: {episode_reward}, Distance Reached: {round(distance_reached, 2)}\n")
            
            if episode % 7 == 0:
                agent.update_epsilon()
        
    agent.save_to_h5()
    print("Training completed.")

if __name__ == "__main__":
    print("----------------------")
    print("ASBTW Training Grounds")
    print("----------------------")
    main()
