from environment import GameEnv
from agent import DoubleDQNAgent
import numpy as np
import time
import torch
import os  # Needed to check for the save file

def train():
    env = GameEnv()
    # 4 Stacked Frames (84x84), 2 Actions (Jump/Fall)
    agent = DoubleDQNAgent(input_shape=(4, 84, 84), num_actions=2)
    
    # --- CHECKPOINT LOADING SYSTEM ---
    model_filename = "jetpack_model.pth"
    
    if os.path.exists(model_filename):
        print("💾 Found a saved brain! Loading it...")
        
        # 1. Load the weights
        checkpoint = torch.load(model_filename)
        agent.online_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        
        # 2. Set Epsilon to 0.8 so it explores a bit before getting serious
        # It will naturally decay from 0.8 -> 0.02 as it plays.
        agent.epsilon = 0.8 
        
        print(f"✅ Brain Loaded. Resuming with Epsilon: {agent.epsilon}")
    else:
        print("👶 No saved brain found. Starting from scratch.")
    # ---------------------------------

    episodes = 7000 # Increased to 7000 as requested
    
    print(f"🚀 Training Started for {episodes} episodes! Switch to the game window now.")
    time.sleep(3) 

    for e in range(episodes):
        state = env.reset()
        
        # Create a stack of 4 frames
        state_stack = np.stack([state] * 4, axis=0)
        
        total_reward = 0
        done = False
        
        while not done:
            # 1. Choose Action
            action = agent.choose_action(state_stack)
            
            # 2. Play
            next_state_img, reward, done = env.step(action)
            
            # 3. Update Stack
            next_state_stack = np.append(state_stack[1:], [next_state_img], axis=0)
            
            # 4. Remember & Learn
            agent.store_transition(state_stack, action, reward, next_state_stack, done)
            agent.learn() # This function automatically lowers epsilon slightly every step
            
            state_stack = next_state_stack
            total_reward += reward
        
        # Update Target Network & Save Model every 10 episodes
        if e % 10 == 0:
            agent.update_target_net()
            torch.save(agent.online_net.state_dict(), model_filename)
            print("💾 Model Saved!")

        print(f"Episode {e} | Reward: {total_reward} | Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    train()