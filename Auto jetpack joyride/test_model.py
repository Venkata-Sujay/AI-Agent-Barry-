from environment import GameEnv
from agent import DoubleDQNAgent
import torch
import numpy as np
import time
import os

def test_brain():
    # 1. Setup Environment
    env = GameEnv()
    
    # 2. Setup Agent (Must match training shape)
    agent = DoubleDQNAgent(input_shape=(4, 84, 84), num_actions=2)
    
    model_filename = "jetpack_model.pth"
    
    # 3. Load the Saved Brain
    if os.path.exists(model_filename):
        print(f"🧠 Loading {model_filename} for testing...")
        checkpoint = torch.load(model_filename, map_location=torch.device('cpu'))
        agent.online_net.load_state_dict(checkpoint)
        
        # CRITICAL: Turn off randomness! 
        # We want to see what it LEARNED, not random luck.
        agent.epsilon = 0.0 
        print("✅ Model loaded! Randomness (Epsilon) set to 0.0")
    else:
        print("❌ Error: No 'jetpack_model.pth' found. Did you train yet?")
        return

    print("🚀 TESTING MODE STARTING IN 3 SECONDS...")
    print("⚠️ CLICK THE GAME WINDOW NOW!")
    time.sleep(3)
    
    # 4. Play 5 Games
    for e in range(1, 6):
        state = env.reset()
        # Stack the first frame 4 times to match input shape
        state_stack = np.stack([state] * 4, axis=0)
        
        total_reward = 0
        done = False
        
        while not done:
            # Agent chooses action based ONLY on the brain
            action = agent.choose_action(state_stack)
            
            # Perform action
            next_state_img, reward, done = env.step(action)
            
            # Update stack
            next_state_stack = np.append(state_stack[1:], [next_state_img], axis=0)
            state_stack = next_state_stack
            
            total_reward += reward
        
        print(f"Game {e} Result: {total_reward} Points")
        time.sleep(2) # Pause briefly between games

if __name__ == "__main__":
    test_brain()