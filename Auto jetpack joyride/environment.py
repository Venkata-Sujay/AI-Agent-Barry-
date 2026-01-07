import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
import time
import pydirectinput
import os

# --- STEP 1: DPI AWARENESS SETUP ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

class GameEnv:
    def __init__(self):
        self.sct = mss.mss()
        self.window_name = "PPSSPP"
        
        # --- LOAD DEATH DETECTION IMAGE ---
        # Checks if you have the "STATISTICS" crop saved
        if os.path.exists("game_over.png"):
            self.game_over_img = cv2.imread("game_over.png", 0) # Load as grayscale
            print("✅ Loaded 'game_over.png' for death detection.")
        else:
            print("⚠️ WARNING: 'game_over.png' not found. Auto-death detection will fail!")
            self.game_over_img = None

        # --- FIND GAME WINDOW ---
        windows = gw.getWindowsWithTitle(self.window_name)
        if not windows:
            raise Exception(f"❌ Game window '{self.window_name}' not found! Is PPSSPP open?")
        
        win = windows[0]
        # Calculate the capture region (removing window borders)
        self.monitor = {
            "top": win.top + 30,
            "left": win.left + 8,
            "width": win.width - 16,
            "height": win.height - 38
        }
        
    def get_raw_screen(self):
        """Helper to grab the full-resolution screen for death detection."""
        raw = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)

    def reset(self):
        """Restarts the game loop."""
        print("🔄 Resetting Environment...")
        
        # Spam space to get past "Game Over" screen
        pydirectinput.press('space')
        time.sleep(0.5)
        pydirectinput.press('space') 
        time.sleep(1.5) # Wait for Barry to respawn
        
        # Return the first frame
        raw_screen = self.get_raw_screen()
        return cv2.resize(raw_screen, (84, 84)) / 255.0

    def step(self, action):
        """
        Performs an action with FRAME SKIPPING.
        Repeats the action 4 times to speed up training.
        """
        total_reward = 0
        done = False
        
        # --- FRAME SKIPPING LOOP (Repeat action 4 times) ---
        for _ in range(4):
            # 1. Perform Action
            if action == 1:
                pydirectinput.keyDown('space')
            else:
                pydirectinput.keyUp('space')
            
            # 2. Wait for 1 Game Frame (approx 0.016s for 60FPS)
            time.sleep(0.015) 
            
            # 3. Accumulate Reward (+1 for every frame we survive)
            total_reward += 1.0
            
            # Note: We don't check for death inside this tiny loop for speed.
            # We check it once after the 4 frames are done.
        # ---------------------------------------------------
        
        # 4. Get New State (Capture screen once after the 4 skips)
        raw_screen = self.get_raw_screen()
        
        # Resize for the AI (The brain needs 84x84)
        observation = cv2.resize(raw_screen, (84, 84)) / 255.0

        # 5. Check Death (The robust check using the full-size image)
        if self.game_over_img is not None:
            res = cv2.matchTemplate(raw_screen, self.game_over_img, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.6: 
                print("💀 Game Over detected (Visual Match)!")
                done = True
                total_reward = -100 # Penalty overrides the survival points
        
        return observation, total_reward, done