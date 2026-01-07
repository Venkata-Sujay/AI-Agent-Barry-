# 🚀 Auto Jetpack Joyride: Deep RL Agent

An autonomous AI agent trained to play **Jetpack Joyride** using Deep Reinforcement Learning (Double DQN) and Computer Vision.

The agent "sees" the game screen in real-time, processes visual data using a Convolutional Neural Network (CNN), and learns to dodge zappers and missiles through trial and error.

---

## 🎮 Demo
*https://www.linkedin.com/posts/venkata-sujay-902a27290_reinforcementlearning-ai-deeplearning-activity-7414724589584519169-qmyx?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEahjlIBpmibRs4RUW55e4geb0VIavR95kk*

## ✨ Key Features
* **Double DQN Architecture:** Uses a Dueling Deep Q-Network to estimate state values and advantages separately for better stability.
* **Computer Vision Interface:** Custom-built environment that captures the screen using `mss` and `OpenCV` (no game API used).
* **Visual Death Detection:** Uses Template Matching to detect the "Game Over" screen instantly and reset the training loop.
* **Frame Stacking & Skipping:** Processes 4 stacked frames at a time to understand velocity and acceleration.
* **Experience Replay:** Stores 10,000+ past moves to learn from diverse gameplay scenarios.

## 🛠️ Tech Stack
* **Language:** Python 3.12+
* **Machine Learning:** PyTorch (Double DQN, CNN)
* **Computer Vision:** OpenCV, NumPy
* **Automation:** MSS (Screen Capture), PyDirectInput (Controls), PyGetWindow

## 📂 Project Structure
* `agent.py`: The Brain (Neural Network logic & Agent).
* `environment.py`: The Body (Screen capture, controls, death detection).
* `train.py`: The Training Loop (Main script).
* `test_model.py`: The Testing Script (Runs the trained model without randomness).
* `jetpack_model.pth`: The Saved Model (Neural weights).
* `game_over.png`: Reference image for death detection.

## ⚙️ Setup & Installation

1.  **Prerequisites:**
    * Python 3.10 or higher
    * PPSSPP Emulator running Jetpack Joyride

2.  **Install Dependencies:**
    ```text
    pip install torch torchvision numpy opencv-python mss pygetwindow pydirectinput
    ```

3.  **Prepare the Game:**
    * Open PPSSPP and start Jetpack Joyride.
    * Ensure the window title is exactly `PPSSPP`.

## 🚀 How to Run

### 1. Train the Agent
Run the training script to let the AI play and learn from scratch:
```text
python train.py
