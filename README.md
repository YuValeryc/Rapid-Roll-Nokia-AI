# Rapid Roll AI

**Rapid Roll AI** is a game built with **Pygame**, where the player controls a ball that moves across platforms. This project also features an **AI agent** trained with the **Deep Q-Network (DQN)** algorithm to play the game automatically.

## Demo

![Demo](assets/demo.gif)

## Features

- **Manual Mode**: Control the ball using ← and → arrow keys.
- **AI Mode**: Automatically plays the game using a pre-trained DQN model.
- **User Interface**: Includes main menu, game over screen, and score display.
- **Increasing Difficulty**: The game scroll speed increases over time, making it progressively more challenging.

## Project Structure

```plaintext
├── main.py              # Main game loop, event handling, and UI
├── rapid_roll_env.py    # Game environment and logic (ball, platforms, physics)
├── dqn_agent.py         # Deep Q-Network agent (model, replay buffer, training logic)
├── train_dqn.py         # Script to train the DQN agent
├── assets/              # Assets folder (e.g., demo.gif)
└── requirements.txt     # Required Python libraries

## Installation

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd Rapid-Roll
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game

- **To train the DQN model:**
  ```bash
  python train_dqn.py
  ```

- **To play manually:**
  ```bash
  python main.py
  ```
