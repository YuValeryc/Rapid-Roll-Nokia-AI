# train_dqn.py

import torch
from rapid_roll_env import RapidRollEnv
from dqn_agent import DQNAgent
from collections import deque
import numpy as np
import time
import os

# --- Các tham số huấn luyện ---
NUM_EPISODES = 2000 
TARGET_UPDATE_FREQ = 10
LEARN_EVERY_N_STEPS = 4
PRINT_EVERY = 10

# --- Tên file model ---
BEST_MODEL_SAVE_PATH = 'dqn_rapid_roll_best.pth'
FINAL_MODEL_SAVE_PATH = 'dqn_rapid_roll_final.pth'

# --- Thiết lập môi trường và agent ---
env = RapidRollEnv(headless=True, jump_strength=0.0)
state_size = 6
action_size = 3
agent = DQNAgent(state_size, action_size)

if os.path.exists(BEST_MODEL_SAVE_PATH):
    print(f"Phát hiện model cũ '{BEST_MODEL_SAVE_PATH}'. Xóa để huấn luyện lại từ đầu.")
    os.remove(BEST_MODEL_SAVE_PATH)
if os.path.exists(FINAL_MODEL_SAVE_PATH):
    os.remove(FINAL_MODEL_SAVE_PATH)

# --- Bắt đầu huấn luyện ---
print(f"Bắt đầu huấn luyện trên thiết bị: {agent.device} ở chế độ HEADLESS...")
print(f"State size: {state_size}, Action size: {action_size}")

scores_window = deque(maxlen=100)
best_avg_score = -float('inf')
total_steps = 0
start_time = time.time()

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(3000): # Giới hạn số bước
        total_steps += 1
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, next_state, reward)
        state = next_state
        total_reward += reward
        
        if total_steps % LEARN_EVERY_N_STEPS == 0:
            agent.learn()
        
        if done:
            break

    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_net()
    
    agent.update_epsilon()
    
    scores_window.append(total_reward)
    current_avg_score = np.mean(scores_window)

    if len(scores_window) == 100 and current_avg_score > best_avg_score:
        best_avg_score = current_avg_score
        torch.save(agent.policy_net.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"\n--- Episode {episode}: KỶ LỤC MỚI! Điểm TB: {current_avg_score:.2f}. Đã lưu model. ---\n")

    if episode % PRINT_EVERY == 0:
        elapsed_time = time.time() - start_time
        eps_per_sec = episode / elapsed_time if elapsed_time > 0 else 0
        print(f"E: {episode}/{NUM_EPISODES} | Avg Score: {current_avg_score:.2f} | Best Avg: {best_avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | Steps: {total_steps} | Speed: {eps_per_sec:.2f} eps/s")


# --- Kết thúc và lưu model cuối cùng ---
total_training_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất trong {total_training_time / 60:.2f} phút.")
torch.save(agent.policy_net.state_dict(), FINAL_MODEL_SAVE_PATH)
print(f"Model cuối cùng đã được lưu tại: {FINAL_MODEL_SAVE_PATH}")
print(f"Model tốt nhất trong quá trình huấn luyện được lưu tại: {BEST_MODEL_SAVE_PATH}")
print(f"Điểm số trung bình cao nhất đạt được: {best_avg_score:.2f}")