# main.py

import pygame
import torch
import sys
import os
from rapid_roll_env import RapidRollEnv, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, YELLOW
from dqn_agent import DQNAgent

# --- Các hằng số và thiết lập ---
MODEL_PATH = 'dqn_rapid_roll_best.pth'
STATE_SIZE = 6
ACTION_SIZE = 3
FPS = 60

# --- Các hàm UI (không thay đổi) ---
def draw_button(text, font, text_color, rect, surface, hover_color=None):
    mx, my = pygame.mouse.get_pos()
    is_hovering = rect.collidepoint((mx, my))
    bg_color = hover_color if is_hovering and hover_color else BLACK
    pygame.draw.rect(surface, bg_color, rect, border_radius=8)
    pygame.draw.rect(surface, WHITE, rect, 2, border_radius=8)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)
    return is_hovering

def draw_text(text, font, color, surface, x, y, center=False):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    if center: textrect.center = (x, y)
    else: textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def main_menu(screen, fonts, high_score, game_settings):
    btn_w, btn_h = 220, 50
    btn_x = SCREEN_WIDTH / 2 - btn_w / 2
    button_new_game = pygame.Rect(btn_x, 250, btn_w, btn_h)
    button_toggle_mode = pygame.Rect(btn_x, 310, btn_w, btn_h)
    button_exit = pygame.Rect(btn_x, 370, btn_w, btn_h)
    mode_text = "Mode: AI" if game_settings['ai_mode'] else "Mode: Human"
    
    running = True
    while running:
        screen.fill(BLACK)
        draw_text('Rapid Roll AI', fonts['title'], WHITE, screen, SCREEN_WIDTH / 2, 100, center=True)
        draw_text(f"High Score: {high_score}", fonts['main'], WHITE, screen, SCREEN_WIDTH / 2, 180, center=True)
        
        is_hover_new = draw_button('New Game', fonts['main'], WHITE, button_new_game, screen, (50, 50, 50))
        is_hover_mode = draw_button(mode_text, fonts['main'], WHITE, button_toggle_mode, screen, (50, 50, 50))
        is_hover_exit = draw_button('Exit', fonts['main'], WHITE, button_exit, screen, (50, 50, 50))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 'EXIT'
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_hover_new: return 'PLAYING'
                if is_hover_mode: return 'TOGGLE_AI'
                if is_hover_exit: return 'EXIT'
        
        pygame.display.update()
        pygame.time.Clock().tick(FPS)

def game_over_screen(screen, fonts, last_score, high_score):
    running = True
    while running:
        screen.fill(BLACK)
        draw_text('GAME OVER', fonts['title'], (255, 0, 0), screen, SCREEN_WIDTH/2, 180, center=True)
        draw_text(f'Score: {last_score}', fonts['main'], WHITE, screen, SCREEN_WIDTH/2, 280, center=True)
        draw_text(f'High Score: {high_score}', fonts['main'], WHITE, screen, SCREEN_WIDTH/2, 330, center=True)
        draw_text('Click or Press Any Key to Continue', fonts['small'], WHITE, screen, SCREEN_WIDTH/2, 450, center=True)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 'EXIT'
            if event.type in [pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
                return 'MAIN_MENU'
        pygame.display.update()
        pygame.time.Clock().tick(FPS)

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Rapid Roll AI")
    clock = pygame.time.Clock()
    fonts = { 'small': pygame.font.Font(None, 30), 'main': pygame.font.Font(None, 45), 'title': pygame.font.Font(None, 70) }

    # --- Tải AI Agent ---
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    ai_available = False
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
            agent.policy_net.eval()
            agent.epsilon = 0.0
            ai_available = True
            print("Tải model AI thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            print("Model có thể không tương thích. Hãy huấn luyện lại. Chế độ AI không khả dụng.")
    else:
        print(f"CẢNH BÁO: Không tìm thấy file model tại '{MODEL_PATH}'. Chế độ AI không khả dụng.")

    game_state = 'MAIN_MENU'
    high_score = 0
    game_settings = { 'ai_mode': ai_available }

    while True:
        if game_state == 'MAIN_MENU':
            result = main_menu(screen, fonts, high_score, game_settings)
            if result == 'TOGGLE_AI':
                if ai_available: game_settings['ai_mode'] = not game_settings['ai_mode']
            elif result == 'EXIT': break
            else: game_state = result
        
        elif game_state == 'PLAYING':
            env = RapidRollEnv(jump_strength=0.0)
            state = env.reset()
            game_mode = 'AI' if game_settings['ai_mode'] else 'HUMAN'
            
            playing = True
            while playing:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                action = 1
                if game_mode == 'AI':
                    action = agent.choose_action(state)
                else:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT] or keys[pygame.K_a]: action = 0
                    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action = 2
                
                next_state, _, done, _ = env.step(action)
                state = next_state
                env.render()
                draw_text(f"Mode: {game_mode}", fonts['small'], WHITE, screen, SCREEN_WIDTH - 100, 10)
                pygame.display.flip()
                clock.tick(FPS)
                if done: playing = False

            last_score = env.score
            if last_score > high_score: high_score = last_score
            game_state = 'GAME_OVER'

        elif game_state == 'GAME_OVER':
            result = game_over_screen(screen, fonts, last_score, high_score)
            if result == 'EXIT': break
            else: game_state = result

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()