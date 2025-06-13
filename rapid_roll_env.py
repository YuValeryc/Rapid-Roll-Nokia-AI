# rapid_roll_env.py

import pygame
import random
import numpy as np

# --- Các hằng số ---
SCREEN_WIDTH, SCREEN_HEIGHT = 360, 600
SCALE_FACTOR = SCREEN_HEIGHT / 400.0
BALL_RADIUS = 10 * SCALE_FACTOR
PLATFORM_WIDTH, PLATFORM_HEIGHT = 60 * SCALE_FACTOR, 10 * SCALE_FACTOR
BALL_SPEED = 4.0 * SCALE_FACTOR
GRAVITY = 0.4 * SCALE_FACTOR
SPIKE_CHANCE = 0.2
NUM_SPIKES = 6

# THAY ĐỔI: Hằng số cho tốc độ
INITIAL_SCROLL_SPEED = 1.5 * SCALE_FACTOR
MAX_SCROLL_SPEED = 4.0 * SCALE_FACTOR
SPEED_INCREASE_RATE = 0.0001 * SCALE_FACTOR # Tăng tốc độ cuộn theo thời gian

# --- Màu sắc ---
WHITE, BLACK, RED, BLUE, YELLOW = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)

# --- Lớp Platform---
class Platform(pygame.Rect):
    def __init__(self, x, y, width, height, is_spike=False):
        super().__init__(x, y, width, height)
        self.is_spike = is_spike

class RapidRollEnv:
    def __init__(self, headless=False, jump_strength=0.0):
        self.headless = headless
        self.jump_boost = -jump_strength * SCALE_FACTOR

        if not self.headless:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Rapid Roll AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, int(30 * SCALE_FACTOR))
        self.platforms = []
        self.reset()
    
    def reset(self):
        self.ball_pos = [SCREEN_WIDTH / 2, 100 * SCALE_FACTOR] 
        self.ball_vel = [0, 0]
        # THÊM: Reset tốc độ mỗi khi chơi lại
        self.current_scroll_speed = INITIAL_SCROLL_SPEED
        self._generate_initial_platforms()
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _generate_initial_platforms(self):
        platforms = []
        y = self.ball_pos[1] + 50 * SCALE_FACTOR
        x = self.ball_pos[0] - PLATFORM_WIDTH / 2
        platforms.append(Platform(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT, is_spike=False))
        
        for i in range(1, 15):
            y = platforms[-1].y + random.randint(int(80 * SCALE_FACTOR), int(120 * SCALE_FACTOR))
            x = random.randint(0, int(SCREEN_WIDTH - PLATFORM_WIDTH))
            is_spike = random.random() < SPIKE_CHANCE
            platforms.append(Platform(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT, is_spike=is_spike))
            
        self.platforms = platforms

    # THAY ĐỔI QUAN TRỌNG: Cập nhật state vector (6 chiều)
    def _get_state(self):
        platforms_below = [p for p in self.platforms if p.y > self.ball_pos[1]]
        
        is_next_spike = 0.0
        if platforms_below:
            closest_platform = min(platforms_below, key=lambda p: p.y)
            relative_px = closest_platform.centerx - self.ball_pos[0]
            relative_py = closest_platform.y - self.ball_pos[1]
            if closest_platform.is_spike:
                is_next_spike = 1.0
        else:
            relative_px = (SCREEN_WIDTH / 2) - self.ball_pos[0]
            relative_py = SCREEN_HEIGHT 

        # Chuẩn hóa tốc độ hiện tại để đưa vào state (giá trị từ 0 đến 1)
        normalized_speed = (self.current_scroll_speed - INITIAL_SCROLL_SPEED) / (MAX_SCROLL_SPEED - INITIAL_SCROLL_SPEED)

        state_vector = [
            self.ball_pos[0] / SCREEN_WIDTH,
            self.ball_vel[1] / (10 * SCALE_FACTOR),
            relative_px / (SCREEN_WIDTH / 2),
            relative_py / SCREEN_HEIGHT,
            is_next_spike,
            np.clip(normalized_speed, 0, 1) # Chiều thứ 6: tốc độ hiện tại
        ]
        return np.array(state_vector, dtype=np.float32)

    def step(self, action):
        if self.game_over:
             return self._get_state(), 0, True, {}

        # THAY ĐỔI: Tăng tốc độ cuộn theo thời gian
        if self.current_scroll_speed < MAX_SCROLL_SPEED:
            self.current_scroll_speed += SPEED_INCREASE_RATE

        # Di chuyển và vật lý
        if action == 0: self.ball_pos[0] -= BALL_SPEED
        elif action == 2: self.ball_pos[0] += BALL_SPEED
        self.ball_pos[0] = np.clip(self.ball_pos[0], BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
        self.ball_vel[1] += GRAVITY
        self.ball_pos[1] += self.ball_vel[1]
        # THAY ĐỔI: Sử dụng tốc độ hiện tại
        for p in self.platforms: p.y -= self.current_scroll_speed

        # Logic còn lại của step không thay đổi
        reward = 0.1
        ball_rect = pygame.Rect(self.ball_pos[0] - BALL_RADIUS, self.ball_pos[1] - BALL_RADIUS, BALL_RADIUS*2, BALL_RADIUS*2)
        
        for p in self.platforms:
            if self.ball_vel[1] > 0 and ball_rect.colliderect(p) and abs(ball_rect.bottom - p.top) < self.ball_vel[1] + 2:
                if p.is_spike:
                    self.game_over = True
                    reward = -200
                    return self._get_state(), reward, self.game_over, {}
                else:
                    self.ball_pos[1] = p.top - BALL_RADIUS
                    self.ball_vel[1] = self.jump_boost
                    reward = 10
                    break
        
        old_platform_count = len(self.platforms)
        self.platforms = [p for p in self.platforms if p.bottom > 0]
        platforms_passed = old_platform_count - len(self.platforms)
        if platforms_passed > 0:
            self.score += platforms_passed
            reward += platforms_passed * 2
        if self.ball_pos[0] < SCREEN_WIDTH * 0.1 or self.ball_pos[0] > SCREEN_WIDTH * 0.9:
            reward -= 0.5
            
        while len(self.platforms) < 15:
            last_y = max(p.y for p in self.platforms) if self.platforms else 0
            y = last_y + random.randint(int(80*SCALE_FACTOR), int(120*SCALE_FACTOR))
            x = random.randint(0, int(SCREEN_WIDTH - PLATFORM_WIDTH))
            is_spike = random.random() < SPIKE_CHANCE
            self.platforms.append(Platform(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT, is_spike=is_spike))
        
        if self.ball_pos[1] - BALL_RADIUS > SCREEN_HEIGHT or self.ball_pos[1] + BALL_RADIUS < 0:
            self.game_over = True
            reward = -100

        return self._get_state(), reward, self.game_over, {}
    
    def _draw_spike_platform(self, surface, platform):
        base_height = platform.height * 0.4
        base_rect = pygame.Rect(platform.left, platform.top + platform.height - base_height, platform.width, base_height)
        pygame.draw.rect(surface, BLUE, base_rect)
        spike_width = platform.width / NUM_SPIKES
        spike_height = platform.height * 0.7
        spike_top_y = platform.top + platform.height - base_height - spike_height
        for i in range(NUM_SPIKES):
            p1 = (platform.left + i * spike_width, base_rect.top)
            p2 = (platform.left + (i + 1) * spike_width, base_rect.top)
            p3 = (platform.left + (i + 0.5) * spike_width, spike_top_y)
            pygame.draw.polygon(surface, YELLOW, [p1, p2, p3])

    def render(self):
        if self.headless: return
        self.screen.fill(BLACK)
        
        pygame.draw.circle(self.screen, RED, (int(self.ball_pos[0]), int(self.ball_pos[1])), int(BALL_RADIUS))
        
        for p in self.platforms:
            if p.is_spike:
                self._draw_spike_platform(self.screen, p)
            else:
                pygame.draw.rect(self.screen, BLUE, p)
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # THÊM: Hiển thị tốc độ hiện tại trên màn hình
        speed_text = self.font.render(f"Speed: {self.current_scroll_speed / SCALE_FACTOR:.2f}", True, WHITE)
        self.screen.blit(speed_text, (10, 40))
        
        pygame.display.flip()

    def tick(self, fps):
        if self.headless: return
        self.clock.tick(fps)