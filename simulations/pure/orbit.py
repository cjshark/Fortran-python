import numpy as np
import pygame
import random
import time


def verlet_step(pos, vel, acc, dt):
    pos += vel * dt + 0.5 * acc * (dt * dt)
    vel += acc * dt


def apply_orbital_gravity(pos, acc, n, sun_x, sun_y, gravity_const):
    dx = sun_x - pos[0]
    dy = sun_y - pos[1]
    dist_sq = dx * dx + dy * dy

    safe = dist_sq > 100.0
    dist = np.where(safe, np.sqrt(np.maximum(dist_sq, 1e-10)), 1.0)
    force_mag = np.where(safe, gravity_const / dist_sq, 0.0)

    acc[0] = force_mag * (dx / dist)
    acc[1] = force_mag * (dy / dist)


STRESS_MODE = True
N_PLANETS = 4000 if STRESS_MODE else 100
WIDTH, HEIGHT = 800, 800
G_CONST = 5000.0
DT = 0.1
INITIAL_DRAW_STRIDE = 4 if STRESS_MODE else 1
TARGET_FPS = 0  # 0 = uncapped

pos = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)
vel = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)
acc = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)

def seed_orbits(pos_arr, vel_arr, sun_x, sun_y, gravity_const):
    for i in range(N_PLANETS):
        r = random.uniform(100, 350)
        angle = random.uniform(0, 2 * np.pi)
        pos_arr[0, i] = sun_x + r * np.cos(angle)
        pos_arr[1, i] = sun_y + r * np.sin(angle)
        v_mag = np.sqrt(gravity_const / r) * random.uniform(0.9, 1.1)
        vel_arr[0, i] = -v_mag * np.sin(angle)
        vel_arr[1, i] = v_mag * np.cos(angle)


sun_x, sun_y = float(WIDTH // 2), float(HEIGHT // 2)
draw_stride = INITIAL_DRAW_STRIDE
paused = False
show_help = True
seed_orbits(pos, vel, sun_x, sun_y, G_CONST)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
physics_ms_ema = None
frame_ms_ema = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            sun_x, sun_y = float(event.pos[0]), float(event.pos[1])
        if event.type == pygame.MOUSEWHEEL:
            G_CONST = float(np.clip(G_CONST * (1.0 + 0.1 * event.y), 100.0, 100000.0))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                seed_orbits(pos, vel, sun_x, sun_y, G_CONST)
            elif event.key == pygame.K_h:
                show_help = not show_help
            elif event.key == pygame.K_UP:
                G_CONST = float(np.clip(G_CONST * 1.1, 100.0, 100000.0))
            elif event.key == pygame.K_DOWN:
                G_CONST = float(np.clip(G_CONST / 1.1, 100.0, 100000.0))
            elif event.key == pygame.K_RIGHT:
                DT = float(np.clip(DT * 1.1, 0.005, 1.0))
            elif event.key == pygame.K_LEFT:
                DT = float(np.clip(DT / 1.1, 0.005, 1.0))
            elif event.key == pygame.K_PAGEUP:
                draw_stride = min(64, draw_stride + 1)
            elif event.key == pygame.K_PAGEDOWN:
                draw_stride = max(1, draw_stride - 1)

    physics_start = time.perf_counter()
    if not paused:
        apply_orbital_gravity(pos, acc, N_PLANETS, sun_x, sun_y, G_CONST)
        verlet_step(pos, vel, acc, DT)
    physics_ms = (time.perf_counter() - physics_start) * 1000.0
    if physics_ms_ema is None:
        physics_ms_ema = physics_ms
    else:
        physics_ms_ema = (0.9 * physics_ms_ema) + (0.1 * physics_ms)

    screen.fill((5, 5, 15))
    pygame.draw.circle(screen, (255, 200, 0), (int(sun_x), int(sun_y)), 20)

    for i in range(0, N_PLANETS, draw_stride):
        px, py = int(pos[0, i]), int(pos[1, i])
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pygame.draw.circle(screen, (100, 200, 255), (px, py), 2)

    if show_help:
        help_lines = [
            "LMB: move sun | Wheel/Up/Down: gravity",
            "Left/Right: dt | PgUp/PgDn: draw stride",
            "Space: pause | R: reseed | H: hide help",
        ]
        for idx, line in enumerate(help_lines):
            screen.blit(font.render(line, True, (220, 220, 220)), (12, 12 + idx * 20))

    frame_ms = clock.get_time()
    if frame_ms_ema is None:
        frame_ms_ema = float(frame_ms)
    else:
        frame_ms_ema = (0.9 * frame_ms_ema) + (0.1 * float(frame_ms))
    fps = clock.get_fps()
    physics_pct = (physics_ms_ema / max(frame_ms_ema, 1e-6)) * 100.0
    pygame.display.set_caption(
        f"orbit | Pure | solve: {physics_ms_ema:.2f}ms | frame: {frame_ms_ema:.2f}ms | {physics_pct:.0f}% | FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(TARGET_FPS)
