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
PLANET_OPTIONS = (10000, 50000, 100000)
DEFAULT_PLANET_INDEX = 2 if STRESS_MODE else 0
WIDTH, HEIGHT = 800, 800
G_CONST = 5000.0
DT = 0.1
TARGET_FPS = 0  # 0 = uncapped

MAX_PLANETS = max(PLANET_OPTIONS)
pos = np.asfortranarray(np.zeros((2, MAX_PLANETS)), dtype=np.float64)
vel = np.asfortranarray(np.zeros((2, MAX_PLANETS)), dtype=np.float64)
acc = np.asfortranarray(np.zeros((2, MAX_PLANETS)), dtype=np.float64)

def seed_orbits(pos_arr, vel_arr, sun_x, sun_y, gravity_const, count):
    pos_arr[:, :count] = 0.0
    vel_arr[:, :count] = 0.0
    for i in range(count):
        r = random.uniform(100, 350)
        angle = random.uniform(0, 2 * np.pi)
        pos_arr[0, i] = sun_x + r * np.cos(angle)
        pos_arr[1, i] = sun_y + r * np.sin(angle)
        v_mag = np.sqrt(gravity_const / r) * random.uniform(0.9, 1.1)
        vel_arr[0, i] = -v_mag * np.sin(angle)
        vel_arr[1, i] = v_mag * np.cos(angle)


sun_x, sun_y = float(WIDTH // 2), float(HEIGHT // 2)
active_planets = PLANET_OPTIONS[DEFAULT_PLANET_INDEX]
paused = False
show_help = True
show_perf_meter = True
seed_orbits(pos, vel, sun_x, sun_y, G_CONST, active_planets)

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
                seed_orbits(pos, vel, sun_x, sun_y, G_CONST, active_planets)
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
            elif event.key == pygame.K_v:
                show_perf_meter = not show_perf_meter
            elif event.key == pygame.K_1:
                active_planets = PLANET_OPTIONS[0]
                seed_orbits(pos, vel, sun_x, sun_y, G_CONST, active_planets)
            elif event.key == pygame.K_2:
                active_planets = PLANET_OPTIONS[1]
                seed_orbits(pos, vel, sun_x, sun_y, G_CONST, active_planets)
            elif event.key == pygame.K_3:
                active_planets = PLANET_OPTIONS[2]
                seed_orbits(pos, vel, sun_x, sun_y, G_CONST, active_planets)

    physics_start = time.perf_counter()
    if not paused:
        active_pos = pos[:, :active_planets]
        active_vel = vel[:, :active_planets]
        active_acc = acc[:, :active_planets]
        apply_orbital_gravity(active_pos, active_acc, active_planets, sun_x, sun_y, G_CONST)
        verlet_step(active_pos, active_vel, active_acc, DT)
    physics_ms = (time.perf_counter() - physics_start) * 1000.0
    if physics_ms_ema is None:
        physics_ms_ema = physics_ms
    else:
        physics_ms_ema = (0.9 * physics_ms_ema) + (0.1 * physics_ms)

    screen.fill((5, 5, 15))
    pygame.draw.circle(screen, (255, 200, 0), (int(sun_x), int(sun_y)), 20)

    for i in range(active_planets):
        px, py = int(pos[0, i]), int(pos[1, i])
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pygame.draw.circle(screen, (100, 200, 255), (px, py), 2)

    if show_help:
        help_lines = [
            "LMB: move sun | Wheel/Up/Down: gravity",
            "Left/Right: dt | 1/2/3: 10k/50k/100k",
            "Space: pause | R: reseed | V: perf meter | H: hide help",
        ]
        for idx, line in enumerate(help_lines):
            screen.blit(font.render(line, True, (220, 220, 220)), (12, 12 + idx * 20))

    if show_perf_meter:
        budget_ratio = physics_ms_ema / 16.67
        if budget_ratio < 0.25:
            perf_label, perf_color = "VERY FAST", (120, 255, 140)
        elif budget_ratio < 0.60:
            perf_label, perf_color = "GOOD", (180, 255, 120)
        elif budget_ratio < 1.00:
            perf_label, perf_color = "OK", (255, 220, 120)
        else:
            perf_label, perf_color = "HEAVY", (255, 120, 120)

        meter_x, meter_y = 12, HEIGHT - 30
        meter_w, meter_h = 260, 12
        fill_ratio = float(np.clip(budget_ratio, 0.0, 1.5)) / 1.5
        pygame.draw.rect(screen, (50, 50, 60), (meter_x, meter_y, meter_w, meter_h), border_radius=6)
        pygame.draw.rect(screen, perf_color, (meter_x, meter_y, int(meter_w * fill_ratio), meter_h), border_radius=6)
        perf_text = f"Pure Python: {perf_label} | planets: {active_planets:,} | {physics_ms_ema:.3f}ms"
        screen.blit(font.render(perf_text, True, perf_color), (12, HEIGHT - 54))

    frame_ms = clock.get_time()
    if frame_ms_ema is None:
        frame_ms_ema = float(frame_ms)
    else:
        frame_ms_ema = (0.9 * frame_ms_ema) + (0.1 * float(frame_ms))
    fps = clock.get_fps()
    pygame.display.set_caption(
        f"orbit | Pure | planets: {active_planets:,} | solve: {physics_ms_ema:.2f}ms | frame: {frame_ms_ema:.2f}ms | FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(TARGET_FPS)
