import numpy as np
import pygame
import time


def verlet_step(pos, vel, acc, dt):
    pos += vel * dt + 0.5 * acc * (dt * dt)
    vel += acc * dt


def apply_constraints(pos, n, rest_len):
    for i in range(n - 1):
        dx = pos[0, i + 1] - pos[0, i]
        dy = pos[1, i + 1] - pos[1, i]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            diff = (rest_len - dist) / dist
            offset_x = dx * diff * 0.5
            offset_y = dy * diff * 0.5
            if i > 0:
                pos[0, i] -= offset_x
                pos[1, i] -= offset_y
            pos[0, i + 1] += offset_x
            pos[1, i + 1] += offset_y


STRESS_MODE = True
N_LINKS = 80 if STRESS_MODE else 10
REST_LEN = 15.0
WIDTH, HEIGHT = 800, 600
DT = 0.2
CONSTRAINT_PASSES = 20 if STRESS_MODE else 5
TARGET_FPS = 0  # 0 = uncapped

pos = np.asfortranarray(np.zeros((2, N_LINKS)), dtype=np.float64)
pos[1, :] = np.arange(N_LINKS) * REST_LEN + 100
pos[0, :] = WIDTH // 2

vel = np.asfortranarray(np.zeros((2, N_LINKS)), dtype=np.float64)
acc = np.asfortranarray(np.tile([[0.0], [0.8]], (1, N_LINKS)), dtype=np.float64)
acc[:, 0] = 0.0  # anchor node is pinned — gravity on it causes drift

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
physics_ms_ema = None
frame_ms_ema = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_pos()
    pos[0, 0], pos[1, 0] = float(mx), float(my)
    vel[:, 0] = 0

    physics_start = time.perf_counter()
    verlet_step(pos, vel, acc, DT)

    for _ in range(CONSTRAINT_PASSES):
        apply_constraints(pos, N_LINKS, REST_LEN)

    # apply_constraints corrects positions only, not velocities.
    # Without damping, stretching velocity accumulates each frame.
    vel *= 0.98
    vel[:, 0] = 0  # re-zero anchor velocity after damping
    physics_ms = (time.perf_counter() - physics_start) * 1000.0
    if physics_ms_ema is None:
        physics_ms_ema = physics_ms
    else:
        physics_ms_ema = (0.9 * physics_ms_ema) + (0.1 * physics_ms)

    screen.fill((20, 20, 30))

    points = pos.T.astype(int)
    if N_LINKS > 1:
        pygame.draw.lines(screen, (200, 200, 200), False, points, 2)

    for i in range(N_LINKS):
        pygame.draw.circle(screen, (255, 100, 0), points[i], 5)

    frame_ms = clock.get_time()
    if frame_ms_ema is None:
        frame_ms_ema = float(frame_ms)
    else:
        frame_ms_ema = (0.9 * frame_ms_ema) + (0.1 * float(frame_ms))
    fps = clock.get_fps()
    pygame.display.set_caption(
        f"chain.py | Pure Python/NumPy | links: {N_LINKS} | solve: {physics_ms_ema:.2f} ms | frame: {frame_ms_ema:.2f} ms | FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
