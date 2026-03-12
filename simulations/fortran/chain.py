import os
import sys
import time
from pathlib import Path
import numpy as np
import pygame

# Locate project root (two levels up from this file) and add build/ to sys.path
ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
sys.path.insert(0, str(BUILD))

if hasattr(os, "add_dll_directory"):
    for dll_dir in (BUILD, BUILD / "physics_engine" / ".libs"):
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
import phys_engine

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
    phys_engine.verlet_step(pos=pos, vel=vel, acc=acc, dt=DT, n=N_LINKS)

    for _ in range(CONSTRAINT_PASSES):
        phys_engine.apply_constraints(pos=pos, n=N_LINKS, rest_len=REST_LEN)

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
    physics_pct = (physics_ms_ema / max(frame_ms_ema, 1e-6)) * 100.0
    pygame.display.set_caption(
        f"chain.py | Fortran | links: {N_LINKS} | solve: {physics_ms_ema:.2f} ms | frame: {frame_ms_ema:.2f} ms | physics: {physics_pct:.0f}% | FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
