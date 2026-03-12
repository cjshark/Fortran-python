import os
import sys
from pathlib import Path
import numpy as np
import pygame
import random

# Locate project root (two levels up from this file) and add build/ to sys.path
ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
sys.path.insert(0, str(BUILD))

if hasattr(os, "add_dll_directory"):
    for dll_dir in (BUILD, BUILD / "physics_engine" / ".libs"):
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
import phys_engine

N_PARTICLES = 5000
WIDTH, HEIGHT = 1280, 720
DT = 0.1
RADIUS = 4.0


class Particle:
    def __init__(self, index, pos_array, vel_array, acc_array):
        self.index = index
        self.pos_view = pos_array[:, index]
        self.vel_view = vel_array[:, index]
        self.acc_view = acc_array[:, index]
        self.color = (random.randint(50, 255),
                      random.randint(50, 255),
                      random.randint(50, 255))

    def draw(self, surface):
        x = int(max(0, min(WIDTH - 1,  self.pos_view[0])))
        y = int(max(0, min(HEIGHT - 1, self.pos_view[1])))
        pygame.draw.circle(surface, self.color, (x, y), int(RADIUS))


pos_master = np.asfortranarray(
    np.random.rand(2, N_PARTICLES) * [[WIDTH], [HEIGHT]], dtype=np.float64)
vel_master = np.asfortranarray(
    (np.random.rand(2, N_PARTICLES) - 0.5) * 10.0, dtype=np.float64)
acc_master = np.asfortranarray(
    np.tile([[0.0], [0.5]], (1, N_PARTICLES)), dtype=np.float64)

particles = [Particle(i, pos_master, vel_master, acc_master)
             for i in range(N_PARTICLES)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    phys_engine.verlet_step(pos=pos_master, vel=vel_master, acc=acc_master, dt=DT, n=N_PARTICLES)
    phys_engine.resolve_collisions(pos=pos_master, vel=vel_master, n=N_PARTICLES, radius=RADIUS)

    mask_floor = pos_master[1, :] >= HEIGHT
    vel_master[1, mask_floor] *= -0.9
    pos_master[1, mask_floor] = HEIGHT - 1.0

    mask_walls = (pos_master[0, :] >= WIDTH) | (pos_master[0, :] <= 0)
    vel_master[0, mask_walls] *= -0.9
    pos_master[0, mask_walls] = np.clip(pos_master[0, mask_walls], 1.0, WIDTH - 1.0)

    screen.fill((30, 30, 30))
    for p in particles:
        p.draw(screen)

    fps = clock.get_fps()
    pygame.display.set_caption(f"particles.py  —  Fortran  |  FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
