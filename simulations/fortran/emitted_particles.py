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

MAX_PARTICLES = 5000
WIDTH, HEIGHT = 800, 600
DT = 0.1
RADIUS = 4.0


class Particle:
    def __init__(self, index, pos_array, vel_array, acc_array):
        self.index = index
        self.pos_view = pos_array[:, index]
        self.vel_view = vel_array[:, index]
        self.acc_view = acc_array[:, index]
        self.color = (random.randint(100, 255), random.randint(150, 255), 255)

    def draw(self, surface):
        x, y = int(self.pos_view[0]), int(self.pos_view[1])
        pygame.draw.circle(surface, self.color, (x, y), int(RADIUS))


pos_master = np.asfortranarray(np.zeros((2, MAX_PARTICLES)), dtype=np.float64)
vel_master = np.asfortranarray(np.zeros((2, MAX_PARTICLES)), dtype=np.float64)
acc_master = np.asfortranarray(
    np.tile([[0.0], [0.5]], (1, MAX_PARTICLES)), dtype=np.float64)

particles = []
active_count = 0

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

spawn_point = (WIDTH // 2, 100)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for _ in range(2):
        if active_count < MAX_PARTICLES:
            pos_master[0, active_count] = spawn_point[0]
            pos_master[1, active_count] = spawn_point[1]
            vel_master[0, active_count] = random.uniform(-5, 5)
            vel_master[1, active_count] = random.uniform(-2, 2)
            particles.append(Particle(active_count, pos_master, vel_master, acc_master))
            active_count += 1

    if active_count > 0:
        active_pos = pos_master[:, :active_count]
        active_vel = vel_master[:, :active_count]
        active_acc = acc_master[:, :active_count]

        phys_engine.verlet_step(pos=active_pos, vel=active_vel, acc=active_acc, dt=DT, n=active_count)
        phys_engine.resolve_collisions(pos=active_pos, vel=active_vel, n=active_count, radius=RADIUS)

        mask_floor = active_pos[1, :] >= HEIGHT
        active_vel[1, mask_floor] *= -0.7
        active_pos[1, mask_floor] = HEIGHT - 1.0

        mask_walls = (active_pos[0, :] >= WIDTH) | (active_pos[0, :] <= 0)
        active_vel[0, mask_walls] *= -0.8
        active_pos[0, mask_walls] = np.clip(active_pos[0, mask_walls], 1.0, WIDTH - 1.0)

    screen.fill((20, 20, 25))
    pygame.draw.circle(screen, (255, 255, 255), spawn_point, 10, 2)
    for p in particles:
        p.draw(screen)

    fps = clock.get_fps()
    pygame.display.set_caption(
        f"emitted_particles.py  —  Fortran  |  particles: {active_count}  |  FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
