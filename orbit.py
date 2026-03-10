import os
import numpy as np
import pygame
import random

os.add_dll_directory(r"D:\TDM-GCC-64\bin")
import phys_engine

# --- Setup ---
N_PLANETS = 100
WIDTH, HEIGHT = 800, 800
SUN_POS = (WIDTH // 2, HEIGHT // 2)
G_CONST = 5000.0 # Tuning parameter for "strength" of the sun
DT = 0.1

# Arrays
pos = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)
vel = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)
acc = np.asfortranarray(np.zeros((2, N_PLANETS)), dtype=np.float64)

# Initialize Planets in orbit
for i in range(N_PLANETS):
    # Random distance from sun
    r = random.uniform(100, 350)
    angle = random.uniform(0, 2 * np.pi)
    
    pos[0, i] = SUN_POS[0] + r * np.cos(angle)
    pos[1, i] = SUN_POS[1] + r * np.sin(angle)
    
    # Tangential velocity formula: v = sqrt(G/r)
    v_mag = np.sqrt(G_CONST / r) * random.uniform(0.9, 1.1) # vary slightly for ellipses
    vel[0, i] = -v_mag * np.sin(angle)
    vel[1, i] =  v_mag * np.cos(angle)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

while True:
    if any(event.type == pygame.QUIT for event in pygame.event.get()): break

    # 1. Update Acceleration based on Sun's Gravity
    phys_engine.apply_orbital_gravity(pos=pos, acc=acc, n=N_PLANETS, sun_x=float(SUN_POS[0]), sun_y=float(SUN_POS[1]), gravity_const=G_CONST)

    # 2. Verlet Step
    phys_engine.verlet_step(pos, vel, acc, DT, N_PLANETS)

    # 3. Draw
    screen.fill((5, 5, 15))
    # Draw Sun
    pygame.draw.circle(screen, (255, 200, 0), SUN_POS, 20)
    
    for i in range(N_PLANETS):
        px, py = int(pos[0, i]), int(pos[1, i])
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pygame.draw.circle(screen, (100, 200, 255), (px, py), 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()