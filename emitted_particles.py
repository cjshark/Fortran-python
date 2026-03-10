import os
import numpy as np
import pygame
import random

# 1. DLL Setup
os.add_dll_directory(r"D:\TDM-GCC-64\bin")
import phys_engine

# 2. Constants
MAX_PARTICLES = 2000 
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

# 3. Pre-allocate Master Arrays at Max Capacity
pos_master = np.asfortranarray(np.zeros((2, MAX_PARTICLES)), dtype=np.float64)
vel_master = np.asfortranarray(np.zeros((2, MAX_PARTICLES)), dtype=np.float64)
acc_master = np.asfortranarray(np.tile([[0.0], [0.5]], (1, MAX_PARTICLES)), dtype=np.float64)

particles = []
active_count = 0

# 4. Pygame Setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

spawn_point = (WIDTH // 2, 100)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- EMITTER LOGIC: Spawn 2 particles per frame until full ---
    for _ in range(2):
        if active_count < MAX_PARTICLES:
            # Initialize the next slot in the master arrays
            pos_master[0, active_count] = spawn_point[0]
            pos_master[1, active_count] = spawn_point[1]
            
            # Give it some random initial spray
            vel_master[0, active_count] = random.uniform(-5, 5)
            vel_master[1, active_count] = random.uniform(-2, 2)
            
            # Create the object tracker
            particles.append(Particle(active_count, pos_master, vel_master, acc_master))
            active_count += 1

    # --- PHYSICS STEP (Only process the active_count slice!) ---
    if active_count > 0:
        # We slice the arrays [:, :active_count] so Fortran doesn't process 2000 zeros
        active_pos = pos_master[:, :active_count]
        active_vel = vel_master[:, :active_count]
        active_acc = acc_master[:, :active_count]

        phys_engine.verlet_step(pos=active_pos, vel=active_vel, acc=active_acc, dt=DT, n=active_count)
        
        # Note: Collisions get heavy as active_count grows
        phys_engine.resolve_collisions(pos=active_pos, vel=active_vel, n=active_count, radius=RADIUS)

        # --- BOUNDARIES (Applied to active slice) ---
        mask_floor = active_pos[1, :] >= HEIGHT
        active_vel[1, mask_floor] *= -0.7
        active_pos[1, mask_floor] = HEIGHT - 1.0
        
        mask_walls = (active_pos[0, :] >= WIDTH) | (active_pos[0, :] <= 0)
        active_vel[0, mask_walls] *= -0.8
        active_pos[0, mask_walls] = np.clip(active_pos[0, mask_walls], 1.0, WIDTH - 1.0)

    # --- DRAWING ---
    screen.fill((20, 20, 25))
    # Draw spawn point
    pygame.draw.circle(screen, (255, 255, 255), spawn_point, 10, 2)
    
    for p in particles:
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()