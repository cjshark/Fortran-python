# import os
# import numpy as np
# import pygame
# import random

# # 1. Tell Python where the Fortran DLLs live (adjust path to your GCC install)
# os.add_dll_directory(r"D:\TDM-GCC-64\bin")
# import phys_engine

# # 2. Constants
# N_PARTICLES = 2000
# WIDTH, HEIGHT = 1270, 720
# DT = 0.1

# # 3. Initialize arrays in (2, N) shape to match Fortran column-major layout
# #    Row 0 = X coordinates, Row 1 = Y coordinates
# pos = np.asfortranarray(
#     np.random.rand(2, N_PARTICLES) * np.array([[WIDTH], [HEIGHT]]),
#     dtype=np.float64
# )
# vel = np.asfortranarray(
#     (np.random.rand(2, N_PARTICLES) - 0.5) * 50.0,
#     dtype=np.float64
# )
# # Gravity only on Y axis (axis 1 = row index 1)
# acc = np.asfortranarray(
#     np.tile([[0.0], [0.5]], (1, N_PARTICLES)),
#     dtype=np.float64
# )

# # --- DIAGNOSTIC: Verify Fortran is mutating arrays correctly ---
# test_pos = np.ascontiguousarray([[100.0], [100.0]], dtype=np.float64)
# test_vel = np.ascontiguousarray([[1.0],   [0.0]],   dtype=np.float64)
# test_acc = np.ascontiguousarray([[0.0],   [0.5]],   dtype=np.float64)

# #print("Before:", test_pos.T)
# phys_engine.verlet_step(test_pos, test_vel, test_acc, 0.1, 1)
# print(f"Particle 0 position: {pos[:, 0]}")
# #print("After: ", test_pos.T)

# # Expected: X should shift by ~0.1, Y should shift slightly due to gravity
# # If Before == After, Fortran is NOT mutating the array (pass-by-value bug)

# # 4. Pygame setup
# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Verlet Particle Simulation")
# clock = pygame.time.Clock()

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     #print("Before:", test_pos.T)
#     #phys_engine.verlet_step(test_pos, test_vel, test_acc, 0.1, 1)
#     #print("After: ", test_pos.T)
#     # --- PHYSICS STEP (Fortran Verlet) ---
#     phys_engine.verlet_step(pos, vel, acc, DT, N_PARTICLES)

#     RADIUS = 4.0
#     phys_engine.resolve_collisions(pos=pos, vel=vel, n=N_PARTICLES, radius=RADIUS)

#     # --- BOUNDARY: Floor bounce ---
#     mask_floor = pos[1, :] >= HEIGHT
#     vel[1, mask_floor] *= -0.9          # Bounce with energy loss
#     pos[1, mask_floor] = HEIGHT - 1.0

#     # --- BOUNDARY: Ceiling bounce ---
#     mask_ceil = pos[1, :] < 0
#     vel[1, mask_ceil] *= -0.9
#     pos[1, mask_ceil] = 1.0

#     # --- BOUNDARY: Left/Right wall bounce ---
#     mask_right = pos[0, :] >= WIDTH
#     vel[0, mask_right] *= -0.9
#     pos[0, mask_right] = WIDTH - 1.0

#     mask_left = pos[0, :] < 0
#     vel[0, mask_left] *= -0.9
#     pos[0, mask_left] = 1.0

#     # --- DRAWING ---
#     screen.fill((30, 30, 30))
#     for i in range(N_PARTICLES):
#         x = int(round(pos[0, i]))
#         y = int(round(pos[1, i]))
#         # Clamp to screen bounds to avoid pygame crash
#         x = max(0, min(WIDTH - 1, x))
#         y = max(0, min(HEIGHT - 1, y))
#         pygame.draw.circle(screen, (0, 255, 125), (x, y), 4)

#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()

import os
import numpy as np
import pygame
import random

# 1. DLL Setup
os.add_dll_directory(r"D:\TDM-GCC-64\bin")
import phys_engine

# 2. Constants
N_PARTICLES = 5000
WIDTH, HEIGHT = 1280, 720
DT = 0.1
RADIUS = 4.0

class Particle:
    def __init__(self, index, pos_array, vel_array, acc_array):
        self.index = index
        # These are references to the master arrays
        self.pos_view = pos_array[:, index]
        self.vel_view = vel_array[:, index]
        self.acc_view = acc_array[:, index]
        
        # Unique property not used by Fortran
        self.color = (random.randint(50, 255), 
                      random.randint(50, 255), 
                      random.randint(50, 255))

    def draw(self, surface):
        # We pull the current values directly from the array view
        x = int(max(0, min(WIDTH - 1, self.pos_view[0])))
        y = int(max(0, min(HEIGHT - 1, self.pos_view[1])))
        pygame.draw.circle(surface, self.color, (x, y), int(RADIUS))

# 3. Initialize Master Arrays (Fortran-order is key!)
pos_master = np.asfortranarray(np.random.rand(2, N_PARTICLES) * [[WIDTH], [HEIGHT]], dtype=np.float64)
vel_master = np.asfortranarray((np.random.rand(2, N_PARTICLES) - 0.5) * 10.0, dtype=np.float64)
acc_master = np.asfortranarray(np.tile([[0.0], [0.5]], (1, N_PARTICLES)), dtype=np.float64)

# 4. Create Object List
particles = [Particle(i, pos_master, vel_master, acc_master) for i in range(N_PARTICLES)]

# 5. Pygame Loop
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- PHYSICS STEP (Fast Fortran using Master Arrays) ---
    phys_engine.verlet_step(pos=pos_master, vel=vel_master, acc=acc_master, dt=DT, n=N_PARTICLES)
    phys_engine.resolve_collisions(pos=pos_master, vel=vel_master, n=N_PARTICLES, radius=RADIUS)

    # --- BOUNDARIES (Still fast via NumPy vectorization) ---
    # Floor bounce
    mask_floor = pos_master[1, :] >= HEIGHT
    vel_master[1, mask_floor] *= -0.9
    pos_master[1, mask_floor] = HEIGHT - 1.0
    
    # Left/Right Wall bounce
    mask_walls = (pos_master[0, :] >= WIDTH) | (pos_master[0, :] <= 0)
    vel_master[0, mask_walls] *= -0.9
    pos_master[0, mask_walls] = np.clip(pos_master[0, mask_walls], 1.0, WIDTH - 1.0)

    # --- DRAWING (Individual Objects) ---
    screen.fill((30, 30, 30))
    for p in particles:
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()