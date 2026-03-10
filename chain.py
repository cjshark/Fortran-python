import os
import numpy as np
import pygame

os.add_dll_directory(r"D:\TDM-GCC-64\bin")
import phys_engine

# --- Setup ---
N_LINKS = 10
REST_LEN = 15.0
WIDTH, HEIGHT = 800, 600
DT = 0.2

# All links start in a vertical line
pos = np.asfortranarray(np.zeros((2, N_LINKS)), dtype=np.float64)
pos[1, :] = np.arange(N_LINKS) * REST_LEN + 100
pos[0, :] = WIDTH // 2

vel = np.asfortranarray(np.zeros((2, N_LINKS)), dtype=np.float64)
acc = np.asfortranarray(np.tile([[0.0], [0.8]], (1, N_LINKS)), dtype=np.float64)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. Anchor the first link to the mouse
    mx, my = pygame.mouse.get_pos()
    pos[0, 0], pos[1, 0] = float(mx), float(my)
    vel[:, 0] = 0 # Stop the anchor from moving via physics

    # 2. Physics Step
    phys_engine.verlet_step(pos=pos, vel=vel, acc=acc, dt=DT, n=N_LINKS)
    
    # 3. Apply Constraints (Repeat 5x for extra "stiffness")
    for _ in range(5):
        phys_engine.apply_constraints(pos=pos, n=N_LINKS, rest_len=REST_LEN)

    # 4. Draw
    screen.fill((20, 20, 30))
    
    # Draw the lines connecting the chain
    points = pos.T.astype(int)
    if N_LINKS > 1:
        pygame.draw.lines(screen, (200, 200, 200), False, points, 2)
    
    # Draw the joints
    for i in range(N_LINKS):
        pygame.draw.circle(screen, (255, 100, 0), points[i], 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()