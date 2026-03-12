import numpy as np
import pygame
import random


def verlet_step(pos, vel, acc, dt):
    pos += vel * dt + 0.5 * acc * (dt * dt)
    vel += acc * dt


def resolve_collisions(pos, vel, n, radius):
    min_dist_sq = (2.0 * radius) ** 2
    i_idx, j_idx = np.triu_indices(n, k=1)

    dx = pos[0, j_idx] - pos[0, i_idx]
    dy = pos[1, j_idx] - pos[1, i_idx]
    dist_sq = dx * dx + dy * dy

    hit = (dist_sq < min_dist_sq) & (dist_sq > 0)
    if not np.any(hit):
        return

    ci, cj = i_idx[hit], j_idx[hit]
    cdx = pos[0, cj] - pos[0, ci]
    cdy = pos[1, cj] - pos[1, ci]
    inv_dist = 1.0 / np.sqrt(cdx * cdx + cdy * cdy)
    nx = cdx * inv_dist
    ny = cdy * inv_dist

    rvx = vel[0, ci] - vel[0, cj]
    rvy = vel[1, ci] - vel[1, cj]
    dp = rvx * nx + rvy * ny
    dp = np.where(dp > 0, dp, 0.0)

    np.add.at(vel[0], ci, -dp * nx)
    np.add.at(vel[1], ci, -dp * ny)
    np.add.at(vel[0], cj,  dp * nx)
    np.add.at(vel[1], cj,  dp * ny)


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

    verlet_step(pos_master, vel_master, acc_master, DT)
    resolve_collisions(pos_master, vel_master, N_PARTICLES, RADIUS)

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
    pygame.display.set_caption(f"particles.py  —  Pure Python/NumPy  |  FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
