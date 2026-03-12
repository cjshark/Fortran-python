import numpy as np
import pygame
import random


def verlet_step(pos, vel, acc, dt):
    pos += vel * dt + 0.5 * acc * (dt * dt)
    vel += acc * dt


def resolve_collisions(pos, vel, n, radius):
    if n < 2:
        return
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

        verlet_step(active_pos, active_vel, active_acc, DT)
        resolve_collisions(active_pos, active_vel, active_count, RADIUS)

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
        f"emitted_particles.py  —  Pure Python/NumPy  |  particles: {active_count}  |  FPS: {fps:.1f}")
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
