import pygame as pygame

def create_circle(flag,w=500,h=400):

    surf = pygame.Surface((w, h))
    c = (0, 255, 0) if flag else (0, 0, 0)
    surf.fill(c)
    r = pygame.draw.circle(surf, c, (200,200), 10)

    return surf