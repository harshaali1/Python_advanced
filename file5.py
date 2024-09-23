import pygame
import colorsys

width, height = 800, 600
scale = 200
max_iter = 255
x_offset, y_offset = -0.5, 0.0

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Set")

def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z**2 + c
        n += 1
    return n

def map_to_color(n):
    if n == max_iter:
        return (0, 0, 0)
    hue = int(n / max_iter * 360)
    saturation = 1
    value = 1 if n < max_iter else 0
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                scale *= 1.1
            elif event.button == 5:  # Scroll down
                scale /= 1.1
        if event.type == pygame.MOUSEMOTION and event.buttons[2]:  # Right click drag
            x_offset -= event.rel[0] / scale
            y_offset -= event.rel[1] / scale

    for x in range(width):
        for y in range(height):
            c = complex((x - width / 2) / scale + x_offset, (y - height / 2) / scale + y_offset)
            n = mandelbrot(c)
            color = map_to_color(n)
            screen.set_at((x, y), color)

    pygame.display.flip()

pygame.quit()
