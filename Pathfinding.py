import pygame

from Settings import *
from Pathfinding_classes import *


def pathfind_sim() -> None:

    visualizer = Visualizer()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    MODES = {'A*': AStar,
             'Dijkstra': Dijkstra}

    running = True
    while running:
        if not visualizer.alg or visualizer.en_menu:
            visualizer.menu(WIN)

            if visualizer.alg == 'Quit': break
            grid = visualizer.make_grid(MODES[visualizer.alg])

            visualizer.en_menu = 0
            start, end = None, None

        visualizer.draw(WIN, grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                spot = grid[pos[0] // TILESIZE, pos[1] // TILESIZE]
                
                if not start and not spot.is_end():
                    start = spot
                    start.set_start()

                elif not end and not spot.is_start():
                    end = spot
                    end.set_end()

                if spot not in [end, start]:
                    spot.set_wall()

            elif pygame.mouse.get_pressed()[2]: # RMB
                pos = pygame.mouse.get_pos()
                spot = grid[pos[0] // TILESIZE, pos[1] // TILESIZE]
                spot.reset()

                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:

                    for row in grid:
                        for spot in row:
                            spot.get_neighbours(grid)
                            if not (spot.is_wall() or spot.is_start() or spot.is_end()):
                                spot.reset()

                    visualizer.draw(WIN, grid)
                    if visualizer.alg == 'Dijkstra':
                        visualizer.DijkstraAlg(lambda: visualizer.draw(WIN, grid),
                                               start, end)
                    else:
                        visualizer.AStarAlg(lambda: visualizer.draw(WIN, grid),
                                            grid, start, end)

                if event.key == pygame.K_r:
                    start = end = None
                    grid = visualizer.make_grid(MODES[visualizer.alg])

                if event.key == pygame.K_ESCAPE: visualizer.en_menu = 1
        
        clock.tick(FPS)
                
    pygame.quit()


if __name__ == '__main__':
    pygame.init()
    pathfind_sim()