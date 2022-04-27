from collections import deque
from typing import Tuple, Callable
import pygame
import numpy as np
import argparse

from Settings import *
from Pathfinding_classes import *

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pathfinfing Visualizer')
clock = pygame.time.Clock()


def _parse_args():

    parser = argparse.ArgumentParser(description='Script for generating json files from forecast csv files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        'alg', choices=['A*', 'Dijkstra'],
        type=str, help='Choose which pathfinding algorithm to use'
    )

    return parser.parse_args()

    
def make_grid(cls: Union[AStar, Dijkstra]) -> np.ndarray:
    grid = []
    for i in range(GRIDWIDTH):
        grid.append([])
        for j in range(GRIDHEIGHT):
            grid[i].append(cls(row=j, col=i, parent=None))

    return np.array(grid)


def draw(win: pygame, grid: Union[list, np.ndarray]) -> None:

    win.fill(MEDGRAY)
    [spot.draw(win) for row in grid for spot in row]

    for x in range(0, WIDTH, TILESIZE):
        pygame.draw.line(win, WHITE, (x, 0), (x, HEIGHT))
        pygame.draw.line(win, WHITE, (0, x), (WIDTH, x))

    pygame.display.update()


def H(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    ret = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return ret


def vec2int(node: AStar) -> Tuple[int, int]:
    return (int(node.row), int(node.col))


def return_path(currentNode: Union[AStar, Dijkstra],
                start: Union[AStar, Dijkstra], draw: Callable) -> None:

    while currentNode.parent != start:
        currentNode = currentNode.parent
        currentNode.set_path()
        draw()


def DijkstraAlg(draw: Callable, start: Dijkstra,
                end: Dijkstra) -> None:

    openList = deque([start])
    visited = {vec2int(start)}

    while len(openList):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = openList.popleft()
        if current == end:
            end.set_end()
            return_path(current, start, draw)
            break

        for neighbour in current.neighbours:
            pos = vec2int(neighbour)
            if pos not in visited:
                neighbour.parent = current
                visited.add(pos)
                openList.append(neighbour)

                if not neighbour.is_end(): neighbour.set_open()

        draw()
        if current != start: current.set_close()

    return


def AStarAlg(draw: Callable, grid: Union[list, np.ndarray],
             start: AStar, end: AStar) -> None:
    
    openList = PriorityQueue()
    cost = {}

    for row in grid:
        for spot in row:
            spot.G = float('inf')
            spot.F = float('inf')
    
    start.F = H(start.get_pos(), end.get_pos())
    start.G = 0
    openList.put(start, start.G)
    cost[vec2int(start)] = 0

    while not openList.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = openList.get()

        if current == end:
            end.set_end()
            return_path(current, start, draw)
            break

        for neighbour in current.neighbours:
            G = current.G + 1

            if G < neighbour.G:
                neighbour.parent = current
                neighbour.G = G
                neighbour.F = G + H(neighbour.get_pos(), end.get_pos())

                pos = vec2int(neighbour)
                if pos not in cost:
                    openList.put(neighbour, neighbour.F)
                    cost[pos] = neighbour.F

                    if not neighbour.is_end(): neighbour.set_open()

        draw()

        if current != start: current.set_close()     

    return 


def pathfind_sim(win: pygame, alg: Union[AStar, Dijkstra]) -> None:

    grid = make_grid(alg)
    start, end = None, None

    running = True
    while running:
        draw(win, grid)
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

                    draw(win, grid)
                    if alg == Dijkstra:
                        DijkstraAlg(lambda: draw(win, grid), start, end)
                    else:
                        AStarAlg(lambda: draw(win, grid), grid, start, end)

                if event.key == pygame.K_r:
                    start = end = None
                    grid = make_grid(alg)

                if event.key == pygame.K_ESCAPE: running = False
        
        clock.tick(FPS)
                
    pygame.quit()


if __name__ == '__main__':
    alg = vars(_parse_args())['alg']
    pathfind_sim(WIN, AStar if alg == 'A*' else Dijkstra)