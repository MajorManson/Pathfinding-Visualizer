from __future__ import annotations
from collections import deque
from typing import Any, List, Optional, Tuple, Union
import pygame
from pydantic import BaseModel
from Settings import *
import heapq
import numpy as np

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pathfinfing Visualizer')
clock = pygame.time.Clock()


class Pathfinding(BaseModel):

    row: int
    col: int

    x: int = 0
    y: int = 0

    color: Tuple[int, int, int] = MEDGRAY
    neighbours: List[AStar] = []
    width: int = TILESIZE

    total_rows = GRIDHEIGHT
    total_cols = GRIDWIDTH

    def __init__(__pydantic_self__, **data: Any) -> None:
        data['x'] = data['row'] * TILESIZE
        data['y'] = data['col'] * TILESIZE
        super().__init__(**data)

    def get_neighbours(self, grid: Union[list, np.ndarray]):

        self.neighbours = []
        if self.col > 0 and not grid[self.col - 1][self.row].is_wall(): #LEFT
            self.neighbours.append(grid[self.col - 1][self.row])

        if self.col < self.total_cols - 1 and not grid[self.col + 1][self.row].is_wall(): #RIGHT
            self.neighbours.append(grid[self.col + 1][self.row])

        if self.row > 0 and not grid[self.col][self.row - 1].is_wall(): #UP
            self.neighbours.append(grid[self.col][self.row - 1])

        if self.row < self.total_rows - 1 and not grid[self.col][self.row + 1].is_wall(): #DOWN
            self.neighbours.append(grid[self.col][self.row + 1])

        # Diagonals
        if issubclass(type(self), Dijkstra): return

        if self.col < self.total_cols - 1 and self.row > 0 and not grid[self.col + 1][self.row - 1].is_wall(): #TOP-RIGHT
            if not (grid[self.col][self.row - 1].is_wall() and grid[self.col + 1][self.row].is_wall()):
                self.neighbours.append(grid[self.col + 1][self.row - 1])
        
        if self.col > 0 and self.row > 0 and not grid[self.col - 1][self.row - 1].is_wall(): #TOP-LEFT
            if not (grid[self.col][self.row - 1].is_wall() and grid[self.col - 1][self.row].is_wall()):
                self.neighbours.append(grid[self.col - 1][self.row - 1])

        if self.col > 0 and self.row < self.total_rows - 1 and not grid[self.col - 1][self.row + 1].is_wall(): #BOTTOM-LEFT
            if not (grid[self.col][self.row + 1].is_wall() and grid[self.col - 1][self.row].is_wall()):
                self.neighbours.append(grid[self.col - 1][self.row + 1])

        if self.col < self.total_cols - 1 and self.row < self.total_rows - 1 and not grid[self.col + 1][self.row + 1].is_wall(): #BOTTOM-RIGHT
            if not (grid[self.col][self.row + 1].is_wall() and grid[self.col + 1][self.row].is_wall()):
                self.neighbours.append(grid[self.col + 1][self.row + 1])
    
    def get_pos(self):
        return self.row, self.col
    
    def is_open(self):
        return self.color == GREEN

    def is_wall(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == BLUE

    def reset(self):
        self.color = MEDGRAY

    def set_start(self):
        self.color = ORANGE

    def set_close(self):
        self.color = RED

    def set_open(self):
        self.color = GREEN

    def set_wall(self):
        self.color = BLACK

    def set_end(self):
        self.color = BLUE

    def set_path(self):
        self.color = YELLOW

    def draw(self, win: pygame):
        if not self.is_open():
            pygame.draw.rect(win, self.color,
                             (self.y, self.x,
                             self.width, self.width))
        else:
            pygame.draw.circle(win, self.color,
                               (self.y + self.width // 2,
                                self.x + self.width // 2),
                               self.width // 3)


class PriorityQueue:
    def __init__(self):
        self.nodes = []

    def put(self, node: Union[AStar, Dijkstra], cost: float):
        heapq.heappush(self.nodes, (cost, node))

    def get(self):
        return heapq.heappop(self.nodes)[1]

    def empty(self):
        return len(self.nodes) == 0


class AStar(Pathfinding):

    parent: Optional[AStar] = None

    F: float = 0.
    G: int = 0
    
    def __lt__(self, other: AStar):
        return self.F < other.F
    
    def __eq__(self, other: AStar):
        return other and self.row == other.row and \
            self.col == other.col


class Dijkstra(Pathfinding):

    parent: Optional[Dijkstra] = None

    def __eq__(self, other: Dijkstra):
        return other and self.row == other.row and \
            self.col == other.col

    
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
                start: Union[AStar, Dijkstra], draw: function) -> None:

    while currentNode.parent != start:
        currentNode = currentNode.parent
        currentNode.set_path()
        draw()


def DijkstraAlg(draw: function, start: Dijkstra,
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


def AStarAlg(draw: function, grid: Union[list, np.ndarray],
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
    pathfind_sim(WIN, Dijkstra)