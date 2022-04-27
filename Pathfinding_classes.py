from __future__ import annotations

from pydantic import BaseModel
from Settings import *
from typing import Any, List, Optional, Tuple, Union
import heapq
import numpy as np
import pygame


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


class PriorityQueue:
    def __init__(self):
        self.nodes = []

    def put(self, node: Union[AStar, Dijkstra], cost: float):
        heapq.heappush(self.nodes, (cost, node))

    def get(self):
        return heapq.heappop(self.nodes)[1]

    def empty(self):
        return len(self.nodes) == 0
