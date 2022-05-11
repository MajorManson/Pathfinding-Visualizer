from __future__ import annotations

from pydantic import BaseModel
from Settings import *
from typing import Any, List, Optional, Tuple, Union
import heapq
import numpy as np
from collections import deque
from typing import Tuple, Callable
import pygame

class Visualizer(BaseModel):

    en_menu: Optional[int] = 0

    alg: Optional[str] = None

    def __init__(__pydantic_self__, **data: Any) -> None:
        pygame.display.set_caption('Pathfinfing Visualizer')
        super().__init__(**data)

    @staticmethod
    def make_grid(cls: Union[AStar, Dijkstra]) -> np.ndarray:
        grid = []
        for i in range(GRIDWIDTH):
            grid.append([])
            for j in range(GRIDHEIGHT):
                grid[i].append(cls(row=j, col=i, parent=None))

        return np.array(grid)

    @staticmethod
    def draw(win: pygame.Surface, grid: Union[list, np.ndarray]) -> None:

        win.fill(MEDGRAY)
        [spot.draw(win) for row in grid for spot in row]

        for x in range(0, WIDTH, TILESIZE):
            pygame.draw.line(win, WHITE, (x, 0), (x, HEIGHT))
            pygame.draw.line(win, WHITE, (0, x), (WIDTH, x))

        pygame.display.update()

    @staticmethod
    def H(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        ret = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return ret

    @staticmethod
    def vec2int(node: AStar) -> Tuple[int, int]:
        return (int(node.row), int(node.col))

    def return_path(self, currentNode: Union[AStar, Dijkstra],
                    start: Union[AStar, Dijkstra],
                    draw: Callable) -> None:

        while currentNode.parent != start:
            currentNode = currentNode.parent
            currentNode.set_path()
            draw()
    
    def menu(self, win: pygame.Surface) -> str:

        # white color 
        color = (255, 255, 255) 
        
        # light shade of the button 
        color_light = (170, 170, 170) 
        
        # dark shade of the button 
        color_dark = (100, 100, 100) 
        
        # stores the width of the 
        # screen into a variable 
        width = win.get_width() 
        
        # stores the height of the 
        # screen into a variable 
        height = win.get_height() 
        
        # defining a font 
        smallfont = pygame.font.SysFont('Corbel', 35) 
        
        # rendering a text written in 
        # this font 
        texts = [(smallfont.render('Quit' , True , color),
                (width / 2 - 30, height / 2 + 83)),
                (smallfont.render('A*' , True , color),
                (width / 2 - 15, height / 2 + 5)),
                (smallfont.render('Dikjstra' , True , color),
                (width / 2 - 50, height / 2 - 78))]

        buttons = [((width / 2 - 150, width / 2 + 150),
                    (height / 2 + 80, height / 2 + 120), 'Quit'),
                ((width / 2 - 150, width / 2 + 150),
                    (height / 2, height / 2 + 40), 'A*'),
                ((width / 2 - 150, width / 2 + 150),
                    (height / 2 - 80, height / 2 - 40), 'Dijkstra')]
        
        while True: 
            
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get(): 
                
                if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                    self.alg = 'Quit'
                    return
                    
                #checks if a mouse is clicked 
                if event.type == pygame.MOUSEBUTTONDOWN: 
                    
                    #if the mouse is clicked on the 
                    # button the game is terminated

                    for button in buttons:
                        if button[0][0] <= mouse[0] <= button[0][1] and \
                            button[1][0] <= mouse[1] <= button[1][1]:
                            self.alg = button[2]
                            return
                        
            # fills the screen with a color 
            win.fill((60,25,60)) 
            
            # stores the (x,y) coordinates into 
            # the variable as a tuple 
            
            for button, text in zip(buttons, texts):
                if button[0][0] <= mouse[0] <= button[0][1] and \
                    button[1][0] <= mouse[1] <= button[1][1]:
                    c = color_dark
                else: c = color_light
                
                pygame.draw.rect(win, c,
                                 [button[0][0], button[1][0], 300, 40],
                                 0, 30)
            
                # superimposing the text onto our button 
                win.blit(*text) 
            
            # updates the frames of the game 
            pygame.display.update()

    def DijkstraAlg(self, draw: Callable, start: Dijkstra,
                    end: Dijkstra) -> None:

        openList = deque([start])
        visited = {self.vec2int(start)}

        while len(openList):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                
                if event.type == pygame.KEYDOWN and \
                    event.key == pygame.K_ESCAPE:
                    self.en_menu = 1
                    return

            current = openList.popleft()
            if current == end:
                end.set_end()
                self.return_path(current, start, draw)
                break

            for neighbour in current.neighbours:
                pos = self.vec2int(neighbour)
                if pos not in visited:
                    neighbour.parent = current
                    visited.add(pos)
                    openList.append(neighbour)

                    if not neighbour.is_end(): neighbour.set_open()

            draw()
            if current != start: current.set_close()

        return

    def AStarAlg(self, draw: Callable, grid: Union[list, np.ndarray],
                 start: AStar, end: AStar) -> None:
        
        openList = PriorityQueue()
        cost = {}

        for row in grid:
            for spot in row:
                spot.G = float('inf')
                spot.F = float('inf')
        
        start.F = self.H(start.get_pos(), end.get_pos())
        start.G = 0
        openList.put(start, start.G)
        cost[self.vec2int(start)] = 0

        while not openList.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                
                if event.type == pygame.KEYDOWN and \
                    event.key == pygame.K_ESCAPE:
                    self.en_menu = 1
                    return

            current = openList.get()

            if current == end:
                end.set_end()
                self.return_path(current, start, draw)
                break

            for neighbour in current.neighbours:
                G = current.G + 1

                if G < neighbour.G:
                    neighbour.parent = current
                    neighbour.G = G
                    neighbour.F = G + self.H(neighbour.get_pos(), end.get_pos())

                    pos = self.vec2int(neighbour)
                    if pos not in cost:
                        openList.put(neighbour, neighbour.F)
                        cost[pos] = neighbour.F

                        if not neighbour.is_end(): neighbour.set_open()

            draw()

            if current != start: current.set_close()     

        return 


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
