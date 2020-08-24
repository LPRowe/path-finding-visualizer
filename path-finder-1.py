# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:07:03 2020

@author: rowe1
"""

import pygame
import numpy as np
import random
from typing import List
import time
import heapq
import math

class Maze(object):
    def __init__(self, height, width, density):
        self.w = width
        self.h = height
        self.p = density
        self.arr = self.construct()
    
    def construct(self) -> List[List[int]]:
        '''
        Returns an array of randomly generated 0 and 1 values with a density of 1 values equal to p
        '''
        return [[1 if (random.random() < self.p) and (3 <= col < self.w-3) else 0 for col in range(self.w)] for _ in range(self.h)]
    
    def shuffle(self) -> None:
        '''
        Updates arr with a new set of values
        '''
        self.arr = self.construct()
    
    def draw(self, surface, x, y, dx, dy, color = (150, 150, 150)) -> None:
        '''
        Draws the array on a surface
        '''
        for r in range(self.h):
            for c in range(self.w):
                if self.arr[r][c] == 1:
                    pygame.draw.rect(surface, color, (x+c*dx, y+r*dy, dx, dy), 0)
        
    def show(self) -> None:
        for row in self.arr:
            print(row)
            
class Djikstra(object):
    
    def __init__(self, start, target, array, current_color, visited_color, target_color):
        self.R, self.C = len(array), len(array[0])
        self.arr = array
        self.color_c = current_color
        self.color_v = visited_color
        self.color_t = target_color
        
        self.start = start
        self.q = [start]
        self.t = target
        self.visited = set([start])
        self.steps = 1
        
        #Store graph of shortest path in a hash table
        self.g = {}
        
        self.solved = False
        self.best_path = []
        
    def reset(self):
        self.q = [self.start]
        self.visited = set([self.start])
        self.steps = 1
        self.g = {}
        self.solved = False
        self.best_path = []
        
    def step(self):
        '''
        Runs the next step in the djikstra algorithm
        '''
        next_level = []
        for r,c in self.q:
            for i,j in ((r+1,c),(r,c+1),(r-1,c),(r,c-1)):
                if (0 <= i < self.R) and (0 <= j < self.C) and (self.arr[i][j] == 0):
                    if (i,j) not in self.visited:
                        self.visited.add((i,j))
                        self.g[(i,j)] = (r,c)
                        next_level.append((i,j))
                        if (i,j) == self.t:
                            self.solved = True
                            self.steps += len(next_level)
                            self.q = []
                            self.trace_path()
                            return None
        
        self.steps += len(next_level)
        self.q = next_level
    
    def trace_path(self) -> None:
        loc = self.t
        while loc != self.start:
            self.best_path.append(loc)
            loc = self.g[self.best_path[-1]]
        return None
    
    def draw(self, surface, x, y, dx, dy) -> None:
        '''
        Color codes visited nodes (orange), current nodes (green), target and start (blue)
        '''
        for r,c in self.visited:
            pygame.draw.rect(surface, self.color_v, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for r,c in self.q:
            pygame.draw.rect(surface, self.color_c, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for r,c in [self.start, self.t]:
            pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
            
        #If Solved, connect path
        if self.solved:
            for r,c in self.best_path:
                pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
                    

class Double_Ended_BFS(object):
    
    def __init__(self, start, target, array, current_color, visited_color, target_color):
        self.R, self.C = len(array), len(array[0])
        self.arr = array
        self.color_c = current_color
        self.color_v = visited_color
        self.color_t = target_color
        
        self.start = start
        self.q = [[start], [target]]
        self.t = target
        self.visited = [{start : start}, {target: target}]
        self.steps = 1
        
        self.solved = False
        self.best_path = []
        
    def reset(self):
        self.q = [[self.start], [self.t]]
        self.visited = [{self.start : self.start}, {self.t: self.t}]        
        self.steps = 1
        self.solved = False
        self.best_path = []
        
    def step(self):
        '''
        Runs the next step in the djikstra algorithm
        '''
        
        if not self.q[0]: return None
        
        #Choose the shorter queue
        index = 0 if len(self.q[0]) <= len(self.q[1]) else 1
        
        q = self.q[index]
        
        
        next_level = []
        for r,c in q:
            for i,j in ((r+1,c),(r,c+1),(r-1,c),(r,c-1)):
                if (0 <= i < self.R) and (0 <= j < self.C) and (self.arr[i][j] == 0):
                    if (i,j) not in self.visited[index]:
                        self.visited[index][(i,j)] = (r,c)
                        next_level.append((i,j))
                        if (i,j) in self.visited[1-index]:
                            self.solved = True
                            self.steps += len(next_level)
                            self.q = [[], []]
                            self.trace_path((i,j)) #Update trace path to pull from visited 0 and visited 1
                            return None
        
        self.steps += len(next_level)
        self.q[index] = next_level
    
    def trace_path(self, midpoint) -> None:
        loc = midpoint
        while loc != self.start:
            self.best_path.append(loc)
            loc = self.visited[0][self.best_path[-1]]
        
        loc = midpoint
        while loc != self.t:
            self.best_path.append(loc)
            loc = self.visited[1][self.best_path[-1]]
            
        return None
    
    def draw(self, surface, x, y, dx, dy) -> None:
        '''
        Color codes visited nodes (orange), current nodes (green), target and start (blue)
        '''
        for r,c in list(self.visited[0].keys())+list(self.visited[1].keys()):
            pygame.draw.rect(surface, self.color_v, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for r,c in self.q[0]+self.q[1]:
            pygame.draw.rect(surface, self.color_c, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for r,c in [self.start, self.t]:
            pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
            
        #If Solved, connect path
        if self.solved:
            for r,c in self.best_path:
                pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
                
class A_Star(object):
    
    def __init__(self, start, target, array, current_color, visited_color, target_color, greed = 0.5):
        self.R, self.C = len(array), len(array[0])
        self.arr = array
        self.color_c = current_color
        self.color_v = visited_color
        self.color_t = target_color
        
        #Control The greed level : high greed = longer path : low greed = slow BFS
        self.greed = greed
        
        self.t = target
        self.start = start
        self.q = [(self.dist(start), 0, start)] # (Manhattan dist to target, location)
        self.visited = set([start])
        self.steps = 1
        self.r = {start : 0} #shortest distance to any given point
        
        #Store graph of shortest path in a hash table
        self.g = {}
        
        self.solved = False
        self.best_path = []
        
    def reset(self):
        self.q = [(self.dist(self.start), 0, self.start)]
        self.visited = set([self.start])
        self.steps = 1
        self.g = {}
        self.solved = False
        self.best_path = []
        self.r = {self.start : 0}
        
    def dist(self, loc):
        h1 = abs(loc[0] - self.start[0]) + abs(loc[1] - self.start[1]) #dijkstra
        h2 = abs(loc[0] - self.t[0]) + abs(loc[1] - self.t[1]) #greedy
        return int((1-self.greed)*h1 + self.greed*h2)
    
    def step(self):
        '''
        Runs the next step in the djikstra algorithm
        '''
        if self.solved: return None
        if not self.q: return None
        
        d, s, (r,c) = heapq.heappop(self.q)
        
        for i,j in ((r+1,c),(r,c+1),(r-1,c),(r,c-1)):
            if (0 <= i < self.R) and (0 <= j < self.C) and (self.arr[i][j] == 0):
                if ((i,j) not in self.visited) or (self.r[(i,j)] > s+1):
                    self.visited.add((i,j))
                    self.g[(i,j)] = (r,c)
                    self.r[(i,j)] = s + 1
                    heapq.heappush(self.q, (self.dist((i,j)), s + 1, (i,j)))
                    if (i,j) == self.t:
                        self.solved = True
                        self.steps += 1
                        self.q = []
                        self.trace_path()
                        return None
        
        #self.steps += 1 + int(math.log2(len(self.q)))
        self.steps += 1
    
    def trace_path(self) -> None:
        loc = self.t
        while loc != self.start:
            self.best_path.append(loc)
            r, c = loc
            loc = self.g[self.best_path[-1]]
            for i,j in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
                if (i,j) in self.g:
                    loc = min(loc, (i,j), key = lambda l: self.r[l])
                    print(self.r[loc], loc)
            
        return None
    
    def draw(self, surface, x, y, dx, dy) -> None:
        '''
        Color codes visited nodes (orange), current nodes (green), target and start (blue)
        '''
        for r,c in self.visited:
            pygame.draw.rect(surface, self.color_v, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for _, s, (r,c) in self.q:
            pygame.draw.rect(surface, self.color_c, (x+c*dx, y+r*dy, dx, dy), 0)
        
        for r,c in [self.start, self.t]:
            pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
            
        #If Solved, connect path
        if self.solved:
            for r,c in self.best_path:
                pygame.draw.rect(surface, self.color_t, (x+c*dx, y+r*dy, dx, dy), 0)
                
class A_Star_Greed(object):
    def __init__(self, greed_val):
        self.greed_val = greed_val
        
        #Font for greed level
        self.font=pygame.font.SysFont('tahoma', 40, bold=True)
        self.font_color = (235,235,235)
        
    def draw(self, surface):
        score_text = self.font.render(str(int(100*self.greed_val))+'%',1,self.font_color)
        surface.blit(score_text, (420, 18))

class Game_Speed(object):
    def __init__(self, sleep_time):
        self.sleep_time = sleep_time
        
        #Font for greed level
        self.font=pygame.font.SysFont('tahoma', 40, bold=True)
        self.font_color = (165,165,165)
        
    def draw(self, surface):
        score_text = self.font.render(str(int(100*(1-self.sleep_time/0.5)))+'%',1,self.font_color)
        surface.blit(score_text, (1472,21))

class Background(object):
    def __init__(self, width, height, rows, columns, x_naught, y_naught, line_width = 1):
        self.x = x_naught #top left corner
        self.y = y_naught
        self.w = width
        self.h = height
        self.r = rows
        self.c = columns
        self.lw = line_width
        self.color_bg = (0, 0, 0)
        self.color_grid = (200, 200, 200)
    
    def draw(self, surface) -> None:
        #Display Border
        pygame.draw.rect(surface, self.color_grid, (-0.5*self.lw, self.y-0.5*self.lw, self.w - 0.5*self.lw, self.h), 3*self.lw)
        
        #Display Grid (horizontal, vertical)
        #for i in np.linspace(self.y, self.y + self.h, self.r):
        #    pygame.draw.line(surface, self.color_grid, (0,i), (self.w, i), self.lw)
        #for i in np.linspace(0, self.w, self.c):
        #    pygame.draw.line(surface, self.color_grid, (i, self.y), (i, self.w + self.y), self.lw)
        
class Score(object):
    def __init__(self):
        self.font=pygame.font.SysFont('tahoma', 40, bold=True)
        self.font_color = (150, 150, 150)
        self.score = 0
        self.path_length = -1
    
    def draw(self, surface, x, y, solved = False):
        score_text = self.font.render('Steps: '+str(int(self.score)), 1, self.font_color)
        surface.blit(score_text, (x, y))
        
        if solved:
            path_text = self.font.render('Path Length: '+str(int(self.path_length)),1,self.font_color)
            surface.blit(path_text, (x, y+70))


pygame.init()
pygame.display.set_caption('Path Finder')
logo=pygame.image.load('./graphics/simple-logo.png')
pygame.display.set_icon(logo)

def main():
    
    #Surface settings
    window_width, window_height = 1000, 800
    window_width, window_height = 2000, 1200
    header_body_ratio = 0.8
    bg_width, bg_height = window_width, window_height*header_body_ratio
    gui_width, gui_height = window_width, window_height*(1-header_body_ratio)

    surface = pygame.display.set_mode((window_width, window_height))
    surface_color = (0,0,0)
    
    grid_color = (200, 200, 200)
    grid_width = 1
    
    #Add GUI Header
    bg_board = [pygame.image.load('./graphics/gui_layout_a_star.png'),
                pygame.image.load('./graphics/gui_layout_djikstra.png'),
                pygame.image.load('./graphics/gui_layout_multisource.png'),
                pygame.image.load('./graphics/gui_layout.png')]
    for b in bg_board:
        pygame.transform.scale(b, (int(bg_width),int(bg_height)))
    
    
    #Maze Settings
    rows = 40
    pixels_per_square = bg_height/rows
    columns = int(rows*bg_width//bg_height)
    bg = Background(bg_width, bg_height, rows, columns, 0, gui_height)
    maze_density = 0.35
    arr = Maze(rows, columns, maze_density)
    
    #Draw background and array outside of loop
    surface.fill(surface_color)
    arr.draw(surface, grid_width,
             gui_height+grid_width, bg_height / rows, bg_height / rows)
    bg.draw(surface)
    
    #Button Dictionary for header
    button_dict = {(293,16,557,74): 'astar',
                   (294,91,558,148): 'dijkstra',
                   (296,169,558,228): 'multi_source',
                   (633,172,900,231): 'random_grid',
                   (986,167,1253,226): 'clear_grid',
                   (1337,168,1607,227): 'solve',
                   (1340,90,1604,151): 'reset',
                   (1794,29,1847,81): 'density_up',
                   (1797,165,1846,223): 'density_down',
                   (120,33,169,74): 'greed_up',
                   (120,164,163,213): 'greed_down',
                   (1030,53,1075,104): 'grid_size_down',
                   (1222,57,1270,106): 'grid_size_up'}
    
    #Choose a solver
    current_color = (27, 232, 0) #green
    visited_color = (232, 187, 0) #orange
    target_color = (0, 232, 232) #blue-green
    start = (rows//2, 1)
    target = (rows//2, columns - 2)
    
    algorithm = 3
    solvers = [A_Star(start, target, arr.arr, current_color, visited_color, target_color),
               Djikstra(start, target, arr.arr, current_color, visited_color, target_color),
               Double_Ended_BFS(start, target, arr.arr, current_color, visited_color, target_color)]
    solver = solvers[1]
    solver.draw(surface, grid_width, gui_height+grid_width, 
                bg_height / rows, bg_height / rows)
    
    #Track the number of steps and shortest path
    score = Score()
    
    #Track the greed level of A*
    greed = A_Star_Greed(solvers[0].greed)
    
    #Limit game speed
    sleep_time = 0.03
    game_speed = Game_Speed(sleep_time)
    
    run = False

    while True:
        
        time.sleep(sleep_time) #slow down run speed for all algorithms besides A*
        
        event = pygame.event.poll()
        #print(event)
        if event.type == pygame.QUIT:
            break
        
        keys=pygame.key.get_pressed()
        if keys[pygame.K_f]:
            #speed up the game
            sleep_time -= 0.01
            sleep_time = max(0, sleep_time)
            game_speed.sleep_time = sleep_time
            game_speed.draw(surface)
        elif keys[pygame.K_s]:
            #slow down the game
            sleep_time += 0.01
            sleep_time = min(sleep_time, 0.5)
            game_speed.sleep_time = sleep_time
            game_speed.draw(surface)
        
        #Handle Mouse Clicks for buttons
        mouse = pygame.mouse
        if mouse.get_pressed()[0] and (mouse.get_pos()[1] <= gui_height):
            x,y = mouse.get_pos()
            print(x,y)
            for x1,y1,x2,y2 in button_dict:
               if (x1 <= x <= x2) and (y1 <= y <= y2):
                   button = button_dict[(x1,y1,x2,y2)]
                   print(button)
                   if button == 'astar':
                       algorithm = 0
                       solver = solvers[algorithm%3]
                       score.path_length = len(solver.best_path)
                   elif button == 'dijkstra':
                       algorithm = 1
                       solver = solvers[algorithm%3]
                       score.path_length = len(solver.best_path)
                   elif button == 'multi_source':
                       algorithm = 2
                       solver = solvers[algorithm%3]
                       score.path_length = len(solver.best_path)
                   elif button == 'solve':
                       solver.arr = arr.arr
                       run = True
                   elif button == 'reset':
                       run = False
                       solver.reset()
                       
                       #Redraw background and array outside of loop
                       surface.fill(surface_color)
                       arr.draw(surface, grid_width,
                                gui_height+grid_width, bg_height / rows, bg_height / rows)
                       bg.draw(surface)
                       
                       solver.draw(surface, grid_width, gui_height+grid_width, 
                                   bg_height / rows, bg_height / rows)
                   elif button == 'random_grid':
                       arr = Maze(rows, columns, maze_density)
                       
                       #Redraw background and array outside of loop
                       surface.fill(surface_color)
                       arr.draw(surface, grid_width,
                                gui_height+grid_width, bg_height / rows, bg_height / rows)
                       bg.draw(surface)
                       
                       #update solvers
                       for s in solvers:
                           s.arr = arr.arr
                       
                       solver.draw(surface, grid_width, gui_height+grid_width, 
                                   bg_height / rows, bg_height / rows)
                           
                   elif button == 'clear_grid':
                       print('a')
                       #Set all values in arr to zero
                       for r in range(arr.h):
                           for c in range(arr.w):
                               arr.arr[r][c] = 0
                               
                       #Redraw background and array outside of loop
                       surface.fill(surface_color)
                       arr.draw(surface, grid_width,
                                gui_height+grid_width, bg_height / rows, bg_height / rows)
                       bg.draw(surface)
                      
                       #Update all solvers
                       for s in solvers:
                           s.arr = arr.arr
                       solver.draw(surface, grid_width, gui_height+grid_width, 
                                   bg_height / rows, bg_height / rows)
                    
                   elif button in ['density_up', 'density_down']:
                       maze_density += 0.01 if button == 'density_up' else -0.01
                       
                       arr = Maze(rows, columns, maze_density)
                       
                       #Redraw background and array outside of loop
                       surface.fill(surface_color)
                       arr.draw(surface, grid_width,
                                gui_height+grid_width, bg_height / rows, bg_height / rows)
                       bg.draw(surface)
                       
                       #update solvers
                       for s in solvers:
                           s.arr = arr.arr
                       
                       solver.draw(surface, grid_width, gui_height+grid_width, 
                                   bg_height / rows, bg_height / rows)
                   elif button in ['greed_up', 'greed_down']:
                       greed_val = solvers[0].greed
                       greed_val += 0.02 if button == 'greed_up' else -0.02
                       greed_val = min(greed_val, 1)
                       greed_val = max(0, greed_val)
                       solvers[0].greed = greed_val
                       greed.greed_val = greed_val
                       greed.draw(surface)
                       time.sleep(0.1)
                   elif button in ['grid_size_up', 'grid_size_down']:
                       rows += 2 if button == 'grid_size_up' else -2
                       rows = max(2, rows)
                       rows = min(200, rows)
                       pixels_per_square = bg_height/rows
                       columns = int(rows*bg_width//bg_height)
                       bg = Background(bg_width, bg_height, rows, columns, 0, gui_height)
                       arr = Maze(rows, columns, maze_density)
                       
                       #Redraw background and array outside of loop
                       surface.fill(surface_color)
                       arr.draw(surface, grid_width,
                                gui_height+grid_width, bg_height / rows, bg_height / rows)
                       bg.draw(surface)
                       
                       #update solvers (also update target, source, R, C)
                       start, target = (rows//2, 1), (rows//2, columns - 2)
                       solvers = [A_Star(start, target, arr.arr, current_color, visited_color, target_color),
                                  Djikstra(start, target, arr.arr, current_color, visited_color, target_color),
                                  Double_Ended_BFS(start, target, arr.arr, current_color, visited_color, target_color)]
                       
                       solver = solvers[algorithm%3]
                       solver.draw(surface, grid_width, gui_height+grid_width, 
                                   bg_height / rows, bg_height / rows)
                        
                        
        elif mouse.get_pressed()[0] or mouse.get_pressed()[2]:
            #handle mouse clicks for drawing new walls and deleting old walls
            x,y = mouse.get_pos()
            print(x,y)
            r = int((y - gui_height) // pixels_per_square)
            c = int(x // pixels_per_square)
            print(r,c)
            arr.arr[r][c] = 1 if mouse.get_pressed()[0] else 0
            
            #Redraw background and array outside of loop
            surface.fill(surface_color)
            arr.draw(surface, grid_width,
                     gui_height+grid_width, bg_height / rows, bg_height / rows)
            bg.draw(surface)
           
            #update solvers
            for s in solvers:
                s.arr[r][c] = arr.arr[r][c]
        
        if run:
            solver.step()
            if not solver.solved: 
                print(solver.steps)
            else:
                score.path_length = len(solver.best_path)
                run = False
        
            #Blit board
            solver.draw(surface, grid_width, gui_height+grid_width, 
                        bg_height / rows, bg_height / rows)
        
        # Blit header
        surface.blit(bg_board[algorithm],(0, 0))
        
        # Blit Score
        score.score = solver.steps - 1
        score.draw(surface, 630, 20, solver.solved)
        
        # Blit Greed
        greed.draw(surface)
        
        # Blit Game Speed
        game_speed.draw(surface)
        
        pygame.display.flip()
        

    pygame.quit()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    