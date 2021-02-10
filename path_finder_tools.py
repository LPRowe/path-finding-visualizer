# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:18:00 2020

@author: rowe1
"""

import pygame
import random
from typing import List
import time
import heapq
import math

def tile_resize(arr, target_size = (30, 62)):
    '''
    :arr: Array of any size
    returns array of dimensions (30 by 62)
    
    Motivation: Convolutional Nerual Net was trained on mazes with dimensions 30 by 62.  Rescaling
    larger/smaller mazes to this size seems to adversely affect the CNN's predictive capabilities.  
    
    Resizing the array by tiling the array (if array is too small) or by taking a sub-sample of
    the array (if the array is too large) will likely improve the CNN's ability to classify mazes.
    '''
    TR, TC = target_size
    R, C = len(arr), len(arr[0])
    if R > 30:
        res = [row[(C//2)-(TC//2):(C//2)+(TC//2)] for row in arr[(R//2)-(TR//2):(R//2)+(TR//2)]]
    else:
        res = [[0]*TC for _ in range(TR)]
        for i in range(TR):
            for j in range(TC):
                res[i][j] = arr[i%R][j%C]
    return res
    

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
    
    def clear(self) -> None:
        '''
        Sets all maze value to zero
        '''
        for r in range(self.h):
            for c in range(self.w):
                self.arr[r][c] = 0
    
    def image_overlay(self, image) -> None:
        '''
        overlays an array onto the current maze
        '''
        image_columns = len(image[0])
        start_x = (self.w-image_columns)//2
        for r in range(self.h):
            for c in range(len(image[0])):
                self.arr[r][c+start_x] = image[r][c]
    
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
        self.r = {start : 0} #shortest distance to any given point
        self.q = [(0, 0, start)] # (Manhattan dist to target, steps taken, location)
        self.visited = set([start])
        self.steps = 1
        
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
        h1 = self.r[loc] #dijkstra
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