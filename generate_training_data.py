# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:55:52 2020

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
        
    def show(self) -> None:
        for row in self.arr:
            print(row)

class A_Star(object):
    
    def __init__(self, start, target, array, greed = 0.5):
        self.R, self.C = len(array), len(array[0])
        self.arr = array
        
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
        return None

class Double_Ended_BFS(object):
    
    def __init__(self, start, target, array):
        self.R, self.C = len(array), len(array[0])
        self.arr = array
        
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




if __name__ == '__main__':
    '''
    1. Generate maze (30 rows by 62 columns)
    2. Perform double source BFS to find the absolute minimum
    3. Perform binary search with different A* greed values to determine the optimal greediness to find the shortest path
    4. Convert Array to 1D and save Array + optimal greediness value
    5. Repeat steps 1-4 for array densities between 9 and 35 percent (by 2)
    '''
    import numpy as np
    
    data = []
    
    rows = 30
    columns = 62
    start = (15, 1)
    target = (15, 60)
    density = np.linspace(0.09, 0.35, 14)
    
    for i in range(len(density)):
        print(i,'/',len(density))
        for _ in range(100):
            arr = Maze(rows, columns, density[i])
            
            BFS = Double_Ended_BFS(start, target, arr.arr)
            while not BFS.solved: 
                if not BFS.q[0] or not BFS.q[1]:
                    print('Unsolvable')
                    arr = Maze(rows, columns, density[i])
                    BFS = Double_Ended_BFS(start, target, arr.arr)
                BFS.step()
        
            shortest_path = len(BFS.best_path)
            
            low = 0.5
            high = 1
            while low < high - 0.01:
                greed = (low + high) / 2
                A = A_Star(start, target, arr.arr, greed)
                while not A.solved:
                    A.step()
                length = len(A.best_path)
                if length > shortest_path:
                    high = greed - 0.01
                else:
                    low = greed
            
            X = []
            for row in arr.arr:
                X.extend(row)
            X.append(low)
            data.append(X)
    
    a = np.array(data)
    np.savetxt('validation_data.csv',a,delimiter=',')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass