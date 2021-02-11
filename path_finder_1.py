# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:07:03 2020

@author: rowe1
"""

import pygame
import random
import time
import glob

from path_finder_tools import Djikstra, Double_Ended_BFS, A_Star
from path_finder_tools import Maze, A_Star_Greed, Game_Speed, Background, Score

from edge_detect_image import Edge_Detect


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
    sleep_time = 0.00
    game_speed = Game_Speed(sleep_time)
    
    #Preset Maze Files
    preset_mazes = glob.glob('./graphics/maze_images/*')
    
    pause = False
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
        elif keys[pygame.K_p]:
            pause = not pause
        elif keys[pygame.K_i]:
            rows = 80
            preset_image = Edge_Detect(random.choice(preset_mazes), rows = rows)
            pixels_per_square = bg_height/rows
            columns = int(rows*bg_width//bg_height)
            
            #Position image in center of array
            arr = Maze(rows, columns, maze_density)
            arr.clear()
            arr.image_overlay(preset_image.edges)
                       
            bg = Background(bg_width, bg_height, rows, columns, 0, gui_height)
            
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
            if not pause:
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