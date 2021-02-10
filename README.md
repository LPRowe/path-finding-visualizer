# path-finding-visualizer

Readme is under construction, cleaner details to come...<br><br>

### Intro

A path finding visualizer that implements BFS, Multisource BFS, and A* algorithms to find the shortest path between the two blue points.<br>
Press i to spawn pdf images where a sobel filter is applied and the images are converted to maze like arrays.<br>
Press f (fast) or s (slow) to adjust program's frame rate.<br>
Greediness of the A* algorithm is an adjustable parameter, in the path_finder_2.py version the optimal greed is automatically estimated by a machine learning model.  <br>
Grids can be randomly generated using the random grid button.<br>
The density of the grids can be adjusted by the grid density buttoms<br>
Use the reset button to clear the previous solution before running a new solution.<br>

<br><br>

### Colors

Grey pixels are obstacles.
Black pixels are unvisited open spaces.
Blue pixels mark the start point, target point, and optimal path.<br>
Orange pixels are visited nodes.<br>
Green pixels are nodes that are currenlty in the queue to be visited.<br><br>



<img src="./graphics/gif/mbfs_maze2.gif"><br><br>

<img src="./graphics/gif/mbfs_maze.gif"><br><br>

<img src="./graphics/gif/bfs_maze.gif"><br><br>

<img src="./graphics/gif/astar_maze.gif"><br><br>

<img src="./graphics/gif/astar_random.gif"><br><br>
