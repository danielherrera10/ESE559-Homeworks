#To run this code, you can input your desired parameters at lines 256-265 and just run it.
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import time
import statistics

#euclidean distance between two points
def distance(pos1,pos2):
    return math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)

def sign(n):
    return (n > 0) - (n < 0)

#function for determining which grid cells are covered by line segment connecting two points.
#Inspired by Fast Voxel Traversal Algorithm by Woo
def traversed(A,B):
    (xA, yA) = A
    (xB, yB) = B
    dx = xB - xA
    dy = yB - yA
    (sx, sy) = (sign(dx), sign(dy)) #indicating whether X and Y are incremented or decremented as the edge crosses cell boundaries

    cellA = (math.floor(xA),math.floor(yA))
    cellB = (math.floor(xB),math.floor(yB))
    (x, y) = cellA #starting grid cell
    traversed = [cellA] #first cell added to traversed list is the cell containing point A

    # Compute tDeltaX and tDeltaY in units of t
    #indicates how far along the ray we must move (in units of t) for the horizontal component of such a movement to equal the width of a voxel
    tDeltaX = abs(1 / dx) if dx != 0 else float('inf')
    tDeltaY = abs(1 / dy) if dy != 0 else float('inf')

    #tMax is value of t at which the ray crosses the first vertical voxel boundary
    if dx != 0:
        tMaxX = (x + sx - xA) / dx
    else:
        tMaxX = float('inf')  # Never crosses a vertical boundary

    if dy != 0:
        tMaxY = (y + sy - yA) / dy
    else:
        tMaxY = float('inf')  # Never crosses a horizontal boundary

    while (x,y) != cellB:

        if tMaxX < tMaxY:
            tMaxX = tMaxX + tDeltaX
            x += sx
        else:
            tMaxY= tMaxY + tDeltaY
            y += sy

        # BOUNDARY CHECK: Ensure x and y stay in the grid
        if x < 0.0 or x > 9.5 or y < 0.0 or y > 9.5:
            break  # Stop if outside grid bounds

        traversed.append( (x,y) )

    return traversed

def angle_diff(theta1, theta2):
    return min(abs(theta1 - theta2), 360 - abs(theta1 - theta2))

#rotates point by angle theta (degrees) around origin point
def rotate(origin, point, theta):
    theta_rad = math.radians(theta)
    ox, oy = origin
    px, py = point
    rx = ox + math.cos(theta_rad) * (px - ox) - math.sin(theta_rad) * (py - oy)
    ry = oy + math.sin(theta_rad) * (px - ox) + math.cos(theta_rad) * (py - oy)
    return (rx, ry)

#function for calculating rotated points from robots init config
def rotate_from_init_config(A, theta):
    Ax, Ay = A
    L_points = [(Ax,Ay+3),(Ax+0.3,Ay+3),(Ax+0.3,Ay+0.3),(Ax+3,Ay+0.3),(Ax+3,Ay)] #starting configuration for L
    if(theta==0 or theta==None):
        return L_points
    L_rotated_points = []
    for p in L_points:
        L_rotated_points.append(rotate(A,p,theta))
    return tuple(L_rotated_points)

#Rotates all corners of L given previous configuration
def rotate_L(A, theta, prev_corners):
    new_corners = []
    for p in prev_corners:
        new_corners.append(rotate(A,p,theta))
    return new_corners

#samples random A, theta and computes the corners
def rand_x(c_free):
    L_in_bounds = False
    while L_in_bounds == False:
        rand_A = random.choice(c_free)
        theta_random = random.uniform(0,360) # also sample random rotation angle in degrees
        corners = rotate_from_init_config(rand_A, theta_random)
        #Boundary check for L corners
        for r in corners:
            rx,ry = r
            if rx > 10 or rx < 0 or ry > 10 or ry < 0:
                L_in_bounds = False
            else:
                L_in_bounds = True
        
    return (rand_A,theta_random,tuple(corners))

#function calcalates which grid cells covered by L given by reference point A and theta of rotation measured counterclockwise from the +x-axis
#and given the coordinates of L corners. #Line segments are A-0, 0-1, 1-2, 2-3, 3-4, A-4 in L_corners
#IF rotation causes L to go out of grid, then it returns False
#Used by obstacleFree
def cellsCoveredBy_L(A,L_corners):
    #contains 5 points not including A and 6 line segments
    #Calculate L_points from previous state
    L_segments = [(A,L_corners[0]), (L_corners[0],L_corners[1]), (L_corners[1],L_corners[2]), (L_corners[2],L_corners[3]), (L_corners[3],L_corners[4]), (A,L_corners[4])]

    coveredCells = set()
    for seg in L_segments:
        covered = traversed(seg[0], seg[1]) #may contain duplicate cells
        for cell in covered:
            coveredCells.add(cell)

    return coveredCells

#nearest vertex
#Graph nodes are also triples (A,theta,tuple(corners))
def nearest_vertex(graph,state):
    A, theta, _ = state #current state includes xy pos of A and theta and corners
    return min(graph.nodes, key=lambda node: distance(A,node[0]) + abs(angle_diff(theta,node[1]))) #adds penalty for large theta differences

#steer function finds point z that minimizes ||z-x_rand|| while remaining within epsilon from x_nearest and theta remaining less than theta_step
def steer(x_nearest,x_rand,theta_step, epsilon,start_cell,c_free):
    x_nearest_A, x_nearest_theta, _ = x_nearest
    x_rand_A, theta_rand, _ = x_rand
    min_d = float('inf')
    z_A = None
    best_theta = None

    for p in c_free:
        if p != start_cell:
            #distance calculations
            d_epsilon = distance(p, x_nearest_A)
            d_p_to_randA = distance(p, x_rand_A)
            if d_p_to_randA < min_d and d_epsilon < epsilon:
                # Calculate the angular difference
                theta_diff = angle_diff(x_nearest_theta, theta_rand)
                
                # Move incrementally within the bounds of theta_step
                if abs(theta_diff) <= theta_step:
                    new_theta = theta_rand  # Directly move to the random theta if it's within the step range
                else:
                    # Move towards the target theta incrementally in the shortest direction
                    new_theta = (x_nearest_theta + theta_step * (1 if theta_diff > 0 else -1)) % 360

                test_corners = rotate_from_init_config(p,best_theta)

                #boundary check
                for corner in test_corners:
                    x,y = corner
                    if x>10 or x<0 or y>10 or y<0:
                        continue

                min_d = d_p_to_randA
                z_A = p
                best_theta = new_theta

    z_corners = rotate_from_init_config(z_A, best_theta)
    return (z_A,best_theta,tuple(z_corners))

#function that checks 1) whether or not the the line connecting A_nearest and A_rand is obstacle free
# 2) orientation of L in xnew is obstacle free.
def obstacleFree(x_nearest,x_new,c_free):
    A_nearest,_,_ = x_nearest
    A_new, _, corners_new = x_new
    cells_covered_by_line = traversed(A_nearest,A_new)
    for cell in cells_covered_by_line:
        if cell not in c_free:
            return False
    cells_covered_by_L = cellsCoveredBy_L(A_new, corners_new)
    for cell in cells_covered_by_L:
        if cell not in c_free:
            return False
    return True

##############  RRT Algorithm ####################
#robot state is a triple of the form (A_pt, theta, corner_pts)
def RRT(n, c_free, graph, epsilon, theta_step, init_robot_cell, goal_pos, goal_rad):
    goal_found = False
    for i in range(n):
        x_rand = rand_x(c_free) #x_rand = (rand_A, rand_theta, corner_pts)
        x_nearest = nearest_vertex(graph,x_rand)
        x_new = steer(x_nearest, x_rand, theta_step, epsilon, init_robot_cell, c_free)
        if obstacleFree(x_nearest,x_new,c_free):
            graph.add_node(x_new)
            graph.add_edge(x_nearest,x_new)
        if distance(x_new[0],goal_pos)<goal_rad: #check if point A in x_new state falls in goal radius
            goal_found = True
            print("x_new_A ="+str(x_new[0]))
            print("Goal Found!")
            return goal_found
    return goal_found

def grid(x_range, y_range, resolution):
    finer_grid = []
    for y in range(y_range):
        for x in range(x_range):
            # Divide each cell into smaller cells based on the resolution
            for dx in range(resolution):
                for dy in range(resolution):
                    # Generate the finer points (dx and dy are the sub-cell positions)
                    finer_x = x + dx / resolution
                    finer_y = y + dy / resolution
                    finer_grid.append((finer_x, finer_y))
    return finer_grid

def plotPathToGoal(obs1_x,obs1_y,obs2_x,obs2_y,graph):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.xlim(0,10)
    plt.ylim(0,10)
    #plot obstacles
    plt.fill(obs1_x,obs1_y,color='black')
    plt.fill(obs2_x,obs2_y,color='black')
    #plot goal
    goal = Circle((9,9),0.5,fill=False)
    ax.add_patch(goal)

    # plot L configurations
    for node in graph.nodes:
        A, _, corners = node
        L_ordered_pts = [A] + list(corners) + [A] #for plotting the L shape
        x, y = zip(*L_ordered_pts)
        plt.fill(x,y,color='green')
    plt.show()


###### RRT PARAMETERS #########
# Create a finer grid by dividing each cell into 2x2 sub-cells
resolution = 2  # Dividing each cell into 2x2 smaller cells (4 points per cell)
finer_grid = grid(x_range=10, y_range=10, resolution=2)

#OBSTACLE
obs_cells = [(4,0),(4,1),(4,9),(4,8),(4,7),(4,6)] #grid cells occupied by obstacle 1 and obstacle 2
obs1_corners = [(4.5,0),(5,0),(5,1.5),(4.5,1.5),(4.5,0)] #corners for plt.fill
obs1_x, obs1_y = zip(*obs1_corners)
obs2_corners = [(4.5,6),(5,6),(5,10),(4.5,10),(4.5,6)]
obs2_x, obs2_y = zip(*obs2_corners)

#Init C_FREE
c_free = [cell for cell in finer_grid if cell not in obs_cells]

n = 100
epsilon = 4.0
theta_step = 20
init_robot_cell = (0,0)
init_robot_pos = (0.5,0.5)
init_theta = 0
init_corners = tuple(rotate_from_init_config(init_robot_pos,init_theta))
init_robot_state = (init_robot_pos, init_theta, init_corners)
goal_pos = (9,9)
goal_rad = 0.5

######  RUNTIMES AND FIGURES
runtimes = []
for i in range(100):
    start = time.time()
    graph = nx.Graph()
    graph.add_node(init_robot_state)
    goal_found = RRT(n, c_free, graph, epsilon, theta_step, init_robot_cell, goal_pos, goal_rad)
    if goal_found == True:
        print("Goal found!")
        plotPathToGoal(obs1_x,obs1_y,obs2_x,obs2_y,graph)
    end = time.time()
    runtime = end - start  # Calculate the runtime
    runtimes.append(runtime)  # Store the runtime



avg_runtime = statistics.mean(runtimes)
var_runtime = statistics.variance(runtimes)

print("avg_runtime = " + str(avg_runtime))
print("variance_runtime = " + str(var_runtime))