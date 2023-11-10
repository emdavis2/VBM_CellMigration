import numpy as np
from shapely.geometry import Polygon, Point

#function to get angle of outward normal vector for one vertex position of interest
#Inputs: 
# a => [x, y] position from vertex of interest to neighboring vertex to the right (type: list of floats of len(2))
# b => [x, y] position from vertex of interest to neighboring vertex to the left (type: list of floats of len(2))
# x => x position of vertex of interest (type: float)
# y => y position of vertex of interest (type: float)
# poly => class from shapely.geometry that represents the current shape of the cell (type: class)
#Outputs:
# norm_angle => angle of the the outward normal vector in reference frame (type: float)
def get_norm_angle(a, b, x, y, poly):
  # get x and y components from each vector of each vector from function input
  ax = a[0]
  ay = a[1]
  bx = b[0]
  by = b[1]
  # normalize x and y components of vectors
  ax_norm = ax/np.sqrt(ax**2 + ay**2)
  ay_norm = ay/np.sqrt(ax**2 + ay**2)
  bx_norm = bx/np.sqrt(bx**2 + by**2)
  by_norm = by/np.sqrt(bx**2 + by**2)
  # add up x components and y components from the two vectors to find resultant components
  x_norm_comp = (ax_norm + bx_norm)
  y_norm_comp = (ay_norm + by_norm)
  # specify a point where the resultant vector is pointing to
  point = Point(x+x_norm_comp, y+y_norm_comp)
  # check if that point is inside or outside the shape (cell)
  if poly.contains(point):
    # if point is inside the shape, flip the resultant vector by 180 degrees
    x_norm_comp = -1 * x_norm_comp
    y_norm_comp = -1 * y_norm_comp
    norm_angle = np.arctan2(y_norm_comp,x_norm_comp)
  else:
    # if point is not inside shape, keep the direction
    norm_angle = np.arctan2(y_norm_comp,x_norm_comp)
  return norm_angle

###################################################################################################################################################

#function to get equally spaced points in a circle to initialize shape of cell
#Inputs:
# r => cell radius (type: float)
# N => number of vertices that represents cell (type: int)
#Outputs:
# x => x positions of vertices of cell (type: list of floats with len(N))
# y => y positions of vertices of cell (type: list of floats with len(N))
def points_in_cell(r,N): # r=cell radius and N=number of points
  x = []
  y = []
  for i in range(0,N):
    x.append(np.cos(2*np.pi/N*i)*r)
    y.append(np.sin(2*np.pi/N*i)*r)
  return x, y

###################################################################################################################################################

#function to get x and y coordinates to plot cell - turns it into a closed polygon so it visually looks nice
#Inputs:
# x => x cooridinates of vertices of cell (type: ndarray of floats of len(N))
# y => y coordinates of vertices of cell (type: ndarray of floats of len(N))
# N => number of vertices that represents cell (type: int)
#Outputs:
# x_p => x coordinates of vertices of cell (type: array.array of floats)
# y_p => y coordinates of vertices of cell (type: array.array of floats)
def getCoordToPlot(x, y, N):
  xy_reformat = np.concatenate((np.reshape(x,(N,1)),np.reshape(y,(N,1))),axis=1)
  poly = Polygon(xy_reformat)
  x_p,y_p = poly.exterior.xy
  return x_p, y_p

###################################################################################################################################################

#function to determine number of immediate nearest neighbors that are protruding for each site (look at neighboring two vertices on either side)
#this function outputs an array that gives number of neighbors with force on for each vertex regardless if that vertex has a force or not
#Input:
# force_ind => indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# N => number of vertices that represents cell (type: int)
#Output:
# num_nearest_neighbors => for each vertex, lists number of neighbors with force on (type: ndarray of ints with len(N))
def calc_num_neighbors_protruding_all_vertices(force_ind, N):
  num_nearest_neighbors = []
  for i in range(N):
    neighbors = set([(i+1)%N, (i-1)%N, (i+2)%N, (i-2)%N])
    protruding_neighbors = neighbors.intersection(set(force_ind))
    num_nn = len(protruding_neighbors)
    num_nearest_neighbors.append(num_nn)
  num_nearest_neighbors = np.array(num_nearest_neighbors)
  return num_nearest_neighbors

###################################################################################################################################################

#function to determine number of immediate nearest neighbors that are protruding for each site (look at neighboring two vertices on either side)
#this function outputs an array that gives number of neighbors with force on only for vertices that have a force on (those located in force_ind)
#Input:
# force_ind => indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# N => number of vertices that represents cell (type: int)
#Output:
# num_nearest_neighbors => for each vertex, lists number of neighbors with force on (type: ndarray of ints with len(force_ind))
def calc_num_neighbors_protruding(force_ind, N):
  num_nearest_neighbors = []
  if len(force_ind) > 0:
    for i in range(N):
      neighbors = set([(i+1)%N, (i-1)%N, (i+2)%N, (i-2)%N])
      protruding_neighbors = neighbors.intersection(set(force_ind))
      num_nn = len(protruding_neighbors)
      num_nearest_neighbors.append(num_nn)
    num_nearest_neighbors = np.take(np.array(num_nearest_neighbors),force_ind)
  return num_nearest_neighbors