import numpy as np
from shapely.geometry import Polygon
from itertools import chain
import math

from general_functions import *

#function to find the intersection between two lines
#source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

###################################################################################################################################################

#function that calculates the rate the outward normal force turns off at vertices that currently have the force on
#this rate is affected by the internal polarity bias direction - vertices that are opposite the direction
#of the internal polarity bias direction have a higher off rate
#Input:
# force_on_ind => indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# pol_ang => direction of internal polarity bias direction (output from update_pol_dir()) (type: int)
#Optional Input:
# koff_star => base off rate - amount to shift rate function up (type: float)
# sigma_off => variance in off rate distribution (spread of how vertices' rates are affected by internal polarity direction) (type: float or int)
#Output:
# off_rates => list of off rates associated with each vertex that has a force on (type: list of floats)
def find_offrate(force_on_ind, pol_ang, x, y, N, koff_star=0.1, sigma_off=30):
  #create an array to store off rates (one off rate per force indice)
  off_rates = np.zeros(len(force_on_ind))

  #find direction opposite to polarity angle (pol_ang)
  pol_ang_opp = pol_ang + np.pi

  #get x and y compnents of the opposite direction polarity angle 
  x_w = np.cos(pol_ang_opp)
  y_w = np.sin(pol_ang_opp)

  #find the smallest angle between the opposite polarity angle and and the vertices
  min_angle = []
  for i in range(len(x)):
    min_angle.append(np.arccos(((x[i]*x_w) + (y[i]*y_w))/(np.sqrt(x[i]**2 + y[i]**2)*np.sqrt(x_w**2 + y_w**2))))

  closest_ind = np.argmin(min_angle)
  #find whether the polarity is to the right or left of that vertex it is closest to
  ang_dir = np.sign(x_w*y[closest_ind] - y_w*x[closest_ind])

  #name the x and y components of the vertex closest to opposite pol ang
  x1 = x[closest_ind]
  y1 = y[closest_ind]

  # find the other point that forms the edge where the polarity angle points towards
  if ang_dir > 0:
    x2 = x[closest_ind - 1]
    y2 = y[closest_ind - 1]
  else:
    x2 = x[(closest_ind + 1)%N]
    y2 = y[(closest_ind + 1)%N]

  #find where the opposite polarity angle lies along cell shape boundary 
  #for cell boundary
  A = [x1, y2]
  B = [x2, y2]

  #for polarity vec 
  C = [0, 0]
  D = [x_w, y_w]

  #this is where the opposite polarity angle vector lies along cell boundary
  intersec = (line_intersection((A, B), (C, D)))

  #calculate distance between vertices
  dist_between_ver = []
  for i in range(len(x)):
    dist_between_ver.append(np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2))

  #find cumulative sum of distance between vertices to get "unwrapped" spatial map of cell shape
  total_dist = np.cumsum(dist_between_ver)

  #location of opp pol ang along cell boundary broken up into x and y terms
  w_x = intersec[0]
  w_y = intersec[1]

  #get location of opp pol ang in this unwrapped spatial map view
  if ang_dir > 0:
    dist = np.sqrt((x2 - w_x)**2 + (y2 - w_y)**2)
    spatial_loc_w = total_dist[closest_ind - 1] + dist
  else:
    dist = np.sqrt((x1 - w_x)**2 + (y1 - w_y)**2)
    spatial_loc_w = total_dist[closest_ind] + dist

  #calculate off rates for each vertex that has a force on (depends on location relative to opp pol ang)
  for i, ind in enumerate(force_on_ind):
    vertex_loc = total_dist[ind]
    off_rates[i] = koff_star + 0.1+np.exp(-1*(((vertex_loc-spatial_loc_w)/sigma_off)**2))

  return off_rates

###################################################################################################################################################

#function that calculates the rate an outward normal force turns on at vertices that currently have the force off
#this rate is affected by the internal polarity bias direction - vertieces that are in the direction 
#of the internal polarity bias direction have a higher on rate
#Input:
# force_off_ind => indices of vertices that currently have outward normal force "off" (type: ndarray of ints)
# E_vertex => tension at one vertex between two neighboring verices (type: float)
#Optional Input:
# kon_star => base on rate (type: float or int)
# t_max => max tension at vertex
#Output:
# on_rates => list of on rates associated with each vertex that has a force off (type: list of floats)
def find_onrate(force_off_ind, E_vertex, kon_star=0.1, t_max=250):
  #create an array to store on rates (one on rate per force indice)
  on_rates = np.zeros(len(force_off_ind))

  #loop through indices with force off
  for i, ind in enumerate(force_off_ind):
    t_ind = E_vertex[ind]
    on_rates[i] = kon_star * np.exp(t_ind/t_max)

  return on_rates

###################################################################################################################################################

#function that turns forces on and off at each vertex based off the on and off rates
#Inputs: 
# all_vertices => set of all vertices that goes from 0 - (N-1) since that are all the labels/indices for the vertices that define the cell border (type: set of ints)
# force_on_ind => indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# pol_ang => direction of internal polarity bias direction (output from update_pol_dir()) (type: int)
# N => number of vertices that represents cell (type: int)
# dt => time step size (type: float)
#Outputs:
# force_on_ind => updated indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# remove_protr_event_num => talley of number of events (force turned off at vertex) that occured during this time step - 0 means nothing happened (no forces turned off) and >1 means that many vertices had force turned off (type: int)
# add_protr_event_num => talley of number of events (force turned on at vertex) that occured during this time step - 0 means nothing happened (no force added) and >1 means that many vertices had force turned on (type: int)
def add_and_remove_protrusions(all_vertices, force_on_ind, x, y, N, pol_ang, E_vertex, dt, kon_star=0.1, t_max=250, koff_star=0.1, sigma_off=30):
  #check and see event number is exponential
  remove_protr_event_num = 0
  add_protr_event_num = 0
  #look at vertices that have a force on - see if any force off events occur
  if len(force_on_ind) > 0:
    #where indices to vertices to be removed will be stored
    ind_to_remove = []
    #calculate rates force transitions to off based off pol ang
    #indicies opposite to direction of pol ang are more likely to transition to force off
    k_off = find_offrate(force_on_ind, pol_ang, x, y, N, koff_star, sigma_off)
    #probability that site with force on will transition to state with force off
    prob_remove_protr = k_off * dt
    #draw random number and see if it is less than the probability to stop exerting force
    rand_num = np.random.uniform(size=len(force_on_ind))
    for ind,v in enumerate(rand_num):
      #if less than probability, site stops exerting force at that location
      if v < prob_remove_protr[ind]:
        ind_to_remove.append(force_on_ind[ind])
        remove_protr_event_num += 1

    force_on_ind = np.array(list(set(force_on_ind) - set(ind_to_remove)))

  #chance to add new location for force, with a force being most likely to form at site of polarity direction
  force_off_ind = np.array(list(all_vertices.difference(set(force_on_ind))))
  ind_to_add = []
  #calculate rates force transitions to on based off pol ang
  #indices that are near the direction of the pol ang are more likely to transition to on
  k_on = find_onrate(force_off_ind, E_vertex, kon_star, t_max)
  prob_add_protr = k_on * dt
  rand_num2 = np.random.uniform(size=len(force_off_ind))
  for ind,v2 in enumerate(rand_num2):
    if v2 < prob_add_protr[ind]:
      ind_to_add.append(force_off_ind[ind])
      add_protr_event_num += 1

  force_on_ind = np.append(force_on_ind, ind_to_add).astype('int')

  return force_on_ind, remove_protr_event_num, add_protr_event_num

###################################################################################################################################################

#function that outputs the right hand side of the ODE for the equations of motion for each vertex - this is what is being solved with the Euler solver
#Inputs: 
# y_val => the x and y positions for each vertex together in one long list where 0-(N-1) values are x positions and N-((N*2)-1) are the y positions (type: list of floats)
# force_arr => indices of vertices that currently have outward normal force "on" (type: ndarray of int)
# N => number of vertices that represents cell (type: int)
# l0 => initial length of edge of cell between vertices (type: float)
# A_0 => resting area [um^2] of cell (type: float)
#Outputs:
# dy => the right hand side of the ODE for the equations of motion for each vertex where dx and dy are together in one long list of len(2*N) [0-(N-1) values are x positions and N-((N*2)-1) are the y positions] (type: ndarray of floats)
# normal_dir => outward normal angle for each vector that has len(N) (type: list of floats)
def UpdateVertices(y_val,force_arr,N,l0,A_0,nu=1.67,lamb=80,Kc=80,nu_w=0.5):
  #Define parameters
  # nu = 1.67 #viscous friction factor [nN min um^(-1)]
  # lamb = 80 #stiffness of cortex [nN um^(-1)]
  # Kc = 80 #lamb #cytoplasmic stiffness (not sure if this is the right value...)

  #Define x and y positions for each vertex
  x = y_val[0:N]
  y = y_val[N:N*2]
  w = y_val[-1]

  #Calculate 
  xy = np.concatenate((np.reshape(x,(N,1)),np.reshape(y,(N,1))),axis=1)
  A_c = Polygon(xy).area #cell area at current time point [um^2]
  poly_cell = Polygon(xy)
  p = Kc*(1 - (A_c/A_0))


  #lengths and directions (edge and normal)
  l = []
  dir_for_l = []
  dir_for_oppl = []
  normal_dir = []
  for i in range(N):
    l.append(np.sqrt((x[(i+1)%N]-x[i])**2 + (y[(i+1)%N]-y[i])**2))
    x_comp = x[(i+1)%N]-x[i]
    y_comp = y[(i+1)%N]-y[i]
    dir_for_l.append(np.arctan2(y_comp, x_comp))
    x_comp_opp = x[i-1]-x[i]
    y_comp_opp = y[i-1]-y[i]
    dir_for_oppl.append(np.arctan2(y_comp_opp, x_comp_opp))
    normal_dir.append(get_norm_angle([x_comp, y_comp], [x_comp_opp, y_comp_opp], x[i], y[i], poly_cell))

  #average lengths
  L = []
  for i in range(N):
    L.append((l[i]+l[i-1])/2)

  #Elastic tension
  E = lamb*((l-l0)/l0)

  E_vertex = []
  for i in range(len(E)):
    E_vertex.append(E[i] + E[i-1])

  #Reactions
  rxn_x = []
  rxn_y = []
  for i in range(N):
    rxn_x.append((1/nu)*((E[i-1]*np.cos(dir_for_oppl[i])) + (E[i]*np.cos(dir_for_l[i])) + ((p/N) + force_arr[i])*np.cos(normal_dir[i])) + 5*np.random.normal(scale=1))
    rxn_y.append((1/nu)*((E[i-1]*np.sin(dir_for_oppl[i])) + (E[i]*np.sin(dir_for_l[i])) + ((p/N) + force_arr[i])*np.sin(normal_dir[i])) + 5*np.random.normal(scale=1))
    
  # num_nearest_neighbors = calc_num_neighbors_protruding_all_vertices(force_on_ind,N)
  num_nearest_neighbors = calc_num_neighbors_protruding_all_vertices(np.nonzero(force_arr)[0],N)
  bias_ang_ind = np.argmax(num_nearest_neighbors)
  bias_ang = math.atan2(y[bias_ang_ind],x[bias_ang_ind])
  x_w = np.cos(w)
  y_w = np.sin(w)
  x_ba = np.cos(bias_ang)
  y_ba = np.sin(bias_ang)
  ang_diff = np.arccos(((x_w*x_ba) + (y_w*y_ba))/(np.sqrt(x_ba**2 + y_ba**2) * np.sqrt(x_w**2 + y_w**2)))
  ang_dir = np.sign(x_w*y_ba - y_w*x_ba)
  E_base = 100 #nN - basal insignificant tension
  F_a = 1 #np.max(E_vertex)/E_base
  rxn_w = ang_dir*(1/nu_w)*(F_a*ang_diff + np.random.normal(scale=1))

  #ODEs
  dy = [rxn_x, rxn_y, [rxn_w]]
  dy = list(chain.from_iterable(dy))

  return np.array(dy), normal_dir, E_vertex

###################################################################################################################################################

#Euler solver function that solves UpdateVertices() function and updates internal polarity bias direction and forces turned on and off at each vertex
#Inputs:
# func => function to solve when ODE is in the form dy/dt = f(t,y) where f(t,y) is func (type: function)
# t_start => time to begin simulation (type: int)
# t_end => end time point to run simulation (type: int)
# dt => step size in solver (type: float)
# y0 => initial condition (in this case starting x and y positions of vertieces) (type: list of floats)
# force_on_ind0 => initial condition for what vertices are starting with outward normal force on (type: ndarray of ints)
# magnitude => magnitude of normal outward force that turns on and off (type: float or int)
# pol_dir0 => initial polarity bias direction (type: int)
# num_nearest_neighbors0 => initial number of neighbors with force on for each vertex (type: ndarray of ints with length=N)
# N => number of vertices that represents cell (type: int)
# l0 => initial length of edge of cell between vertices (type: float)
# A_0 => resting area [um^2] of cell (type: float)
# params => list of parameters associated with rates for force on/off and polarity bias direction in order [k_w, sigma_w, sigma_off, sigma_on] (type: list of floats)
#Outputs:
# T => time points for which solution of ODE was solved at with len(((t_end-t_start)/h)+1) (type: list of floats)
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# Norm_Dir => normal directions (angles) for each vertex at all time points (type: list of lists of floats with shape=(t_end/dt,N))
# force_on_ind => indices of vertices with force on at end of simulatoin (type: array of ints with variable length)
# remove_protr_events => number of force off events at each time step (type: list of ints with len=t_end/dt)
# add_protr_events => number of force on events at each time step (type: list of ints with len=t_end/dt)
# pol_dir_all => the direction of the polarity bias at each time step (type: list of ints with len=t_end/dt)
def EulerSolver(func, t_start, t_end, dt, y0, force_on_ind0, magnitude, num_nearest_neighbors0, N, l0, A_0, params):
  T = [t_start] #np.arange(1, t, h)
  Y = [y0]
  Norm_Dir = []
  tension = []
  num_nn = []

  #test to see if exponential distribution - list of number of events for each time step
  remove_protr_events = []
  add_protr_events = []

  number_steps = int(t_end/dt)

  all_vertices = set(np.arange(0,N))

  y_prev = y0
  t_prev = t_start
  force_on_ind = force_on_ind0

  #unpack params (params = [k_w, sigma_w, sigma_off, sigma_on])
  kon_star = params[0]
  t_max = params[1]
  koff_star = params[2]
  sigma_off = params[3]

  #Now solve ODE
  for i in range(number_steps):
    force_arr = np.zeros(N)
    if len(force_on_ind) > 0:
      force_arr[force_on_ind] = magnitude
      # force_arr[(force_on_ind+1)%N] = magnitude/2 #use if want direct neighbors to also have force on but with half the magnitude 
      # force_arr[(force_on_ind-1)%N] = magnitude/2
    m, normal_dir, E_vertex = func(y_prev, force_arr, N, l0, A_0)
    y_curr = y_prev + dt*m
    t_curr = t_prev + dt
    Y.append(y_curr)
    T.append(t_curr)
    Norm_Dir.append(normal_dir)
    tension.append(E_vertex)

    y_prev = y_curr
    t_prev = t_curr

    #number of nearest neighbors for each site protruding
    num_nearest_neighbors = calc_num_neighbors_protruding_all_vertices(force_on_ind,N)
    num_nn.append(num_nearest_neighbors)

    pol_ang = y_curr[-1] #also known as w

    x = y_curr[0:N]
    y = y_curr[N:N*2]

    force_on_ind, remove_protr_event_num, add_protr_event_num = add_and_remove_protrusions(all_vertices, force_on_ind, x, y, N, pol_ang, E_vertex, dt, kon_star, t_max, koff_star, sigma_off)

    remove_protr_events.append(remove_protr_event_num)
    add_protr_events.append(add_protr_event_num)


  return T, Y, Norm_Dir, force_on_ind, remove_protr_events, add_protr_events, tension, num_nn