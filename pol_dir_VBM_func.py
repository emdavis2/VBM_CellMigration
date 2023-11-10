import numpy as np
from shapely.geometry import Polygon
from itertools import chain

from general_functions import *

#function that updates the direction of the internal polarity bias direction 
#what is meant by polarity bias direction is the indice of the vertex in which the 
#internal polarity bias direction vector is pointing to
#Input:
# w_prev => previous direction of internal polarity bias direction (type: int)
# N => number of vertices that represents cell (type: int)
# dt => time step size (type: float)
#Output:
# new_w => updated direction of internal polarity bias direction (type: int)
def update_pol_dir(w_prev, N, dt):
  k_w = 0.1
  sigma_w = 1
  rand_num = np.random.uniform()
  prob_change_w = k_w * dt
  if rand_num < prob_change_w :
    new_w = round(np.random.normal(w_prev,sigma_w))%N
  else:
    new_w = w_prev
  return new_w

###################################################################################################################################################

#function that calculates the rate the outward normal force turns off at vertices that currently have the force on
#this rate is affected by the internal polarity bias direction - vertices that are opposite the direction
#of the internal polarity bias direction have a higher off rate
#Input:
# force_on_ind => indices of vertices that currently have outward normal force "on" (type: ndarray of ints)
# pol_ang => direction of internal polarity bias direction (output from update_pol_dir()) (type: int)
# N => number of vertices that represents cell (type: int)
#Output:
# off_rates => list of off rates associated with each vertex that has a force on (type: list of floats)
def find_offrate(force_on_ind, pol_ang, N):
  #create an array to store off rates (one off rate per force indice)
  off_rates = np.zeros(len(force_on_ind))
  #find indices opposite the indices that currently have on forces
  #opp_force_on_ind = (force_on_ind+int(N/2))%N #actually we don't need?

  #specify mu and sigma for function associated with off rate
  mu_opp = (pol_ang+int(N/2))%N
  sigma_off = 3

  #create a long "unfolded" array of vertex indices (not modular)
  x_val = np.arange(-N,2*N)
  #calculate off rates with unfolded array
  k_off = (np.exp(-1*(((x_val-mu_opp)/sigma_off)**2)))

  #create modular version of unfolded array
  x_val_mod = x_val%N

  #loop through indices with force on
  for i, ind in enumerate(force_on_ind):
    #find indices in "modularified" unfolded array that are associated with indice of vertex with force on
    associated_ind = np.where(x_val_mod == ind)[0]
    #find off rate by summing k_off values associated with that indice of vertex with force on and add to array where final off rates are stored
    off_rates[i] = np.sum(k_off[associated_ind])

  return off_rates

###################################################################################################################################################

#function that calculates the rate an outward normal force turns on at vertices that currently have the force off
#this rate is affected by the internal polarity bias direction - vertieces that are in the direction 
#of the internal polarity bias direction have a higher on rate
#Input:
# force_off_ind => indices of vertices that currently have outward normal force "off" (type: ndarray of ints)
# pol_ang => direction of internal polarity bias direction (output from update_pol_dir()) (type: int)
# N => number of vertices that represents cell (type: int)
#Output:
# on_rates => list of on rates associated with each vertex that has a force off (type: list of floats)
def find_onrate(force_off_ind, pol_ang, N):
  #create an array to store on rates (one on rate per force indice)
  on_rates = np.zeros(len(force_off_ind))

  #specify mu and sigma for function associated with on rate
  mu = pol_ang
  sigma_on = 3

  #create a long "unfolded" array of vertex indices (not modular)
  x_val = np.arange(-N,2*N)
  #calculate on rates with unfolded array
  k_on = (np.exp(-1*(((x_val-mu)/sigma_on)**2)))

  #create modular version of unfolded array
  x_val_mod = x_val%N

  #loop through indices with force off
  for i, ind in enumerate(force_off_ind):
    #find indices in "modularified" unfolded array that are associated with indice of vertex with force off
    associated_ind = np.where(x_val_mod == ind)[0]
    #find on rate by summing k_off values associated with that indice of vertex with force off and add to array where final on rates are stored
    on_rates[i] = np.sum(k_on[associated_ind])

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
def add_and_remove_protrusions(all_vertices,force_on_ind,pol_ang,N,dt):
  #check and see event number is exponential
  remove_protr_event_num = 0
  add_protr_event_num = 0
  #look at vertices that have a force on - see if any force off events occur
  if len(force_on_ind) > 0:
    #where indices to vertices to be removed will be stored
    ind_to_remove = []
    #calculate rates force transitions to off based off pol ang
    #indicies opposite to direction of pol ang are more likely to transition to force off
    # mu = pol_ang
    # mu_opp = (pol_ang+int(N/2))%N
    # sigma = 2
    # opp_force_on_ind = (force_on_ind+int(N/2))%N
    # k_off = (np.exp(-1*(((opp_force_on_ind-mu_opp)/sigma)**2)))
    k_off = find_offrate(force_on_ind, pol_ang, N)
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
  # mu = pol_ang
  # sigma = 2
  # k_on = (np.exp(-1*(((force_off_ind-mu)/sigma)**2)))
  k_on = find_onrate(force_off_ind, pol_ang, N)
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
def UpdateVertices(y_val,force_arr,N,l0,A_0,nu=1.67,lamb=80,Kc=80):
  #Define parameters
  # nu = 1.67 #viscous friction factor [nN min um^(-1)]
  # lamb = 80 #stiffness of cortex [nN um^(-1)]
  # Kc = 80 #lamb #cytoplasmic stiffness (not sure if this is the right value...)

  #Define x and y positions for each vertex
  x = y_val[0:N]
  y = y_val[N:N*2]

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

  #Reactions
  rxn_x = []
  rxn_y = []
  for i in range(N):
    rxn_x.append((1/nu)*((E[i-1]*np.cos(dir_for_oppl[i])) + (E[i]*np.cos(dir_for_l[i])) + ((p/N) + force_arr[i])*np.cos(normal_dir[i])) + 5*np.random.normal(scale=1))
    rxn_y.append((1/nu)*((E[i-1]*np.sin(dir_for_oppl[i])) + (E[i]*np.sin(dir_for_l[i])) + ((p/N) + force_arr[i])*np.sin(normal_dir[i])) + 5*np.random.normal(scale=1))

  #ODEs
  dy = [rxn_x, rxn_y]
  dy = list(chain.from_iterable(dy))

  return np.array(dy), normal_dir

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
#Outputs:
# T => time points for which solution of ODE was solved at with len(((t_end-t_start)/h)+1) (type: list of floats)
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# Norm_Dir => normal directions (angles) for each vertex at all time points (type: list of lists of floats with shape=(t_end/dt,N))
# force_on_ind => indices of vertices with force on at end of simulatoin (type: array of ints with variable length)
# remove_protr_events => number of force off events at each time step (type: list of ints with len=t_end/dt)
# add_protr_events => number of force on events at each time step (type: list of ints with len=t_end/dt)
# pol_dir_all => the direction of the polarity bias at each time step (type: list of ints with len=t_end/dt)
def EulerSolver(func, t_start, t_end, dt, y0, force_on_ind0, magnitude, pol_dir0, num_nearest_neighbors0, N, l0, A_0):
  T = [t_start] #np.arange(1, t, h)
  Y = [y0]
  Norm_Dir = []

  #test to see if exponential distribution - list of number of events for each time step
  remove_protr_events = []
  add_protr_events = []

  number_steps = int(t_end/dt)

  all_vertices = set(np.arange(0,N))

  num_nearest_neighbors = num_nearest_neighbors0

  y_prev = y0
  t_prev = t_start
  force_on_ind = force_on_ind0
  pol_dir_prev = pol_dir0

  pol_dir_all = [pol_dir_prev]

  for i in range(number_steps):
    force_arr = np.zeros(N)
    if len(force_on_ind) > 0:
      force_arr[force_on_ind] = magnitude
      # force_arr[(force_on_ind+1)%N] = magnitude/2 #use if want direct neighbors to also have force on but with half the magnitude 
      # force_arr[(force_on_ind-1)%N] = magnitude/2
    m, normal_dir = func(y_prev, force_arr, N, l0, A_0)
    y_curr = y_prev + dt*m
    t_curr = t_prev + dt
    Y.append(y_curr)
    T.append(t_curr)
    Norm_Dir.append(normal_dir)

    y_prev = y_curr
    t_prev = t_curr

    #number of nearest neighbors for each site protruding
    num_nearest_neighbors = calc_num_neighbors_protruding(force_on_ind,N)

    pol_dir = update_pol_dir(pol_dir_prev, N, dt)

    force_on_ind, remove_protr_event_num, add_protr_event_num = add_and_remove_protrusions(all_vertices,force_on_ind,pol_dir,N,dt)

    remove_protr_events.append(remove_protr_event_num)
    add_protr_events.append(add_protr_event_num)

    pol_dir_prev = pol_dir

    pol_dir_all.append(pol_dir_prev)

  return T, Y, Norm_Dir, force_on_ind, remove_protr_events, add_protr_events, pol_dir_all