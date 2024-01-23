import numpy as np
import os
from itertools import chain
from shapely.geometry import Polygon, Point

from general_functions import *
from VBM_func_omega_numnn import *
from plot_functions import *
from shape_and_acf_functions import *
from event_verifi_plot_functions import *

#Global variable for number of vertices to represent cell shape
global N
N = 25

magnitude = 50 #0.05 * (0.3*N*l0/0.4) * 2000 [nN]

#start with no forces on vertices
force_on_ind0 = np.array([]) #needs to be array

#initial polarity bias direction
w0 = np.pi/4

#initialize cell coordinates and get rest area
d = 100 #cell diameter [um]
l0 = d*np.sin(np.pi/N) #initial length of edge of cell between vertices
x, y = points_in_cell(d/2,N)
xy = np.concatenate((np.reshape(x,(N,1)),np.reshape(y,(N,1))),axis=1)
A_0 = Polygon(xy).area #resting area [um^2]# #initialize random state

#Total sim time
T_tot = 300 #[min?]
#step size
dt = .05

num_nearest_neighbors0 = calc_num_neighbors_protruding(force_on_ind0,N)

#set initial conditions for vertex coordinates
y0 = [x, y, [w0]]
y0 = list(chain.from_iterable(y0))

#set parameter values
kon_star = 0.1
t_max = 250
koff_star = 0.1
sigma_off = 30
params = [kon_star, t_max, koff_star, sigma_off]

T, Y, Norm_Dir, force_ind, remove_p_events, add_p_events, tension, num_nn = EulerSolver(UpdateVertices, 0, T_tot, dt, y0, force_on_ind0, magnitude, num_nearest_neighbors0, N, l0, A_0, params)

save_path = '/Users/elizabeth/Desktop/VBM_CellMigration_updated/figures/randmig_1cell_omeganumnn_konstar{}_koffstar{}_tmax{}_changeFa'.format(kon_star, koff_star, t_max)

if not os.path.exists(save_path):
  os.mkdir(save_path)

#create new file to write parameters to
other_params = ['N: {}\n'.format(N), 'magnitude: {}\n'.format(magnitude), 'd: {}\n'.format(d), 'nu: 1.67\n', 'nu_w: 0.5\n', 'lamb: 80\n', 'Kc: 80\n']
param_vals = open(save_path+'/params.txt','w')
file_lines = other_params + ['kon_star: {}\n'.format(kon_star), 't_max: {}\n'.format(t_max), 'koff_star: {}\n'.format(koff_star), 'sigma_off: {}\n'.format(sigma_off)]
#write lines to text file 
param_vals.writelines(file_lines)
param_vals.close() 

#Plot cell centroid position over time and cell shape at beginning and end of simulation
plot_centroid(Y, N, save_path)

plot_cellshape(Y, 0, N, save_path)
plot_cellshape(Y, T_tot+1, N, save_path)

#Make movie of cell progressing over time
make_movie(Y, T_tot, dt, N, -1000, 1000, save_path)

#Make dataframe of shape and motion metrics for track
onewalker_df = make_shape_motion_df(Y, dt, N)

#Plot velocity acf
plot_vel_acf_onecell(onewalker_df['vx'],onewalker_df['vy'],save_path)

plot_timebtw_force_onoff(add_p_events, remove_p_events, dt, save_path)
plot_cumnumevents(add_p_events, remove_p_events, T_tot, save_path)
plot_events_5min_win(add_p_events, remove_p_events, T_tot, dt, save_path)

plot_spatial_tension(tension, save_path)

plot_num_nn(num_nn, save_path)