import numpy as np
import os
from itertools import chain
from shapely.geometry import Polygon, Point

from general_functions import *
from dirmig_poldir_VBM_func import *
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
pol_dir0 = 5

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
y0 = [x, y]
y0 = list(chain.from_iterable(y0))

#set parameter values
sigma_w = 2
sigma_off = 3
sigma_on = 3
a = 0.05
b = 2
params = [sigma_w, sigma_off, sigma_on, a, b]

T, Y, Norm_Dir, force_ind, remove_p_events, add_p_events, pol_dir_all = EulerSolver(UpdateVertices, 0, T_tot, dt, y0, force_on_ind0, magnitude, pol_dir0, num_nearest_neighbors0, N, l0, A_0, params)

save_path = '/Users/elizabethdavis/Desktop/Models/VBM/figures/dirmig_1cell'

if not os.path.exists(save_path):
  os.mkdir(save_path)

#create new file to write parameters to
other_params = ['N: {}\n'.format(N), 'magnitude: {}\n'.format(magnitude), 'd: {}\n'.format(d), 'nu: 1.67\n', 'lamb: 80\n', 'Kc: 80\n']
param_vals = open(save_path+'/params.txt','w')
file_lines = other_params + ['sigma_w: {}\n'.format(sigma_w), 'sigma_off: {}\n'.format(1), 'sigma_on: {}\n'.format(sigma_on), 'k_off: {}+({}*np.exp(-1*(((x_val-mu_opp)/sigma_off)**2)))\n'.format(a, b), 'k_on: {}+({}*np.exp(-1*(((x_val-mu)/sigma_on)**2)))\n'.format(a, b)]
#write lines to text file 
param_vals.writelines(file_lines)
param_vals.close() 

#Plot cell centroid position over time and cell shape at beginning and end of simulation
plot_centroid(Y, N, save_path)

plot_cellshape(Y, 0, N, save_path)
plot_cellshape(Y, T_tot+1, N, save_path)

#Make movie of cell progressing over time
#make_movie(Y, T_tot, dt, N, -1000, 1000, save_path)

#Make dataframe of shape and motion metrics for track
onewalker_df = make_shape_motion_df(Y, dt, N)

#Plot velocity acf
plot_vel_acf_onecell(onewalker_df['vx'],onewalker_df['vy'],save_path)

plot_polaritybias_time(pol_dir_all,T,N,save_path)
plot_timebtw_force_onoff(add_p_events, remove_p_events, dt, save_path)
plot_cumnumevents(add_p_events, remove_p_events, T_tot, save_path)
plot_events_5min_win(add_p_events, remove_p_events, T_tot, dt, save_path)