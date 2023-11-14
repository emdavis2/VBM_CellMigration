import numpy as np
import os
from itertools import chain
from shapely.geometry import Polygon, Point

from general_functions import *
from dirmig_poldir_VBM_func import *
from plot_functions import *
from shape_and_acf_functions import *

#Global variable for number of vertices to represent cell shape
global N
N = 25

magnitude = 50 #0.05 * (0.3*N*l0/0.4) * 2000 [nN]

#start with no forces on vertices
force_on_ind0 = np.array([]) #needs to be array

#initialize cell coordinates and get rest area
d = 50 #cell diameter [um]
l0 = d*np.sin(np.pi/N) #initial length of edge of cell between vertices
x, y = points_in_cell(d/2,N)
xy = np.concatenate((np.reshape(x,(N,1)),np.reshape(y,(N,1))),axis=1)
A_0 = Polygon(xy).area #resting area [um^2]# #initialize random state

#Total sim time
T_tot = 300 #[min?]
#step size
dt = .05

num_nearest_neighbors0 = calc_num_neighbors_protruding(force_on_ind0,N)

#set initial conditions
y0 = [x, y]
y0 = list(chain.from_iterable(y0))

#set parameter values
sigma_w = 2
sigma_off = 3
sigma_on = 3
a = 0.05
b = 2
params = [sigma_w, sigma_off, sigma_on, a, b]

save_path = '/Users/elizabethdavis/Desktop/Models/VBM/figures/dirmig_manycells'
if not os.path.exists(save_path):
  os.mkdir(save_path)

#create new file to write parameters to
other_params = ['N: {}\n'.format(N), 'magnitude: {}\n'.format(magnitude), 'd: {}\n'.format(d), 'nu: 1.67\n', 'lamb: 80\n', 'Kc: 80\n']
param_vals = open(save_path+'/params.txt','w')
file_lines = other_params + ['sigma_w: {}\n'.format(sigma_w), 'sigma_off: {}\n'.format(1), 'sigma_on: {}\n'.format(sigma_on), 'k_off: {}+({}*np.exp(-1*(((x_val-mu_opp)/sigma_off)**2)))\n'.format(a, b), 'k_on: {}+({}*np.exp(-1*(((x_val-mu)/sigma_on)**2)))\n'.format(a, b)]
#write lines to text file 
param_vals.writelines(file_lines)
param_vals.close() 

num_walkers = 10
#Where data is stored from all walkers in sim
data_sim = []
for walker in range(num_walkers):
  
    #initial polarity bias direction
    pol_dir0 = round(np.random.uniform(0,N))%N

    T, Y, Norm_Dir, force_ind, remove_p_events, add_p_events, pol_dir_all = EulerSolver(UpdateVertices, 0, T_tot, dt, y0, force_on_ind0, magnitude, pol_dir0, num_nearest_neighbors0, N, l0, A_0, params)

    #Make dataframe of shape and motion metrics for track
    onewalker_df = make_shape_motion_df(Y, dt, N)

    data_sim.append(onewalker_df)



#Plot cell centroid position over time and cell shape at beginning and end of simulation
plot_centroid_manycells(data_sim, N, save_path)

#Plot velocity acf
plot_vel_acf_manycells(data_sim,save_path)

#Plot boxplots for shape and motion metrics
make_shape_motion_boxplots(data_sim, save_path)