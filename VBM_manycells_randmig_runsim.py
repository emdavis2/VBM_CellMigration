import numpy as np
import os
from itertools import chain
from shapely.geometry import Polygon, Point

from general_functions import *
from pol_dir_VBM_func import *
from plot_functions import *
from shape_and_acf_functions import *

#Global variable for number of vertices to represent cell shape
global N
N = 25

magnitude = 50*N #0.05 * (0.3*N*l0/0.4) * 2000 [nN]

#start with no forces on vertices
force_on_ind0 = np.array([]) #needs to be array

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

#set parameter values
kon_star = 0.05
t_max = 250
koff_star = 0.05
koff_max = .1
sigma_off = 30
params = [kon_star, t_max, koff_star, koff_max, sigma_off]

save_path = '/Users/elizabeth/Desktop/VBM_CellMigration_updated/figures/randmig_manycells_konstar{}_koffstar{}_tmax{}'.format(kon_star, koff_star, t_max)
if not os.path.exists(save_path):
  os.mkdir(save_path)

#create new file to write parameters to
other_params = ['N: {}\n'.format(N), 'magnitude: {}\n'.format(magnitude), 'd: {}\n'.format(d), 'nu: 1.67\n', 'nu_w: 0.5\n', 'lamb: 80\n', 'Kc: 80\n']
param_vals = open(save_path+'/params.txt','w')
file_lines = other_params + ['kon_star: {}\n'.format(kon_star), 't_max: {}\n'.format(t_max), 'koff_star: {}\n'.format(koff_star), 'koff_max: {}\n'.format(koff_max), 'sigma_off: {}\n'.format(sigma_off)]
#write lines to text file 
param_vals.writelines(file_lines)
param_vals.close() 

num_walkers = 10
#Where data is stored from all walkers in sim
data_sim = []
for walker in range(num_walkers):
    
    #set initial conditions for vertex coordinates
    #initial polarity bias direction
    w0 = np.random.uniform(0, 2*np.pi)

    y0 = [x, y, [w0]]
    y0 = list(chain.from_iterable(y0))

    T, Y, Norm_Dir, force_ind, remove_p_events, add_p_events, tension, num_nn, kon_rates, koff_rates, bias_ang_true = EulerSolver(UpdateVertices, 0, T_tot, dt, y0, force_on_ind0, magnitude, num_nearest_neighbors0, N, l0, A_0, params)

    #Make dataframe of shape and motion metrics for track
    onewalker_df = make_shape_motion_df(Y, dt, N)

    data_sim.append(onewalker_df)



#Plot cell centroid position over time and cell shape at beginning and end of simulation
plot_centroid_manycells(data_sim, N, save_path)

#Plot velocity acf
plot_vel_acf_manycells(data_sim,save_path)

#Plot boxplots for shape and motion metrics
make_shape_motion_boxplots(data_sim, save_path)