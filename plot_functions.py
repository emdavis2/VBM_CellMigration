import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sns

from general_functions import *
from shape_and_acf_functions import *
from pol_dir_VBM_func import *

#function that plots centroid position of the cell over time
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# N => number of vertices that represents cell (type: int)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of centroid position saved as png to directory specified in save_path variable
def plot_centroid(Y,N,save_path):
    centroid_x = []
    centroid_y = []
    for time_point in Y:
        centroid_x.append(np.sum(time_point[0:N])/N)
        centroid_y.append(np.sum(time_point[N:N*2])/N)

    plt.plot(centroid_x,centroid_y)
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('centroid position of cell over time')
    plt.savefig(save_path+'/centroid_pos.png')
    plt.clf()

###################################################################################################################################################

#function that plots centroid position of many cells over time
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell centroid as well shape and motion metrics for each track (type: list of pandas dataframes)
# N => number of vertices that represents cell (type: int)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of centroid position saved as png to directory specified in save_path variable
def plot_centroid_manycells(data_sim,N,save_path):
    for df in data_sim:
        plt.plot(df['x'],df['y'])
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('centroid position of cell over time')
    plt.savefig(save_path+'/centroid_pos_manycells.png')
    plt.clf()

###################################################################################################################################################

#function that plots smoothed centroid position of the cell over time - right now it smooths so data is sampled equivalent to every 5 minutes to match data 
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# N => number of vertices that represents cell (type: int)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of smoothed centroid position saved as png to directory specified in save_path variable
def plot_centroid_smooth(Y,N,dt,save_path):
    centroid_x = []
    centroid_y = []
    for time_point in Y:
        centroid_x.append(np.sum(time_point[0:N])/N)
        centroid_y.append(np.sum(time_point[N:N*2])/N)

    x_centroid_smooth = centroid_x[::int(5/dt)]
    y_centroid_smooth = centroid_y[::int(5/dt)]

    plt.plot(x_centroid_smooth,y_centroid_smooth)
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('centroid position of cell over time smoothed')
    plt.savefig(save_path+'/centroid_pos_smooth.png')
    plt.clf()

###################################################################################################################################################

#function that plots cell shape at specified time point
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# ti => indice of time point to be plotted (corresponds to T[ti] in output from EulerSolver() function)
# N => number of vertices that represents cell (type: int)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of cell shape at specified time point saved as png to directory specified in save_path variable
def plot_cellshape(Y,ti,N,save_path):
    x_ti = Y[ti][0:N]
    y_ti = Y[ti][N:N*2]

    x_ti, y_ti = getCoordToPlot(x_ti, y_ti, N)

    plt.plot(x_ti, y_ti)
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.savefig(save_path+'/cellshape_t{}.png'.format(ti))
    plt.clf()

###################################################################################################################################################

#function that makes a mp4 video of the cell evolving over time - note: movie shape is always square...doesn't have to be but is for now
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# t_end => end time point to run simulation (type: int)
# dt => step size in solver (type: float)
# N => number of vertices that represents cell (type: int)
# xy_min => lower bound for x and y axis limits (type: float)
# xy_max => upper bound for x and y axis limits (type: float)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# movie of cell evolving over time saved as mp4
def make_movie(Y,t_end,dt,N,xy_min,xy_max,save_path):
    fig = plt.figure()
    ax = plt.axes(xlim=(xy_min, xy_max), ylim=(xy_min, xy_max))
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x_ti = Y[i][0:N]
        y_ti = Y[i][N:N*2]

        x_ti, y_ti = getCoordToPlot(x_ti, y_ti, N)
        line.set_data(x_ti, y_ti)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=int((t_end/dt)), interval=20, blit=True)

    FFwriter = FFMpegWriter(fps=20)
    anim.save(save_path + '/VBM_sim_movie.mp4', writer=FFwriter)

###################################################################################################################################################

#function to plot velocity acf for one cell
#Inputs:
# x_vel => x component of velocity (type: pandas series of floats)
# y_vel => y componenet of velocity (type: pandas series of floats)
# save_path => path to folder where plot will be saved to (type: string)
# Optional:
# min_track_length => specified track length to cut off autocorrelation results (what value of tau to cut off at)
#Outputs:
# figure of velocity autocorrelation for lags up to min_track_length-4
def plot_vel_acf_onecell(x_vel, y_vel, save_path, min_track_length=30):
    combined = make_comb_df(x_vel, y_vel)
    poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr_vector(combined, min_track_length)

    plt.plot(poslagsmean,label = "positive lag")
    plt.hlines(y=0,xmin=0,xmax=100,color='k')
    plt.xlim(0,min_track_length-4)
    plt.ylim(-0.5,1)
    plt.xlabel('lag (5 min)')
    plt.title("Autocorrelaton velocity")
    plt.savefig(save_path+'/velocity_acf.png')
    plt.clf()

###################################################################################################################################################

#function to plot average velocity acf across lags from many cells
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell centroid as well shape and motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
# Optional:
# min_track_length => specified track length to cut off autocorrelation results (what value of tau to cut off at)
#Outputs:
# figure of average velocity autocorrelation across lags from many cells up to min_track_length-4
def plot_vel_acf_manycells(data_sim, save_path, min_track_length=30):
    poslagaverage = np.zeros(30000)
    Nposlagtotal = np.zeros(30000)
    all_ac = []
    for df in data_sim:
        combined = make_comb_df(df['vx'], df['vy'])
        poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr_vector(combined, min_track_length)

        #remove nans here
        poslagsmean[np.isnan(poslagsmean)] = 0
        all_ac.append(poslagsmean)
        poslagaverage[0:len(poslagsmean)] += poslagsmean # Nposlags*poslagsmean

    poslagaverage /= len(data_sim) #Nposlagtotal

    std_err = np.std(all_ac,axis=0,ddof=1)/np.sqrt(np.shape(all_ac)[0])

    plt.errorbar(np.arange(0,min_track_length-4),poslagaverage[0:min_track_length-4],yerr=std_err)
    plt.hlines(y=0,xmin=0,xmax=100,color='k')
    plt.xlim(0,min_track_length-4)
    plt.ylim(-0.5,1)
    plt.xlabel('lag (5 min)')
    plt.title("Autocorrelaton velocity")
    plt.savefig(save_path+'/velocity_acf_avgcells.png')
    plt.clf()

###################################################################################################################################################

#function to plot boxplots for motion and shape metrics calculated in make_shape_motion_df()
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell centroid as well shape and motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
#Outputs:
# figures of boxplots for D/T, Speed, Area, and solidity
def make_shape_motion_boxplots(data_sim, save_path):
    DT = []
    speed = []
    solidity = []
    area = []
    for df in data_sim:
        DT.append(df['DoverT'][0])
        speed.append(df['Speed'][0])
        solidity.append(df['Solidity'][0])
        area.append(df['Area'][0])

    sns.boxplot(data=DT)
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('D/T')
    plt.savefig(save_path+'/DT_boxplot.png')
    plt.clf()

    sns.boxplot(data=np.array(speed)/5) #because sampling rate is every 5 minutes
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('Speed $\mu m$/min')
    plt.savefig(save_path+'/Speed_boxplot.png')
    plt.clf()

    sns.boxplot(data=np.array(area))
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('Area') #$\mu m$^2')
    plt.savefig(save_path+'/Area_boxplot.png')
    plt.clf()

    sns.boxplot(data=np.array(solidity))
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('Area') #$\mu m$^2')
    plt.savefig(save_path+'/Solidity_boxplot.png')
    plt.clf()

###################################################################################################################################################

def plot_spatial_tension(tension, save_path):
    for l in tension:
        plt.plot(l)
    plt.xlabel('vertex number')
    plt.ylabel('tension (nN?)')
    plt.savefig(save_path+'/spatialtension.png')
    plt.clf()

###################################################################################################################################################

def plot_num_nn(num_nn, save_path):
    for l in num_nn:
        plt.plot(l)
    plt.xlabel('vertex number')
    plt.ylabel('Number of nearest neighbors protruding')
    plt.savefig(save_path+'/num_nn.png')
    plt.clf()

###################################################################################################################################################

def plot_omega_dir(Y, save_path):
    omega_over_time = []
    for i in range(len(Y)):
        omega_ti = Y[i][-1]
        omega_over_time.append(omega_ti%(2*np.pi))
    plt.plot(omega_over_time)
    plt.xlabel('time')
    plt.ylabel('direction of internal polarity')
    plt.savefig(save_path+'/pol_dir_over_time.png')
    plt.clf()

###################################################################################################################################################

def plot_bias_ang(bias_ang, save_path):
    plt.plot(bias_ang)
    plt.xlabel('time')
    plt.ylabel('direction of bias angle (true)')
    plt.savefig(save_path+'/biasangtrue_over_time.png')
    plt.clf()

###################################################################################################################################################

def plot_unwrapped_cellboundary(Y, N, save_path):
    total_dist_alltime = []
    spatial_loc_w_alltime = []
    for time_point in Y:
        pol_ang = time_point[-1]
        #find direction opposite to polarity angle (pol_ang)
        pol_ang_opp = pol_ang + np.pi
        x = time_point[0:N]
        y = time_point[N:N*2]
        dist_between_ver = []
        for i in range(len(x)):
            dist_between_ver.append(np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2))
        #find cumulative sum of distance between vertices to get "unwrapped" spatial map of cell shape
        total_dist = np.cumsum(dist_between_ver)
        total_dist_alltime.append(total_dist)

        #calculate cell centroid
        centroid_x = np.sum(x)/N
        centroid_y = np.sum(y)/N

        #get x and y compnents of the opposite direction polarity angle 
        x_w = centroid_x + np.cos(pol_ang_opp)
        y_w = centroid_y + np.sin(pol_ang_opp)

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
        C = [centroid_x, centroid_y]
        D = [x_w, y_w]

        #this is where the opposite polarity angle vector lies along cell boundary
        intersec = (line_intersection((A, B), (C, D)))
        #location of opp pol ang along cell boundary broken up into x and y terms
        w_x = intersec[0]
        w_y = intersec[1]

        #get location of opp pol ang in this unwrapped spatial map view
        if ang_dir > 0:
            dist = (np.sqrt((x2 - w_x)**2 + (y2 - w_y)**2))%total_dist[-1]
            spatial_loc_w = (total_dist[closest_ind - 1] + dist)%total_dist[-1]
        else:
            dist = (np.sqrt((x1 - w_x)**2 + (y1 - w_y)**2))%total_dist[-1]
            spatial_loc_w = (total_dist[closest_ind] + dist)%total_dist[-1]
        #spatial_loc_w_alltime.append(spatial_loc_w)
        spatial_loc_w_alltime.append(dist)
    
    for bound in total_dist_alltime:
        plt.plot(bound)
    plt.xlabel('vertex number')
    plt.ylabel('unwrapped view of cell boundary')
    plt.savefig(save_path+'/unwrapped_coords.png')
    plt.clf()

    plt.plot(spatial_loc_w_alltime)
    plt.xlabel('time')
    plt.ylabel('unwrapped coordinates of polarity angle')
    plt.savefig(save_path+'/unwrapped_coords_polang.png')
    plt.clf()

###################################################################################################################################################

def make_movie_polang(Y,t_end,dt,N,xy_min,xy_max,save_path):
    fig = plt.figure()
    ax = plt.axes(xlim=(xy_min, xy_max), ylim=(xy_min, xy_max))
    line1, = ax.plot([], [])
    line2, = ax.plot([], [])

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    def animate(i):
        x_ti = Y[i][0:N]
        y_ti = Y[i][N:N*2]

        w_ti = Y[i][-1]

        #calculate cell centroid
        centroid_x = np.sum(x_ti)/N
        centroid_y = np.sum(y_ti)/N

        x_ti, y_ti = getCoordToPlot(x_ti, y_ti, N)
        line1.set_data(x_ti, y_ti)
        line2.set_data([centroid_x, centroid_x+(10*np.cos(w_ti))], [centroid_y, centroid_y+(10*np.sin(w_ti))])
        return line1, line2

    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=int((t_end/dt)), interval=20, blit=True)

    FFwriter = FFMpegWriter(fps=20)
    anim.save(save_path + '/VBM_sim_movie.mp4', writer=FFwriter)