import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull

#function to get smoothed velocity with rolling window size of 3 (either vx or vy)
#Inputs:
# pos => list of x positions or y positions of cell over time (type: list of floats)
#Output:
# vel => x or y velocity (depending on input) (type: pandas series of floats that is same length of input but first and last two entries are zeros due to rolling window average)
def calc_vel(pos):
    series = pd.Series(pos)
    series_smooth = series.rolling(3,center=True).mean()
    series_smooth_dx = series_smooth.diff()
    vel = series_smooth_dx.rolling(2).mean().shift(-1)
    vel[np.isnan(vel)] = 0
    return vel

###################################################################################################################################################

#function to calculate solidity of cell (ratio of area of cell to area of convex hull - measure of concavity so closer to 1 means less concave and more regular boundary)
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# dt => step size in solver (type: float)
# N => number of vertices that represents cell (type: int)
#Output:
# solidity => solidity for cell at each time point (note: this finds solidity for smoothed coordinates) (type: list of floats with length=t_end/(5/dt)) 
def calc_solidity(Y,dt,N):
  solidity = []
  for points in Y[::int(5/dt)]:
    x_points = points[0:N]
    y_points = points[N:N*2]

    xy = np.concatenate((np.reshape(x_points,(N,1)),np.reshape(y_points,(N,1))),axis=1)
    A_cell = Polygon(xy).area #cell area at current time point [um^2]

    #calculate convex hull from vertices of cell
    hull = ConvexHull(xy)

    #get coordinates of convex hull
    hull_shape = []
    for vertex in hull.vertices:
      hull_shape.append(xy[vertex])
    hull_shape=(np.reshape(hull_shape, (len(hull.vertices),2)))

    A_convexhull = Polygon(hull_shape).area

    solidity.append(A_cell/A_convexhull)

  return solidity

###################################################################################################################################################

#function to calculate area of cell
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# dt => step size in solver (type: float)
# N => number of vertices that represents cell (type: int)
#Output:
# area => area for cell at each time point (note: this finds area for smoothed coordinates) (type: list of floats with length=t_end/(5/dt)) 
def calc_area(Y,dt,N):
  area = []
  for points in Y[::int(5/dt)]:
    x_points = points[0:N]
    y_points = points[N:N*2]

    xy = np.concatenate((np.reshape(x_points,(N,1)),np.reshape(y_points,(N,1))),axis=1)
    A_cell = Polygon(xy).area #cell area at current time point [um^2]


    area.append(A_cell)

  return area

###################################################################################################################################################

#function to format track from 1 cell into a dataframe with shape and motion metrics
#Inputs:
# Y => x and y positions for each vertex for each time point with (type: list of ndarrays of floats; list is shape=(len(T),N) and arrays are len=N*2)
# dt => step size in solver (type: float)
# N => number of vertices that represents cell (type: int)
#Output:
# onewalker_df => dataframe for one cell track that includes x and y coordinates of centroid (sampled every 5 min), x velocity, y velocity, and magnitude of velocity, D/T (net displacement over total distance traveled),
#speed (total distance traveled/over time of sim), solidity, and area (type: pandas dataframe)
def make_shape_motion_df(Y, dt, N):
    centroid_x = []
    centroid_y = []
    for time_point in Y:
        centroid_x.append(np.sum(time_point[0:N])/N)
        centroid_y.append(np.sum(time_point[N:N*2])/N)

    x_centroid_smooth = centroid_x[::int(5/dt)]
    y_centroid_smooth = centroid_y[::int(5/dt)]

    stepsize = np.sqrt((np.array(x_centroid_smooth[0:-1])-np.array(x_centroid_smooth[1:]))**2 +(np.array(y_centroid_smooth[0:-1])-np.array(y_centroid_smooth[1:]))**2)
    r = np.sqrt(np.array(x_centroid_smooth)**2 + np.array(y_centroid_smooth)**2)
    net_disp = abs(r[-1] - r[0])

    speed = np.sum(stepsize)/(1)/(len(x_centroid_smooth)-1)

    solidity = calc_solidity(Y,dt,N)

    area = calc_area(Y,dt,N)

    x_vel = calc_vel(x_centroid_smooth)
    y_vel = calc_vel(y_centroid_smooth)
    vel = np.sqrt(x_vel**2 + y_vel**2)

    onewalker = {'x': x_centroid_smooth, 'y': y_centroid_smooth, 'vx': x_vel, 'vy': y_vel, 'v': vel, 'DoverT': net_disp/np.sum(stepsize), 'Speed': speed, 'Solidity': solidity, 'Area': area}
    onewalker_df = pd.DataFrame(data=onewalker)

    return onewalker_df

###################################################################################################################################################

#function that makes a dataframe from x velocity and y velocity (outputs from calc_vel()) and repeats them 
# (ex. if input is series 'A' and series 'B', the dataframe would have columns: 'A' 'B' 'A' 'B') this is used when calculating the ACF
#Inputs:
# vx => x component of velocity (type: pandas series of floats)
# vy => y componenet of velocity (type: pandas series of floats)
#Output:
# comnined => dataframe of x velocity and y velocity columns repeated twice
def make_comb_df(vx, vy):
    d = {'vx': vx, 'vy': vy}
    vel_df=pd.DataFrame(data=d)
    combined = pd.concat([vel_df[['vx','vy']].reset_index(drop=True), vel_df[['vx','vy']].reset_index(drop=True)], axis = 1 )
    return combined

###################################################################################################################################################

#function that calculates the autocorrelation function for the velocity vector (E[X_(t)X_(t+tau)])
#Inputs:
# dfraw => dataframe of vx and vy repeated twice as columns (output from make_comb_df()) (type: pandas dataframe)
# Optional:
# min_track_length => specified track length to cut off autocorrelation results (what value of tau to cut off at)
#Outputs:
# 1st output => autocorrelation(tau) with positive lags divided by first value to "normalize" and force first value to be 1 (type: ndarray of floats with length=min_track_length-4)
# 2nd output => number of time points used when computing the autocorrelaton - note: as tau increases, this number will decrease and as tau->length(vx), Nposlags->1 (type: ndarray of ints with length=min_track_length-4)
# 3rd output => autocorrelation(-tau) with negative lags divided by first value to "normalize" and force first value to be 1 (type: ndarray of floats with length=min_track_length-4)
# 4th output => number of time points used when computing the negative autocorrelaton - note: as |tau| increases, this number will decrease and as |tau|->length(vx), Nneglags->1 (type: ndarray of ints with length=min_track_length-4)
def xcorr_vector(dfraw, min_track_length=30):
  df = dfraw.dropna()
  v1x = np.asarray(df.iloc[:,0])
  v1y = np.asarray(df.iloc[:,1])
  v2x = np.asarray(df.iloc[:,2])
  v2y = np.asarray(df.iloc[:,3])

  length = len(df)
  poslagsmean=[]
  neglagsmean=[]
  Nposlags=[]
  Nneglags=[]
  for lag in range(length):
    poslags =  v2x[lag:length]*v1x[0:length-lag] + v2y[lag:length]*v1y[0:length-lag] #dot product (vx_(t)vx_(t+tau) + vy_(t)vy_(t+tau))
    neglags =  v2x[0:length-lag]*v1x[lag:length] + v2y[0:length-lag]*v1y[lag:length]
    poslagsmean.append(np.nanmean(poslags))
    neglagsmean.append(np.nanmean(neglags))
    Nposlags.append(sum(~np.isnan(poslags)))
    Nneglags.append(sum(~np.isnan(neglags)))

  return np.asarray(poslagsmean[0:min_track_length-4])/poslagsmean[0], np.asarray(Nposlags[0:min_track_length-4]), np.asarray(neglagsmean[0:min_track_length-4])/neglagsmean[0], np.asarray(Nneglags[0:min_track_length-4])