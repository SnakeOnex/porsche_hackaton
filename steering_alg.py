import math


import numpy as np
from scipy import signal


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def get_lookahead_point(lookahead, path):
    """
    Given piecewise linear function and distance, returns a point that is that distance away on the line
    Args:
      lookahead - distance constant
      path - Nx2 numpy array
    Ret:
      target - 2D point
    """
    target = path[-1, :]

    cum_dist = 0.
    for i in range(0, path.shape[0] - 1):
        line = path[i + 1] - path[i]
        dist = np.linalg.norm(line)
        cum_dist += dist

        if cum_dist >= lookahead:
            last_piece_coef = (cum_dist - lookahead) / dist
            target = path[i + 1] - last_piece_coef * line
            break
    return target


def control(points,yaw,speed,yawr,yawr_b,dt):
    g = 1.9
    if points is None:
        return 0
    points = np.vstack(points).T
    #print(points.shape)
    point1 = points[(len(points)//2)+1][0]-680//2
    point2 = 480-points[(len(points)//2)+1][1]
    point11 = 0#points[(len(points)//4)][0]-680//2
    point22 = 0#480-points[(len(points)//4)][1]
    angle = np.arctan2(point2-point22,point1-point11)
    angle = angle - yaw - np.pi/2
    angle *= -g
    print(abs(yawr-yawr_b)/dt)
    if abs(yawr-yawr_b)/dt > 20:
      angle -= 0.05*np.deg2rad(yawr)
    #print(np.rad2deg(angle))

    angle = np.clip(angle, -1, 1)
    return angle
