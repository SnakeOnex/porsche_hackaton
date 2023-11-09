import math


import numpy as np
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


def control(points,yaw,speed):
    g = 0.5
    target = get_lookahead_point(1.0, points)
    angle = np.atan2(target[1]-375, target[0]-700)
    angle = angle - yaw
    angle *= g
    angle = np.clip(angle, -1, 1)
    return angle
